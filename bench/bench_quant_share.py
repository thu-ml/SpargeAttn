"""Profile quant-kernel share of topk=0.5 sparse attention wall-clock.

Times each stage of the sparse pipeline independently:
  1. get_block_map_meansim_fuse_quant(q, k, km)  (q/k quant + LUT)
  2. fused.transpose_pad_permute_cuda(v)         (v transpose/pad)
  3. fused.scale_fuse_quant_cuda(v_fp8, v_scale) (v quant)
  4. qk_int8_sv_f8_accum_f16_...                 (attention kernel)

QA: quant-kernel share (stages 1-3) of topk=0.5 wall-clock <= 25%.
"""
import os
import json
import torch
import torch.nn.functional as F

from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
from spas_sage_attn.core import get_cuda_arch_versions
from spas_sage_attn.utils import get_block_map_meansim_fuse_quant
from spas_sage_attn.utils import hyperparameter_check
import spas_sage_attn._fused as fused
import spas_sage_attn._qattn as qattn

torch.manual_seed(0)

# S3-equivalent point + a couple representative shapes
SHAPES = [
    (32768, 128, False),
    (16384, 128, False),
    (16384, 64, False),
]
B, H, TOPK = 1, 32, 0.5
WARMUP, ITERS = 3, 5


def _time_event(fn, *args, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn(*args)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def run_profile(out_path="bench/results/sm120_quant_share.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arch = get_cuda_arch_versions()[0]
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"arch={arch} SMs={nsm}")
    results = []

    for seq, hd, causal in SHAPES:
        torch.manual_seed(0)
        q = torch.randn(B, H, seq, hd, dtype=torch.float16, device="cuda") * 0.5
        k = torch.randn(B, H, seq, hd, dtype=torch.float16, device="cuda") * 0.5
        v = torch.randn(B, H, seq, hd, dtype=torch.float16, device="cuda") * 0.5
        km = k.mean(dim=-2, keepdim=True)

        # Pre-allocate buffers used by the pipeline (same as core.py)
        headdim = q.size(-1)
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        o = torch.empty_like(q)
        pvthreshd = hyperparameter_check(50, q.size(-3), q.device)
        scale = 1.0 / (headdim ** 0.5)

        # ---- Stage 1: q/k quant + LUT (Triton) ----
        def stage1():
            return get_block_map_meansim_fuse_quant(
                q, k, km, is_causal=causal, simthreshd1=-0.1, cdfthreshd=None,
                topk=TOPK, return_lut=True, attention_sink=False, BLKQ=128, BLKK=64)

        ms_lut = _time_event(stage1)
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = stage1()

        # ---- Stage 2: v transpose/pad ----
        def stage2():
            fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        ms_vtrans = _time_event(stage2)

        # ---- Stage 3: v quant ----
        def stage3():
            fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)
        ms_vquant = _time_event(stage3)

        # ---- Stage 4: attention kernel (Sage2++ fp16-acc PV) ----
        def stage4():
            qattn.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd,
                q_scale, k_scale, v_scale, 1, int(causal), 1, scale, 0)
        ms_attn = _time_event(stage4)

        # ---- Full end-to-end (topk=0.5) for reference ----
        ms_sparse = _time_event(lambda: spas_sage2_attn_meansim_topk_cuda(
            q, k, v, topk=TOPK, is_causal=causal))

        quant_time = ms_lut + ms_vtrans + ms_vquant
        quant_share = quant_time / ms_sparse * 100.0 if ms_sparse else float("inf")

        entry = {
            "seq": seq, "hd": hd, "causal": causal,
            "ms_lut": round(ms_lut, 3),
            "ms_vtrans": round(ms_vtrans, 3),
            "ms_vquant": round(ms_vquant, 3),
            "ms_quant_total": round(quant_time, 3),
            "ms_attn": round(ms_attn, 3),
            "ms_sparse_e2e": round(ms_sparse, 3),
            "quant_share_pct": round(quant_share, 1),
            "nsm": nsm,
        }
        results.append(entry)
        print(f"  seq={seq:6d} hd={hd:3d} causal={str(causal):5s}  "
              f"lut={ms_lut:7.2f}  vtrans={ms_vtrans:6.2f}  vquant={ms_vquant:6.2f}  "
              f"quant_total={quant_time:7.2f}  attn={ms_attn:7.2f}  "
              f"sparse_e2e={ms_sparse:7.2f}  quant_share={quant_share:5.1f}%")

    quant_pass = all(r["quant_share_pct"] <= 25.0 for r in results)
    max_share = max(r["quant_share_pct"] for r in results)
    payload = {"arch": arch, "nsm": nsm, "topk": TOPK, "results": results,
               "quant_share_gate": {"pass": quant_pass, "max_share_pct": round(max_share, 1),
                                    "threshold_pct": 25.0}}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nQuant-kernel share gate (<=25%): {'PASS' if quant_pass else 'FAIL'} (max={max_share:.1f}%)")
    print(f"Wrote {out_path}")
    return payload


if __name__ == "__main__":
    run_profile()
