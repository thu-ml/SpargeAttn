"""S5 gate benchmark: topk=0.5 sparse vs dense SDPA wall-clock.

Compares spas_sage2_attn_meansim_topk_cuda(topk=0.5) vs torch SDPA
at pinned shape b=1, h=32, seq 4096/8192/16384/32768, hd 64/128, causal T/F.
QA: topk=0.5 >= 1.3x dense SDPA wall-clock at seq>=8192 (S5 gate).
"""
import os
import time
import json
import torch
import torch.nn.functional as F

from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
from spas_sage_attn.core import get_cuda_arch_versions

torch.manual_seed(0)

SHAPES = [
    (4096, 64, False),
    (4096, 128, False),
    (4096, 64, True),
    (4096, 128, True),
    (8192, 64, False),
    (8192, 128, False),
    (8192, 64, True),
    (8192, 128, True),
    (16384, 64, False),
    (16384, 128, False),
    (16384, 64, True),
    (16384, 128, True),
    (32768, 64, False),
    (32768, 128, False),
    (32768, 64, True),
    (32768, 128, True),
]
B, H, TOPK = 1, 32, 0.5
WARMUP, ITERS = 3, 5


def _time_sparse(q, k, v, is_causal, sparse_fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        sparse_fn(q, k, v, topk=TOPK, is_causal=is_causal)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        sparse_fn(q, k, v, topk=TOPK, is_causal=is_causal)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0


def _time_dense(q, k, v, is_causal, dense_fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        dense_fn(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dense_fn(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0


def run_s5(out_path="bench/results/sm120_s5.json"):
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
        # sparse
        ms_sparse = _time_sparse(q, k, v, causal, spas_sage2_attn_meansim_topk_cuda)
        tflops_sparse = 4.0 * B * H * seq * seq * hd / (ms_sparse * 1e9)
        # dense SDPA baseline (same dtype)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        ms_dense = _time_dense(q, k, v, causal, lambda *a, **kw: F.scaled_dot_product_attention(*a, **kw))
        tflops_dense = 4.0 * B * H * seq * seq * hd / (ms_dense * 1e9)
        speedup = ms_dense / ms_sparse
        entry = {
            "seq": seq, "hd": hd, "causal": causal,
            "ms_sparse": round(ms_sparse, 3), "tflops_sparse": round(tflops_sparse, 2),
            "ms_dense_sdpa": round(ms_dense, 3), "tflops_dense_sdpa": round(tflops_dense, 2),
            "speedup_sparse_over_dense": round(speedup, 3),
            "nsm": nsm,
            "tflops_per_smf_sparse": round(tflops_sparse / nsm, 3),
        }
        results.append(entry)
        print(f"  seq={seq:6d} hd={hd:3d} causal={str(causal):5s} "
              f"sparse={ms_sparse:8.3f}ms ({tflops_sparse:8.2f} TFLOPS) "
              f"dense_sdpa={ms_dense:8.3f}ms ({tflops_dense:8.2f} TFLOPS) "
              f"speedup={speedup:.3f}x")
    # S5 gate: topk=0.5 >= 1.3x dense SDPA wall-clock at seq>=8192
    # (sparse attention has fixed overhead that dominates at seq=4096;
    #  this is expected sparse-attention behavior, not a bug)
    s5_results = [r for r in results if r["seq"] >= 8192]
    s5_pass = all(r["speedup_sparse_over_dense"] >= 1.3 for r in s5_results)
    min_speedup = min(r["speedup_sparse_over_dense"] for r in s5_results) if s5_results else float("inf")
    payload = {"arch": arch, "nsm": nsm, "topk": TOPK, "results": results, "s5_gate": {"pass": s5_pass, "min_speedup": round(min_speedup, 3), "shapes_checked": [(r["seq"], r["hd"], r["causal"]) for r in s5_results]}}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nS5 gate (topk={TOPK} >= 1.3x dense SDPA at seq>=8192): {'PASS' if s5_pass else 'FAIL'} (min speedup={min_speedup:.3f}x)")
    print(f"Wrote {out_path}")
    return payload


if __name__ == "__main__":
    run_s5()
