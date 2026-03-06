"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from .utils import hyperparameter_check, get_block_map_meansim, get_block_map_meansim_fuse_quant, get_vanilla_qk_quant, block_map_lut_triton
from .quant_per_block import per_block_int8, per_warp_int8
from einops import rearrange

import spas_sage_attn._qattn as qattn
import spas_sage_attn._fused as fused

SAGE2PP_ENABLED = True
try:
    from spas_sage_attn._qattn import qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
except:
    print("Warning: Sage2++ NOT enabled")
    SAGE2PP_ENABLED = False

# Detect ROCm
IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None

def get_gpu_arch_info():
    """
    Get GPU architecture info that works on both CUDA and ROCm.
    Returns: (arch_string, is_rocm, is_rdna, is_mi_series)
    """
    if not torch.cuda.is_available():
        return None, False, False, False
    
    props = torch.cuda.get_device_properties(0)
    if IS_ROCM:
        # ROCm: use gcnArchName
        arch = props.gcnArchName.split(':')[0] if hasattr(props, 'gcnArchName') else "gfx1100"
        is_rdna = arch.startswith("gfx10") or arch.startswith("gfx11")
        is_mi_series = arch.startswith("gfx9")
        return arch, True, is_rdna, is_mi_series
    else:
        # CUDA: compute capability
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm{major}{minor}", False, False, False

def get_cuda_arch_versions():
    """
    Get architecture strings for all GPUs.
    On CUDA: returns ["sm80", "sm90", etc.]
    On ROCm: returns ["gfx1100", "gfx1151", etc.]
    """
    archs = []
    for i in range(torch.cuda.device_count()):
        if IS_ROCM:
            props = torch.cuda.get_device_properties(i)
            arch = props.gcnArchName.split(':')[0] if hasattr(props, 'gcnArchName') else "gfx1100"
            archs.append(arch)
        else:
            major, minor = torch.cuda.get_device_capability(i)
            archs.append(f"sm{major}{minor}")
    return archs

def is_sm90_or_equivalent(arch):
    """Check if architecture supports SM90-level features (wgmma, etc.)"""
    if IS_ROCM:
        # On ROCm, MI series (gfx9xx) has similar capabilities
        return arch.startswith("gfx9")
    else:
        return arch == "sm90"

def supports_fp8(arch):
    """Check if architecture supports FP8."""
    if IS_ROCM:
        # Only MI series supports FP8 on ROCm
        return arch.startswith("gfx9")
    else:
        # CUDA: sm89+ supports FP8
        try:
            sm_num = int(arch[2:])
            return sm_num >= 89
        except:
            return False

@torch.compiler.disable
def spas_sage2_attn_meansim_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    min_seq = 64 if IS_ROCM else 128
    assert q.size(-2)>=min_seq, f"seq_len should be not less than {min_seq}."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)

    arch = get_cuda_arch_versions()[q.device.index]
    
    # Check if this architecture supports FP8 (required for sage2)
    if not supports_fp8(arch):
        raise RuntimeError(
            f"spas_sage2_attn_meansim_cuda requires FP8 support, but {arch} does not support FP8. "
            f"On RDNA GPUs (gfx10xx/gfx11xx), use spas_sage_attn_meansim_cuda instead."
        )
    
    # Choose block sizes based on architecture
    # For ROCm, use CTA_Q=64, CTA_K=64 to fit shared memory constraints
    if is_sm90_or_equivalent(arch):
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=128)
    elif IS_ROCM:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=64)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=128, BLKK=64)

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    o = torch.empty_like(q)

    # sm80/sm86/sm87 don't support FP8, use FP16 kernel directly
    if arch in ("sm80", "sm86", "sm87"):
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0
        )
    else:
        # sm89+ supports FP8, quantize V
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

        if arch == "sm90":
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        elif SAGE2PP_ENABLED:
            qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        else:
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)

    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o

@torch.compiler.disable
def spas_sage2_attn_meansim_topk_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=-0.1, cdfthreshd=None, topk=0.5, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    min_seq = 64 if IS_ROCM else 128
    assert q.size(-2)>=min_seq, f"seq_len should be not less than {min_seq}."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)

    arch = get_cuda_arch_versions()[q.device.index]
    
    # Check if this architecture supports FP8 (required for sage2)
    if not supports_fp8(arch):
        raise RuntimeError(
            f"spas_sage2_attn_meansim_topk_cuda requires FP8 support, but {arch} does not support FP8. "
            f"On RDNA GPUs (gfx10xx/gfx11xx), use spas_sage_attn_meansim_topk_cuda instead."
        )
    
    if is_sm90_or_equivalent(arch):
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=128)
    elif IS_ROCM:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=64)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=128, BLKK=64)

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    o = torch.empty_like(q)

    # sm80/sm86/sm87 don't support FP8, use FP16 kernel directly
    if arch in ("sm80", "sm86", "sm87"):
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0
        )
    else:
        # sm89+ supports FP8, quantize V
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

        if arch == "sm90":
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        elif SAGE2PP_ENABLED:
            qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        else:
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)

    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o

@torch.compiler.disable
def block_sparse_sage2_attn_cuda(q, k, v, mask_id=None, dropout_p=0.0, scale=None, smooth_k=True, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    min_seq = 64 if IS_ROCM else 128
    assert q.size(-2)>=min_seq, f"seq_len should be not less than {min_seq}."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)
    
    arch = get_cuda_arch_versions()[q.device.index]
    
    # Check if this architecture supports FP8 (required for sage2)
    if not supports_fp8(arch):
        raise RuntimeError(
            f"block_sparse_sage2_attn_cuda requires FP8 support, but {arch} does not support FP8. "
            f"On RDNA GPUs (gfx10xx/gfx11xx), use spas_sage_attn_meansim_cuda instead."
        )
    
    if is_sm90_or_equivalent(arch):
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 64, 128)
    elif IS_ROCM:
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 64, 64)
    else:
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 128, 64)
    lut, valid_block_num = block_map_lut_triton(block_map=mask_id)
    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    o = torch.empty_like(q)

    # sm80/sm86/sm87 don't support FP8, use FP16 kernel directly
    if arch in ("sm80", "sm86", "sm87"):
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0
        )
    else:
        # sm89+ supports FP8, quantize V
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 448.0, 1)

        if arch == "sm90":
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        else:
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)

    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    if return_sparsity:
        qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o

@torch.compiler.disable
def spas_sage_attn_meansim_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    
    headdim = q.size(-1)
    # min_seq depends on CTA_Q which depends on headdim for ROCm
    if IS_ROCM:
        min_seq = 32 if headdim == 128 else 64
    else:
        min_seq = 128
    assert q.size(-2)>=min_seq, f"seq_len should be not less than {min_seq}."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    # Always compute km for block map generation
    km = k.mean(dim=-2, keepdim=True)
    # if smooth_k:
    #     k = k - km

    # For ROCm, use smaller tiles for headdim=128 to reduce register pressure:
    # - headdim=64: CTA_Q=64, CTA_K=64
    # - headdim=128: CTA_Q=32, CTA_K=16 (best performance at 10% sparsity)
    if IS_ROCM:
        blkq = 32 if headdim == 128 else 64
        blkk = 16 if headdim == 128 else 64
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=blkq, BLKK=blkk)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink) 

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    _is_causal = 1 if is_causal else 0
    o = torch.empty(q.shape, dtype=v.dtype, device=q.device)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, scale, 0)
    # Sync to ensure kernel completes before subsequent operations (helps with ROCm resource management)
    if IS_ROCM:
        torch.cuda.synchronize()
    # Convert output back to original dtype if needed
    if o.dtype != dtype:
        o = o.to(dtype)
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o

@torch.compiler.disable
def spas_sage_attn_meansim_topk_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=-0.1, cdfthreshd=None, topk=0.5, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    
    headdim = q.size(-1)
    # min_seq depends on CTA_Q which depends on headdim for ROCm
    if IS_ROCM:
        min_seq = 32 if headdim == 128 else 64
    else:
        min_seq = 128
    assert q.size(-2)>=min_seq, f"seq_len should be not less than {min_seq}."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)

    # For ROCm, use smaller tiles for headdim=128 to reduce register pressure:
    # - headdim=64: CTA_Q=64, CTA_K=64
    # - headdim=128: CTA_Q=32, CTA_K=16 (best performance at 10% sparsity)
    if IS_ROCM:
        blkq = 32 if headdim == 128 else 64
        blkk = 16 if headdim == 128 else 64
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=blkq, BLKK=blkk)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink) 

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    _is_causal = 1 if is_causal else 0
    o = torch.empty(q.shape, dtype=v.dtype, device=q.device)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, scale, 0)
    # Sync to ensure kernel completes before subsequent operations (helps with ROCm resource management)
    if IS_ROCM:
        torch.cuda.synchronize()
    # Convert output back to original dtype if needed
    if o.dtype != dtype:
        o = o.to(dtype)
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o
