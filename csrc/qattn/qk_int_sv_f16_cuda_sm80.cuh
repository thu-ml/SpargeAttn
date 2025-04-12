/*
 * Copyright (c) 2025 by SpargeAttn team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

#include "../cp_async.cuh"
#include "../mma.cuh"
#include "../permuted_smem.cuh"
#include "../math.cuh"
#include "../reduction_utils.cuh"
#include "attn_utils.cuh"

#define PACK_SIZE_QK 16 // as if it is int8
#define PACK_SIZE_V 8   // fp16
#define PACK_SIZE_O 8   // fp16

// treat as if int8 tensor core
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

// fp16 tensor core
#define MMA_SV_M 16
#define MMA_SV_N 16
#define MMA_SV_K 16

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, DataType DTypeQK, QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
        bool use_inst_buffer = false, PVThresholdMode pv_threashold_mode, typename DTypeOut = half, ComputeUnit DenominatorAccumUnit, MaskMode mask_mode = MaskMode::kNone, bool return_pv_count = false>
__global__ void qk_int_sv_f16_block_sparse_attn_kernel(int8_t *__restrict__ Q, int8_t *__restrict__ K, half *__restrict__ V, DTypeOut *__restrict__ O, int32_t *__restrict__ PV_Count, int32_t *__restrict__ Lut, int32_t *__restrict__ Valid_Block_Num, float *__restrict__ PV_Threshold,
                      float *__restrict__ Q_scale, float *__restrict__ K_scale,
                      const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                      const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
                      const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
                      const uint32_t stride_bz_v, const uint32_t stride_seq_v, const uint32_t stride_h_v,
                      const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
                      float sm_scale)
{
  // compile time check
  static_assert(DTypeQK == DataType::kInt8 || DTypeQK == DataType::kInt4, "DTypeQK must be int8 or int4");
  static_assert(Q_GRAN == QuantGranularity::kPerBlock || Q_GRAN == QuantGranularity::kPerWarp || Q_GRAN == QuantGranularity::kPerThread, "Q_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp || K_GRAN == QuantGranularity::kPerThread, "K_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(std::is_same<DTypeOut, half>::value || std::is_same<DTypeOut, nv_bfloat16>::value, "DTypeOut must be half or nv_bfloat16");
  static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
  static_assert(CTA_Q / CTA_K <= 2); // for efficient causal implementation

  using DTypeOut2 = typename std::conditional<std::is_same<DTypeOut, half>::value, half2, nv_bfloat162>::type;

  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = CTA_K / WARP_K;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = (DTypeQK == DataType::kInt8) ? (head_dim / MMA_QK_K) : (head_dim / 2 / MMA_QK_K);
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;

  constexpr uint32_t QK_SMEM_STRIDE = (DTypeQK == DataType::kInt8) ? (head_dim) : (head_dim / 2);
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  constexpr uint32_t V_SMEM_STRIDE = head_dim;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  // maximize L2 hit rate
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;

  // transfer to base 2 instead of base e with better numerical efficiency
  sm_scale *= math::log2e;

  float pv_threshold;
  int pv_count = 0;

  if constexpr (pv_threashold_mode != PVThresholdMode::kNone)
  {
    pv_threshold = PV_Threshold[head_id];
  }

  // RS holds the fragment of S
  int32_t RS[num_tiles_q][num_tiles_k][8];
  float RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2]; // max
  float d[num_tiles_q][2]; // denominator

  uint32_t q_scale_idx, k_scale_idx;

  if constexpr (Q_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_q = gridDim.x;
    q_scale_idx = batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * num_warps_q + get_warp_idx_q<num_warps_q, num_warps_k>();
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) + bx * (num_warps_q * 8) + get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
  }

  if constexpr (K_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k + (head_id / num_kv_groups) * num_block_k;
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_warp_block_k + (head_id / num_kv_groups) * num_warp_block_k + get_warp_idx_k<num_warps_q, num_warps_k>();
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) + (head_id / num_kv_groups) * (num_warp_block_k * 4) + get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
  }

  constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerBlock) ? 1 : (K_GRAN == QuantGranularity::kPerWarp) ? (CTA_K / WARP_K) : (CTA_K / WARP_K) * 4;

  // initialize o, m, d
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {        
        RO[fq][fv][k] = 0.0f;
      }
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      m[fq][k] = -5000000.0f;
      d[fq][k] = 1.0f;
    }
  }

  constexpr uint32_t K_smem_idx_offset = CTA_Q;
  constexpr uint32_t V_smem_idx_offset = CTA_Q + CTA_K;

  constexpr SwizzleMode swizzle_mode_QK = (QK_SMEM_STRIDE == 32) ? SwizzleMode::k32B : (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_V = (V_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_O = (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2 : (QK_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_QK = (QK_SMEM_STRIDE == 32) ? 16 : (QK_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_V = (V_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_V = (V_SMEM_STRIDE == 32) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_O = (O_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_O = (O_SMEM_STRIDE == 32) ? 8 : 4;

  constexpr uint32_t QK_smem_iters_row = QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row = V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
  constexpr uint32_t V_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row = O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

  int8_t *Q_lane_base_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_q + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  int8_t *K_lane_base_ptr = K + batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_k + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  half *V_lane_base_ptr = V + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V) * stride_seq_v + (lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V;
  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t K_smem_offset_load = smem_K.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t V_smem_offset_load = smem_V.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col + lane_id / global_to_shared_line_lanes_V, lane_id % global_to_shared_line_lanes_V);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16, lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 16, lane_id / 16);

  // for causal masking
  uint32_t Q_idx_lane_base = bx * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
  uint32_t K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);

  // for loading
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t V_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_V;

  // read from Valid_Block_Num to get how much iterations we need to do
  const uint32_t num_block_q = gridDim.x;
  const uint32_t num_iterations = Valid_Block_Num[batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx];

  if (num_iterations == 0)
  {
    return;
  }

  // move Lut to the correct place
  const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
  Lut += batch_id * num_qo_heads * num_block_q * num_block_k + head_id * num_block_q * num_block_k + bx * num_block_k;

  // load Q with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
    Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  uint32_t KV_block_increm = 0;

  // load K with predicate
  KV_block_increm = Lut[0];
  K_lane_base_ptr += KV_block_increm * CTA_K * stride_seq_k;
  K_load_idx_lane_base += KV_block_increm * CTA_K;
  K_idx_lane_base += KV_block_increm * CTA_K;
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
    K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  float q_scale = Q_scale[q_scale_idx];

  float original_sm_scale = sm_scale;
  k_scale_idx += KV_block_increm * k_scale_advance_offset;
  float dequant_scale = q_scale * K_scale[k_scale_idx];

  sm_scale = original_sm_scale * dequant_scale;

  // load V with predicate
  V_lane_base_ptr += KV_block_increm * CTA_K * stride_seq_v;
  V_load_idx_lane_base += KV_block_increm * CTA_K;
  load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
    V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
  cp_async::commit_group();

#pragma unroll
  for (uint32_t iter = 1; iter < num_iterations - 1; iter++)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
      smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
        }
      }
    }

    if constexpr (pv_threashold_mode != PVThresholdMode::kNone) // use pv_threshold to skip unnecessary computation
    {
      __syncthreads();

      // first issue the load for K
      KV_block_increm = Lut[iter];
      K_lane_base_ptr += KV_block_increm * CTA_K * stride_seq_k;
      K_load_idx_lane_base += KV_block_increm * CTA_K;
      K_idx_lane_base += KV_block_increm * CTA_K;
      load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
        K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
      cp_async::commit_group();
  
      float local_max_diff = update_mo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, sm_scale);

      // reduce max diff in a warp
      local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x4));
      local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x8));
      local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x10));

      if constexpr (pv_threashold_mode == PVThresholdMode::kPerBlock)
      {
        // reduce max diff in a block
        static __shared__ float reduced_buffer[num_warps * 32];
        reduced_buffer[lane_id + warp_id * 32] = local_max_diff;
        __syncthreads();

        if constexpr (num_warps == 4)
        {
          local_max_diff = reduced_buffer[(lane_id % 4) * 32 + lane_id];
          local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x1));
          local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x2));
        }
        else if constexpr (num_warps == 8)
        {
          local_max_diff = reduced_buffer[(lane_id % 8) * 32 + lane_id];
          local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x1));
          local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x2));
          local_max_diff = max(local_max_diff, __shfl_xor_sync(0xffffffff, local_max_diff, 0x4));
        }
        else
        {
          static_assert(num_warps == 4 || num_warps == 8, "num_warps must be 4 or 8");
        }
      }

      // ensure V is ready
      cp_async::wait_group<1>();
      __syncthreads();

      // skip the computation on warp level
      if (local_max_diff + pv_threshold > 0)
      {

        if constexpr (return_pv_count)
        {
          pv_count++;
        }

        exponentiate_r<num_tiles_q, num_tiles_k, false>(RS_f32, m, sm_scale);

        if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
        {
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
        }

        uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
        RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

        if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
        {
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
        }

        if constexpr (!use_inst_buffer)
        {
          compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
            smem_V, RS_f16, RO, d, V_smem_offset_mma);
        }
        else
        {
          compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
            smem_V, RS_f16, RO, d, V_smem_offset_mma); 
        }
      }
    }
    else // if we don't use pv_threshold, we just do the computation
    {

      if constexpr (return_pv_count)
      {
        pv_count++;
      }
      
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, sm_scale);

      if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
      {
        accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
      }

      uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
      RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

      if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
      {
        accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
      }

      __syncthreads();

      // load K
      KV_block_increm = Lut[iter];
      K_lane_base_ptr += KV_block_increm * CTA_K * stride_seq_k;
      K_load_idx_lane_base += KV_block_increm * CTA_K;
      K_idx_lane_base += KV_block_increm * CTA_K;
      load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
        K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
      cp_async::commit_group();

      // ensure V is ready
      cp_async::wait_group<1>();
      __syncthreads();

      if constexpr (!use_inst_buffer)
      {
        compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
          smem_V, RS_f16, RO, d, V_smem_offset_mma);
      }
      else
      {
        compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
          smem_V, RS_f16, RO, d, V_smem_offset_mma); 
      }
    }
    
    __syncthreads();

    // load V
    V_lane_base_ptr += KV_block_increm * CTA_K * stride_seq_v;
    V_load_idx_lane_base += KV_block_increm * CTA_K;
    load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V);
    cp_async::commit_group();

    k_scale_idx += KV_block_increm * k_scale_advance_offset;
    dequant_scale = q_scale * K_scale[k_scale_idx];
    sm_scale = original_sm_scale * dequant_scale;
  }
  
  // second last iter, apply causal mask
  if (num_iterations > 1)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
      smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    __syncthreads();

    // load K with predicate
    KV_block_increm = Lut[num_iterations - 1];
    K_lane_base_ptr += KV_block_increm * CTA_K * stride_seq_k;
    K_load_idx_lane_base += KV_block_increm * CTA_K;
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    K_idx_lane_base += KV_block_increm * CTA_K;

    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, original_sm_scale);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
    }

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    if constexpr (return_pv_count)
    {
      pv_count++;
    }

    if constexpr (!use_inst_buffer)
    {
      compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }
    else
    {
      compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }

    __syncthreads();
    // load V with predicate
    V_lane_base_ptr += KV_block_increm * CTA_K * stride_seq_v;
    V_load_idx_lane_base += KV_block_increm * CTA_K;
    load_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      V_lane_base_ptr, V_smem_offset_load, stride_seq_v, smem_V, V_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    k_scale_idx += KV_block_increm * k_scale_advance_offset;
    dequant_scale = q_scale * K_scale[k_scale_idx];
    sm_scale = original_sm_scale * dequant_scale;

  }

  // last iter, apply causal mask and out of bound mask
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
      smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
            RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    // check out of bound in the last iter
    apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);

    update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(RS_f32, RO, m, d, original_sm_scale);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
    }

    uint32_t RS_f16[num_tiles_q][num_tiles_k][4];
    RS_32_to_16<num_tiles_q, num_tiles_k>(RS_f32, RS_f16);

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kTensorCore>(RS_f16, d);
    }

    // ensure V is ready
    cp_async::wait_group<0>();
    __syncthreads();

    if constexpr (return_pv_count)
    {
      pv_count++;
    }

    if constexpr (!use_inst_buffer)
    {
      compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }
    else
    {
      compute_fp16_sv_permuted_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
        smem_V, RS_f16, RO, d, V_smem_offset_mma);
    }

    __syncthreads();

  }

  if constexpr (return_pv_count)
  {

    if (lane_id == 0)
    {
      PV_Count[batch_id * num_qo_heads * num_block_q * num_warps_q + head_id * num_block_q * num_warps_q + bx * num_warps_q + get_warp_idx_q<num_warps_q, num_warps_k>()] = pv_count;
    }

    __syncthreads();
  }

  normalize_d<num_tiles_q, num_tiles_v, DenominatorAccumUnit>(RO, m, d);

  // save the result to shared memory
  uint32_t smem_O_row_base = get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));

      // convert RO to half
      uint32_t RO_f16[4];
#pragma unroll
      for (uint32_t k = 0; k < 4; k++)
      { 
        if constexpr (std::is_same<DTypeOut, half>::value)
        {
          ((half2*)RO_f16)[k] = __float22half2_rn(((float2*)RO[fq][fv])[k]);
        }
        else if constexpr (std::is_same<DTypeOut, nv_bfloat16>::value)
        {
          ((nv_bfloat162*)RO_f16)[k] = __float22bfloat162_rn(((float2*)RO[fq][fv])[k]);
        }
      }

      ((int32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
      ((int32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];

      // ! permuted, make sure you know what you are doing
      ((int32_t*)(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = RO_f16[2];
      ((int32_t*)(smem_O.base + (offset_O ^ 0x1) + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
    }
  }

  // ! do we need to sync here?
  __syncwarp();

  // shared memory to global memory
  DTypeOut *O_lane_ptr = O + batch_id * stride_bz_o + head_id * stride_h_o + (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() + lane_id / global_to_shared_line_lanes_O) * stride_seq_o + lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / global_to_shared_line_lanes_O, lane_id % global_to_shared_line_lanes_O);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_O;

#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++)
    {
      if (O_load_idx_lane_base < qo_len)
      {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.advance_offset_by_column<global_to_shared_line_lanes_O>(offset_O);
    }

    offset_O = smem_O.advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(offset_O - (O_smem_iters_row * global_to_shared_line_lanes_O));
    O_lane_ptr += ((global_to_shared_copy_lines_per_warp_O * stride_seq_o) - (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
  }
}


template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, uint32_t qk_quant_gran, typename DTypePVAccum, bool use_inst_buffer, uint32_t pv_threashold_mode, typename DTypeOut, bool is_causal, bool return_pv_count>
void SpargeAttentionSM80Dispatched(
  int8_t* Q, int8_t* K, half* V, DTypeOut* O,
  int32_t* PV_Count, int32_t *__restrict__ Lut, int32_t *__restrict__ Valid_Block_Num, float *__restrict__ PV_Threshold,
  float* Q_scale, float* K_scale,
  const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
  const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
  const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
  const uint32_t stride_bz_v, const uint32_t stride_seq_v, const uint32_t stride_h_v,
  const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
  float sm_scale) 
{
  constexpr MaskMode mask_mode = is_causal ? MaskMode::kCausal : MaskMode::kNone;

  //                                     smem_Q                                     smem_K                            smem_V                     smem_O
  size_t smem_max = std::max(CTA_Q * head_dim * sizeof(int8_t) + CTA_K * head_dim * sizeof(int8_t) + CTA_K * head_dim * sizeof(half), CTA_Q * head_dim * sizeof(half));
  
  auto kernel_func = qk_int_sv_f16_block_sparse_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, head_dim, DataType::kInt8, static_cast<QuantGranularity>(qk_quant_gran), static_cast<QuantGranularity>(qk_quant_gran), use_inst_buffer, static_cast<PVThresholdMode>(pv_threashold_mode), DTypeOut, ComputeUnit::kTensorCore, 
                                                mask_mode, return_pv_count>;

  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

  kernel_func<<<grid, block, smem_max>>>(
    Q, 
    K,
    V,
    O,
    PV_Count,
    Lut,
    Valid_Block_Num,
    PV_Threshold,
    Q_scale,
    K_scale,
    qo_len,
    kv_len,
    num_qo_heads / num_kv_heads,
    stride_bz_q, stride_seq_q, stride_h_q,
    stride_bz_k, stride_seq_k, stride_h_k,
    stride_bz_v, stride_seq_v, stride_h_v,
    stride_bz_o, stride_seq_o, stride_h_o,
    sm_scale);
}