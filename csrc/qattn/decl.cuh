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

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

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
  float sm_scale);

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, uint32_t qk_quant_gran, typename DTypePVAccum, bool use_inst_buffer, bool use_pv_fp16_accu, uint32_t pv_threashold_mode, typename DTypeOut, bool is_causal, bool fuse_v_scale, bool return_pv_count>
void SpargeAttentionSM89Dispatched(
  int8_t* Q, int8_t* K, __nv_fp8_e4m3* V, DTypeOut* O,
  int32_t* PV_Count, int32_t *__restrict__ Lut, int32_t *__restrict__ Valid_Block_Num, float *__restrict__ PV_Threshold,
  float* Q_scale, float* K_scale, float* V_scale,
  const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
  const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
  const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
  const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
  const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
  float sm_scale);

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t NUM_THREADS, uint32_t head_dim, uint32_t qk_quant_gran, uint32_t pv_threashold_mode, typename DTypeOut, bool is_causal, bool fuse_v_scale, bool return_pv_count>
void SpargeAttentionSM90Dispatched(
  int8_t* Q, int8_t* K, __nv_fp8_e4m3* V, DTypeOut* O,
  int32_t* PV_Count, int32_t *__restrict__ Lut, int32_t *__restrict__ Valid_Block_Num, float *__restrict__ PV_Threshold,
  float* Q_scale, float* K_scale, float* V_scale,
  const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len, const uint32_t padded_kv_len, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
  const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
  const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
  const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
  const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
  float sm_scale);