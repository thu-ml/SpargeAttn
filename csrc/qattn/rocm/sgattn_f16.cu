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

/*
 * Register-based attention kernel for AMD RDNA3 (gfx11) using rocWMMA.
 * 
 * RDNA3 WMMA D[row][col] layout (from AMD Matrix Instruction Calculator):
 *   fragment.x[reg] where reg ∈ [0,7]:
 *     row = reg * 2 + (lane_id >= 16 ? 1 : 0)
 *     col = lane_id % 16
 * 
 * Each lane handles ONE column across 8 rows (either even or odd rows).
 * Lanes 0-15 handle even rows (0,2,4,...,14)
 * Lanes 16-31 handle odd rows (1,3,5,...,15)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

namespace gfx11Params {
    constexpr uint32_t WAVE_SIZE = 32u;
    constexpr uint32_t WMMA_M = 16u;
    constexpr uint32_t WMMA_N = 16u;
    constexpr uint32_t WMMA_K_INT8 = 16u;    // RDNA3 WMMA K for INT8
    constexpr uint32_t WMMA_K_FP16 = 16u;    // RDNA3 WMMA K for FP16/BF16
}

constexpr float LOG2E = 1.44269504088896340736f;
#define div_ceil_hip(M, N) (((M) + (N) - 1) / (N))

enum class QuantGranularity {
    kPerTensor = 0,
    kPerBlock = 1,
    kPerWarp = 2,
    kPerThread = 3,
};

enum class MaskMode {
    kNone = 0,
    kCausal = 1,
};

// Type traits for FP16/BF16 support
template<typename T> struct TypeTraits;

template<>
struct TypeTraits<half> {
    __device__ static float to_float(half val) { return __half2float(val); }
    __device__ static half from_float(float val) { return __float2half(val); }
};

template<>
struct TypeTraits<hip_bfloat16> {
    __device__ static float to_float(hip_bfloat16 val) { return static_cast<float>(val); }
    __device__ static hip_bfloat16 from_float(float val) { return hip_bfloat16(val); }
};

// Fragment types for QK phase: INT8 M16N16K16
using FragA_QK = fragment<matrix_a, gfx11Params::WMMA_M, gfx11Params::WMMA_N, gfx11Params::WMMA_K_INT8, int8_t, col_major>;
using FragB_QK = fragment<matrix_b, gfx11Params::WMMA_M, gfx11Params::WMMA_N, gfx11Params::WMMA_K_INT8, int8_t, row_major>;
using FragAcc_QK = fragment<accumulator, gfx11Params::WMMA_M, gfx11Params::WMMA_N, gfx11Params::WMMA_K_INT8, int32_t>;

// Fragment types for SV phase: FP16 M16N16K16
template<typename DTypeV> struct SVFragmentTypes;

template<>
struct SVFragmentTypes<half> {
    using FragA = fragment<matrix_a, gfx11Params::WMMA_M, gfx11Params::WMMA_N, gfx11Params::WMMA_K_FP16, half, row_major>;
    using FragB = fragment<matrix_b, gfx11Params::WMMA_M, gfx11Params::WMMA_N, gfx11Params::WMMA_K_FP16, half, row_major>;
    using FragAcc = fragment<accumulator, gfx11Params::WMMA_M, gfx11Params::WMMA_N, gfx11Params::WMMA_K_FP16, float>;
};

template<>
struct SVFragmentTypes<hip_bfloat16> {
    using FragA = fragment<matrix_a, gfx11Params::WMMA_M, gfx11Params::WMMA_N, gfx11Params::WMMA_K_FP16, bfloat16_t, row_major>;
    using FragB = fragment<matrix_b, gfx11Params::WMMA_M, gfx11Params::WMMA_N, gfx11Params::WMMA_K_FP16, bfloat16_t, row_major>;
    using FragAcc = fragment<accumulator, gfx11Params::WMMA_M, gfx11Params::WMMA_N, gfx11Params::WMMA_K_FP16, float>;
};

/*
 * RDNA3 WMMA element-to-matrix mapping helpers.
 * Based on AMD Matrix Instruction Calculator output.
 */
__device__ __forceinline__ uint32_t wmma_elem_row(uint32_t reg, uint32_t lane_id) {
    // reg ∈ [0,7], row = reg * 2 + (lane_id >= 16 ? 1 : 0)
    return reg * 2 + (lane_id >> 4);
}

__device__ __forceinline__ uint32_t wmma_elem_col(uint32_t lane_id) {
    // col = lane_id % 16
    return lane_id & 15;
}

/*
 * Register-based attention kernel.
 * 
 * Key insight for RDNA3 softmax:
 * - Each lane owns one column across 8 rows
 * - For row-wise softmax, we need to reduce across 16 lanes (0-15 for even rows, 16-31 for odd)
 * - Use __shfl_xor within the lane group for fast reduction
 */
template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t HEAD_DIM,
         QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
         bool use_inst_buffer, uint32_t pv_threshold_mode,
         typename DTypeV, MaskMode mask_mode, bool return_pv_count,
         uint32_t NUM_THREADS = (CTA_Q / WARP_Q) * gfx11Params::WAVE_SIZE>
__global__ void __launch_bounds__(NUM_THREADS)
qk_int_sv_f16_block_sparse_attn_kernel_rocm(
    int8_t* __restrict__ Q, 
    int8_t* __restrict__ K, 
    DTypeV* __restrict__ V, 
    DTypeV* __restrict__ O,
    int32_t* __restrict__ PV_Count, 
    int32_t* __restrict__ Lut, 
    int32_t* __restrict__ Valid_Block_Num,
    float* __restrict__ PV_Threshold, 
    float* __restrict__ Q_scale, 
    float* __restrict__ K_scale,
    const uint32_t qo_len, 
    const uint32_t kv_len, 
    const uint32_t num_kv_groups,
    const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
    const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
    const uint32_t stride_bz_v, const uint32_t stride_seq_v, const uint32_t stride_h_v,
    const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
    float sm_scale)
{
    using namespace gfx11Params;
    using Traits = TypeTraits<DTypeV>;
    using SVFrags = SVFragmentTypes<DTypeV>;
    using FragA_SV_T = typename SVFrags::FragA;
    using FragB_SV_T = typename SVFrags::FragB;
    using FragAcc_SV_T = typename SVFrags::FragAcc;
    
    constexpr uint32_t NUM_WARPS_Q = CTA_Q / WARP_Q;
    constexpr uint32_t NUM_WARPS_K = 1;  // Always 1 for our configuration
    constexpr uint32_t NUM_TILES_K = CTA_K / WMMA_N;       // 4 for CTA_K=64
    constexpr uint32_t NUM_TILES_V = HEAD_DIM / WMMA_N;    // 8 for HD=128, 4 for HD=64
    constexpr uint32_t NUM_K_ITERS = HEAD_DIM / WMMA_K_INT8;   // 8 for HD=128 (K=16)
    constexpr uint32_t NUM_SV_ITERS = CTA_K / WMMA_K_FP16;     // 4 for CTA_K=64, K=16
    
    const uint32_t batch_id = blockIdx.z;
    const uint32_t bx = blockIdx.x;
    const uint32_t head_id = blockIdx.y;
    const uint32_t num_qo_heads = gridDim.y;
    
    const uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    const uint32_t warp_id = tid / WAVE_SIZE;
    const uint32_t lane_id = tid % WAVE_SIZE;
    const uint32_t warp_idx_q = warp_id / NUM_WARPS_K;
    const uint32_t warp_idx_k = warp_id % NUM_WARPS_K;
    
    // For RDNA3 WMMA layout: lane 0-15 handle even rows, lane 16-31 handle odd rows
    const uint32_t is_odd_row_lane = lane_id >> 4;  // 0 for lanes 0-15, 1 for lanes 16-31
    const uint32_t lane_col = lane_id & 15;         // Column index within 16x16 tile
    
    sm_scale *= LOG2E;
    
    const uint32_t num_block_q = gridDim.x;
    const uint32_t num_block_k = div_ceil_hip(kv_len, CTA_K);
    const uint32_t num_iterations = Valid_Block_Num[batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx];
    
    if (num_iterations == 0) return;
    
    const int32_t* lut_ptr = Lut + batch_id * num_qo_heads * num_block_q * num_block_k + 
                             head_id * num_block_q * num_block_k + bx * num_block_k;
    
    // Shared memory layout
    extern __shared__ char smem[];
    
    int8_t* smem_Q = reinterpret_cast<int8_t*>(smem);
    int8_t* smem_K = smem_Q + HEAD_DIM * CTA_Q;
    DTypeV* smem_V = reinterpret_cast<DTypeV*>(smem_K + HEAD_DIM * CTA_K);
    DTypeV* smem_S = smem_V + CTA_K * HEAD_DIM;  // For storing S after softmax
    
    const uint32_t q_start = bx * CTA_Q;
    const uint32_t q_tile_row = warp_idx_q * WMMA_M;  // Each warp handles 16 Q rows
    
    // ========================================
    // REGISTER-BASED STATE
    // ========================================
    
    // Output accumulator: RO[NUM_TILES_V][8] - 8 elements per 16x16 tile
    float RO[NUM_TILES_V][8];
    
    // Per-row softmax state: m (max) and d (sum)
    // Each lane owns 8 elements in the same column but different rows
    // We need per-row max/sum, so each element has its own m and d
    float m[8];  // max for each of the 8 rows this lane owns
    float d[8];  // sum for each of the 8 rows this lane owns
    
    // Initialize RO, m, d
    #pragma unroll
    for (uint32_t fv = 0; fv < NUM_TILES_V; fv++) {
        #pragma unroll
        for (uint32_t i = 0; i < 8; i++) {
            RO[fv][i] = 0.0f;
        }
    }
    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        m[i] = -5000000.0f;
        d[i] = 0.0f;
    }
    
    // Load Q to shared memory (col-major for WMMA A)
    for (uint32_t i = tid; i < CTA_Q * HEAD_DIM; i += NUM_THREADS) {
        uint32_t q_row = i % CTA_Q;
        uint32_t q_col = i / CTA_Q;
        uint32_t q_idx = q_start + q_row;
        int8_t val = 0;
        if (q_idx < qo_len) {
            val = Q[batch_id * stride_bz_q + q_idx * stride_seq_q + head_id * stride_h_q + q_col];
        }
        smem_Q[i] = val;
    }
    __syncthreads();
    
    // Get Q scale
    float q_scale_val;
    if constexpr (Q_GRAN == QuantGranularity::kPerBlock) {
        q_scale_val = Q_scale[batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx];
    } else if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
        const uint32_t num_warp_block_q = num_block_q * NUM_WARPS_Q;
        q_scale_val = Q_scale[batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * NUM_WARPS_Q + warp_idx_q];
    }
    
    // Main loop over K blocks
    uint32_t k_block_idx = 0;
    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        k_block_idx += lut_ptr[iter];
        uint32_t k_start = k_block_idx * CTA_K;
        
        // Load K to shared memory (as K^T)
        for (uint32_t i = tid; i < CTA_K * HEAD_DIM; i += NUM_THREADS) {
            uint32_t n = i % CTA_K;
            uint32_t k = i / CTA_K;
            uint32_t k_idx = k_start + n;
            int8_t val = 0;
            if (k_idx < kv_len) {
                val = K[batch_id * stride_bz_k + k_idx * stride_seq_k + (head_id / num_kv_groups) * stride_h_k + k];
            }
            smem_K[k * CTA_K + n] = val;
        }
        
        // Load V to shared memory
        for (uint32_t i = tid; i < CTA_K * HEAD_DIM; i += NUM_THREADS) {
            uint32_t v_row = i / HEAD_DIM;
            uint32_t v_col = i % HEAD_DIM;
            uint32_t v_idx = k_start + v_row;
            DTypeV val = Traits::from_float(0.0f);
            if (v_idx < kv_len) {
                val = V[batch_id * stride_bz_v + v_idx * stride_seq_v + (head_id / num_kv_groups) * stride_h_v + v_col];
            }
            smem_V[v_row * HEAD_DIM + v_col] = val;
        }
        __syncthreads();
        
        // Get K scale
        float k_scale_val;
        if constexpr (K_GRAN == QuantGranularity::kPerBlock) {
            const uint32_t num_kv_heads = num_qo_heads / num_kv_groups;
            k_scale_val = K_scale[batch_id * num_kv_heads * num_block_k + (head_id / num_kv_groups) * num_block_k + k_block_idx];
        } else if constexpr (K_GRAN == QuantGranularity::kPerWarp) {
            const uint32_t num_kv_heads = num_qo_heads / num_kv_groups;
            const uint32_t num_warp_block_k = num_block_k * NUM_WARPS_K;
            k_scale_val = K_scale[batch_id * num_kv_heads * num_warp_block_k + (head_id / num_kv_groups) * num_warp_block_k + k_block_idx * NUM_WARPS_K + warp_idx_k];
        }
        
        float dequant_scale = q_scale_val * k_scale_val * sm_scale;
        
        // ============================================
        // Phase 1: Compute QK^T using rocWMMA INT8
        // RS[NUM_TILES_K][8] holds the QK results in registers
        // ============================================
        
        float RS[NUM_TILES_K][8];  // QK results for all K tiles
        
        #pragma unroll
        for (uint32_t tile_k = 0; tile_k < NUM_TILES_K; tile_k++) {
            FragAcc_QK acc_qk;
            fill_fragment(acc_qk, 0);
            
            #pragma unroll
            for (uint32_t k_iter = 0; k_iter < NUM_K_ITERS; k_iter++) {
                FragA_QK frag_q;
                load_matrix_sync(frag_q, smem_Q + k_iter * WMMA_K_INT8 * CTA_Q + q_tile_row, CTA_Q);
                
                FragB_QK frag_k;
                load_matrix_sync(frag_k, smem_K + k_iter * WMMA_K_INT8 * CTA_K + tile_k * WMMA_N, CTA_K);
                mma_sync(acc_qk, frag_q, frag_k, acc_qk);
            }
            
            // Dequantize and apply masks directly in registers
            #pragma unroll
            for (uint32_t reg = 0; reg < 8; reg++) {
                float val = static_cast<float>(acc_qk.x[reg]) * dequant_scale;
                
                // Apply out-of-bounds and causal masks
                uint32_t row = wmma_elem_row(reg, lane_id);
                uint32_t col = wmma_elem_col(lane_id);
                uint32_t q_idx = q_start + q_tile_row + row;
                uint32_t k_idx = k_start + tile_k * WMMA_N + col;
                
                if (k_idx >= kv_len) val = -5000000.0f;
                if constexpr (mask_mode == MaskMode::kCausal) {
                    if (k_idx > q_idx) val = -5000000.0f;
                }
                
                RS[tile_k][reg] = val;
            }
        }
        
        // ============================================
        // Phase 2: Online softmax update (register-based)
        // For each of the 8 rows this lane owns, find max across K tiles
        // Then reduce across 16 lanes (shuffle within lane group 0-15 or 16-31)
        // ============================================
        
        #pragma unroll
        for (uint32_t reg = 0; reg < 8; reg++) {
            float m_prev = m[reg];
            
            // Find max across all K tiles for this row element
            float m_local = RS[0][reg];
            #pragma unroll
            for (uint32_t tile_k = 1; tile_k < NUM_TILES_K; tile_k++) {
                m_local = fmaxf(m_local, RS[tile_k][reg]);
            }
            
            // Warp-level max reduction across 16 lanes (either 0-15 or 16-31)
            // Only reduce within the same row-parity group
            #pragma unroll
            for (uint32_t offset = 8; offset > 0; offset /= 2) {
                m_local = fmaxf(m_local, __shfl_xor(m_local, offset, WAVE_SIZE));
            }
            
            m[reg] = fmaxf(m_prev, m_local);
            float o_scale = exp2f(m_prev - m[reg]);
            
            // Scale existing d and RO
            d[reg] *= o_scale;
            #pragma unroll
            for (uint32_t fv = 0; fv < NUM_TILES_V; fv++) {
                RO[fv][reg] *= o_scale;
            }
            
            // Compute exp and accumulate to d
            float local_sum = 0.0f;
            #pragma unroll
            for (uint32_t tile_k = 0; tile_k < NUM_TILES_K; tile_k++) {
                RS[tile_k][reg] = exp2f(RS[tile_k][reg] - m[reg]);
                local_sum += RS[tile_k][reg];
            }
            
            // Warp-level sum reduction across 16 lanes
            #pragma unroll
            for (uint32_t offset = 8; offset > 0; offset /= 2) {
                local_sum += __shfl_xor(local_sum, offset, WAVE_SIZE);
            }
            
            d[reg] += local_sum;
        }
        
        // ============================================
        // Phase 3: Store S to shared memory for S@V
        // Convert RS to DTypeV and store in row-major layout
        // ============================================
        
        #pragma unroll
        for (uint32_t tile_k = 0; tile_k < NUM_TILES_K; tile_k++) {
            #pragma unroll
            for (uint32_t reg = 0; reg < 8; reg++) {
                uint32_t row = wmma_elem_row(reg, lane_id);
                uint32_t col = wmma_elem_col(lane_id);
                uint32_t global_row = q_tile_row + row;
                uint32_t global_col = tile_k * WMMA_N + col;
                
                smem_S[global_row * CTA_K + global_col] = Traits::from_float(RS[tile_k][reg]);
            }
        }
        __syncthreads();
        
        // ============================================
        // Phase 4: Compute S @ V using rocWMMA FP16
        // Accumulate into RO registers
        // ============================================
        
        #pragma unroll
        for (uint32_t tile_v = 0; tile_v < NUM_TILES_V; tile_v++) {
            FragAcc_SV_T acc_sv;
            fill_fragment(acc_sv, 0.0f);
            
            #pragma unroll
            for (uint32_t k_iter = 0; k_iter < NUM_SV_ITERS; k_iter++) {
                FragA_SV_T frag_s;
                // S is stored row-major: [CTA_Q, CTA_K]
                load_matrix_sync(frag_s, smem_S + q_tile_row * CTA_K + k_iter * WMMA_K_FP16, CTA_K);
                
                FragB_SV_T frag_v;
                load_matrix_sync(frag_v, smem_V + k_iter * WMMA_K_FP16 * HEAD_DIM + tile_v * WMMA_N, HEAD_DIM);
                
                mma_sync(acc_sv, frag_s, frag_v, acc_sv);
            }
            
            // Accumulate to RO using direct element access
            #pragma unroll
            for (uint32_t reg = 0; reg < 8; reg++) {
                RO[tile_v][reg] += acc_sv.x[reg];
            }
        }
        
        __syncthreads();
    }
    
    // ============================================
    // Final: Normalize by d and write to output
    // ============================================
    
    // Write RO to output via shared memory
    // Reuse smem_S as temp buffer
    DTypeV* smem_out = smem_S;
    
    #pragma unroll
    for (uint32_t tile_v = 0; tile_v < NUM_TILES_V; tile_v++) {
        #pragma unroll
        for (uint32_t reg = 0; reg < 8; reg++) {
            uint32_t row = wmma_elem_row(reg, lane_id);
            uint32_t col = wmma_elem_col(lane_id);
            uint32_t global_row = q_tile_row + row;
            uint32_t global_col = tile_v * WMMA_N + col;
            
            float inv_d = 1.0f / d[reg];
            float out_val = RO[tile_v][reg] * inv_d;
            
            smem_out[global_row * HEAD_DIM + global_col] = Traits::from_float(out_val);
        }
    }
    __syncthreads();
    
    // Copy from smem to global memory
    for (uint32_t i = tid; i < CTA_Q * HEAD_DIM; i += NUM_THREADS) {
        uint32_t row = i / HEAD_DIM;
        uint32_t col = i % HEAD_DIM;
        uint32_t o_idx = q_start + row;
        
        if (o_idx < qo_len) {
            O[batch_id * stride_bz_o + o_idx * stride_seq_o + head_id * stride_h_o + col] = smem_out[row * HEAD_DIM + col];
        }
    }
}

// Kernel launcher
template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t HEAD_DIM,
         uint32_t qk_quant_gran, bool use_inst_buffer, uint32_t pv_threshold_mode,
         typename DTypeV, bool is_causal, bool return_pv_count>
void SpargeAttentionROCmF16Dispatched(
    int8_t* Q, int8_t* K, DTypeV* V, DTypeV* O,
    int32_t* PV_Count, int32_t* Lut, int32_t* Valid_Block_Num, float* PV_Threshold,
    float* Q_scale, float* K_scale,
    const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len,
    const uint32_t num_qo_heads, const uint32_t num_kv_heads,
    const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
    const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
    const uint32_t stride_bz_v, const uint32_t stride_seq_v, const uint32_t stride_h_v,
    const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
    float sm_scale)
{
    constexpr QuantGranularity Q_GRAN = (qk_quant_gran == 1) ? QuantGranularity::kPerBlock : QuantGranularity::kPerWarp;
    constexpr QuantGranularity K_GRAN = Q_GRAN;
    constexpr MaskMode mask_mode = is_causal ? MaskMode::kCausal : MaskMode::kNone;
    
    const uint32_t num_kv_groups = num_qo_heads / num_kv_heads;
    const uint32_t num_block_q = div_ceil_hip(qo_len, CTA_Q);
    
    // Calculate shared memory size (reduced - no smem_O, no smem_QK_float!)
    // smem_Q: HEAD_DIM * CTA_Q (int8)
    // smem_K: HEAD_DIM * CTA_K (int8)
    // smem_V: CTA_K * HEAD_DIM (DTypeV)
    // smem_S: CTA_Q * CTA_K (DTypeV) - also used as output buffer
    
    size_t smem_size = HEAD_DIM * CTA_Q * sizeof(int8_t) +      // smem_Q
                       HEAD_DIM * CTA_K * sizeof(int8_t) +      // smem_K
                       CTA_K * HEAD_DIM * sizeof(DTypeV) +      // smem_V
                       max(CTA_Q * CTA_K, CTA_Q * HEAD_DIM) * sizeof(DTypeV); // smem_S / smem_out
    
    constexpr uint32_t NUM_WARPS = CTA_Q / WARP_Q;
    constexpr uint32_t NUM_THREADS = NUM_WARPS * gfx11Params::WAVE_SIZE;
    
    dim3 grid(num_block_q, num_qo_heads, batch_size);
    dim3 block(NUM_THREADS, 1, 1);
    
    hipLaunchKernelGGL((qk_int_sv_f16_block_sparse_attn_kernel_rocm<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM,
                        Q_GRAN, K_GRAN, use_inst_buffer, pv_threshold_mode, DTypeV, mask_mode, return_pv_count, NUM_THREADS>),
                       grid, block, smem_size, 0,
                       Q, K, V, O, PV_Count, Lut, Valid_Block_Num, PV_Threshold, Q_scale, K_scale,
                       qo_len, kv_len, num_kv_groups,
                       stride_bz_q, stride_seq_q, stride_h_q,
                       stride_bz_k, stride_seq_k, stride_h_k,
                       stride_bz_v, stride_seq_v, stride_h_v,
                       stride_bz_o, stride_seq_o, stride_h_o,
                       sm_scale);
}

// Explicit template instantiations
// qk_quant_gran=1 (kPerBlock)
// CTA_Q=64, HEAD_DIM=64, half
template void SpargeAttentionROCmF16Dispatched<64, 64, 16, 64, 64, 1, true, 0, half, true, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<64, 64, 16, 64, 64, 1, true, 0, half, false, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// CTA_Q=64, HEAD_DIM=64, bfloat16
template void SpargeAttentionROCmF16Dispatched<64, 64, 16, 64, 64, 1, true, 0, hip_bfloat16, true, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<64, 64, 16, 64, 64, 1, true, 0, hip_bfloat16, false, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// CTA_Q=32, HEAD_DIM=128, half
template void SpargeAttentionROCmF16Dispatched<32, 64, 16, 64, 128, 1, true, 0, half, true, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 64, 16, 64, 128, 1, true, 0, half, false, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// CTA_Q=32, HEAD_DIM=128, bfloat16
template void SpargeAttentionROCmF16Dispatched<32, 64, 16, 64, 128, 1, true, 0, hip_bfloat16, true, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 64, 16, 64, 128, 1, true, 0, hip_bfloat16, false, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// qk_quant_gran=2 (kPerWarp)
// CTA_Q=64, HEAD_DIM=64, half
template void SpargeAttentionROCmF16Dispatched<64, 64, 16, 64, 64, 2, true, 0, half, true, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<64, 64, 16, 64, 64, 2, true, 0, half, false, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// CTA_Q=64, HEAD_DIM=64, bfloat16
template void SpargeAttentionROCmF16Dispatched<64, 64, 16, 64, 64, 2, true, 0, hip_bfloat16, true, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<64, 64, 16, 64, 64, 2, true, 0, hip_bfloat16, false, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// CTA_Q=32, HEAD_DIM=128, half
template void SpargeAttentionROCmF16Dispatched<32, 64, 16, 64, 128, 2, true, 0, half, true, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 64, 16, 64, 128, 2, true, 0, half, false, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// CTA_Q=32, HEAD_DIM=128, bfloat16
template void SpargeAttentionROCmF16Dispatched<32, 64, 16, 64, 128, 2, true, 0, hip_bfloat16, true, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 64, 16, 64, 128, 2, true, 0, hip_bfloat16, false, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// ========================================
// CTA_K=32 instantiations for reduced register pressure (HEAD_DIM=128 only)
// ========================================

// qk_quant_gran=1 (kPerBlock), CTA_Q=32, CTA_K=32, HEAD_DIM=128, half
template void SpargeAttentionROCmF16Dispatched<32, 32, 16, 32, 128, 1, true, 0, half, true, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 32, 16, 32, 128, 1, true, 0, half, false, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// qk_quant_gran=1 (kPerBlock), CTA_Q=32, CTA_K=32, HEAD_DIM=128, bfloat16
template void SpargeAttentionROCmF16Dispatched<32, 32, 16, 32, 128, 1, true, 0, hip_bfloat16, true, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 32, 16, 32, 128, 1, true, 0, hip_bfloat16, false, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// qk_quant_gran=2 (kPerWarp), CTA_Q=32, CTA_K=32, HEAD_DIM=128, half
template void SpargeAttentionROCmF16Dispatched<32, 32, 16, 32, 128, 2, true, 0, half, true, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 32, 16, 32, 128, 2, true, 0, half, false, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// qk_quant_gran=2 (kPerWarp), CTA_Q=32, CTA_K=32, HEAD_DIM=128, bfloat16
template void SpargeAttentionROCmF16Dispatched<32, 32, 16, 32, 128, 2, true, 0, hip_bfloat16, true, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 32, 16, 32, 128, 2, true, 0, hip_bfloat16, false, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// ========================================
// CTA_K=16 instantiations (even finer granularity)
// ========================================

// qk_quant_gran=1 (kPerBlock), CTA_Q=32, CTA_K=16, HEAD_DIM=128, half
template void SpargeAttentionROCmF16Dispatched<32, 16, 16, 16, 128, 1, true, 0, half, true, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 16, 16, 16, 128, 1, true, 0, half, false, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// qk_quant_gran=1 (kPerBlock), CTA_Q=32, CTA_K=16, HEAD_DIM=128, bfloat16
template void SpargeAttentionROCmF16Dispatched<32, 16, 16, 16, 128, 1, true, 0, hip_bfloat16, true, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 16, 16, 16, 128, 1, true, 0, hip_bfloat16, false, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// qk_quant_gran=2 (kPerWarp), CTA_Q=32, CTA_K=16, HEAD_DIM=128, half
template void SpargeAttentionROCmF16Dispatched<32, 16, 16, 16, 128, 2, true, 0, half, true, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 16, 16, 16, 128, 2, true, 0, half, false, false>(
    int8_t*, int8_t*, half*, half*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

// qk_quant_gran=2 (kPerWarp), CTA_Q=32, CTA_K=16, HEAD_DIM=128, bfloat16
template void SpargeAttentionROCmF16Dispatched<32, 16, 16, 16, 128, 2, true, 0, hip_bfloat16, true, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

template void SpargeAttentionROCmF16Dispatched<32, 16, 16, 16, 128, 2, true, 0, hip_bfloat16, false, false>(
    int8_t*, int8_t*, hip_bfloat16*, hip_bfloat16*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);
