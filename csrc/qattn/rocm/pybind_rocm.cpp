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

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "attn_rocm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  // FP16 V matrix - works on all ROCm architectures
  m.def("qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf", 
        &qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf,
        "QK int8 SV f16 block sparse attention (all ROCm GPUs)");
  
  m.def("qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold",
        &qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold,
        "QK int8 SV f16 block sparse attention with PV threshold (all ROCm GPUs)");

#if defined(SA_ARCH_MI_SERIES)
  // FP8 V matrix - MI series only
  m.def("qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale",
        &qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale,
        "QK int8 SV f8 block sparse attention (MI series GPUs)");
  
  m.def("qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
        &qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold,
        "QK int8 SV f8 block sparse attention with PV threshold (MI series GPUs)");
#endif
}
