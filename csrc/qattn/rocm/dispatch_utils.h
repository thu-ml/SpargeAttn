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

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)           \
  [&]() {                                                                          \
    if (pytorch_dtype == at::ScalarType::Half) {                                   \
      using c_type = __half;                                                       \
      return __VA_ARGS__();                                                        \
    } else if (pytorch_dtype == at::ScalarType::BFloat16) {                        \
      using c_type = __hip_bfloat16;                                               \
      return __VA_ARGS__();                                                        \
    } else {                                                                       \
      TORCH_CHECK(false, "Unsupported dtype: ", pytorch_dtype);                    \
    }                                                                              \
  }()

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)                                 \
  [&]() {                                                                          \
    if (head_dim == 64) {                                                          \
      constexpr int HEAD_DIM = 64;                                                 \
      return __VA_ARGS__();                                                        \
    } else if (head_dim == 128) {                                                  \
      constexpr int HEAD_DIM = 128;                                                \
      return __VA_ARGS__();                                                        \
    } else {                                                                       \
      TORCH_CHECK(false, "Unsupported head_dim: ", head_dim);                      \
    }                                                                              \
  }()

#define DISPATCH_CAUSAL(is_causal, IS_CAUSAL, ...)                                 \
  [&]() {                                                                          \
    if (is_causal) {                                                               \
      constexpr bool IS_CAUSAL = true;                                             \
      return __VA_ARGS__();                                                        \
    } else {                                                                       \
      constexpr bool IS_CAUSAL = false;                                            \
      return __VA_ARGS__();                                                        \
    }                                                                              \
  }()

#define DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, ...)                  \
  [&]() {                                                                          \
    if (qk_quant_gran == 1) {                                                      \
      constexpr int QK_QUANT_GRAN = 1;                                             \
      return __VA_ARGS__();                                                        \
    } else if (qk_quant_gran == 2) {                                               \
      constexpr int QK_QUANT_GRAN = 2;                                             \
      return __VA_ARGS__();                                                        \
    } else if (qk_quant_gran == 3) {                                               \
      constexpr int QK_QUANT_GRAN = 3;                                             \
      return __VA_ARGS__();                                                        \
    } else {                                                                       \
      TORCH_CHECK(false, "Unsupported qk_quant_gran: ", qk_quant_gran);            \
    }                                                                              \
  }()
