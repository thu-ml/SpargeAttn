# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers import WanTransformer3DModel
from diffusers.utils import logging
from diffusers.models.attention_dispatch import dispatch_attention_fn

from diffusers.models.transformers.transformer_wan import _get_qkv_projections, _get_added_kv_projections, WanAttention
from spas_sage_attn import spas_sage2_attn_meansim_cuda, spas_sage2_attn_meansim_topk_cuda


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SpargeWanAttnProcessor:
    _attention_backend = None

    def __init__(self, idx: int, mode: str, value: Optional[float] = None):
        self.idx = idx
        self.mode = mode
        self.value = value
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Decide whether to use Sparge or full attention on the main branch
        use_sparge = (
            query.is_cuda and key.is_cuda and value.is_cuda
            and query.shape[1] >= 128                    # seq len
            and query.shape[-1] in (64, 128)             # head dim
            and attention_mask is None                   # attn1 has no mask
        )

        if use_sparge and self.mode != "full":
            # Convert to HND explicitly to avoid internal rearrange; Make sure tensors are contiguous
            q_hnd = query.permute(0, 2, 1, 3).contiguous()  # [B, H, N, D]
            k_hnd = key.permute(0, 2, 1, 3).contiguous()
            v_hnd = value.permute(0, 2, 1, 3).contiguous()
            # Call Sparge in HND layout with only the necessary args
            if self.mode == "cdfthreshd":
                o_hnd = spas_sage2_attn_meansim_cuda(
                    q_hnd, k_hnd, v_hnd,
                    # simthreshd1=-0.1,
                    cdfthreshd=self.value,
                )
            else:
                o_hnd = spas_sage2_attn_meansim_topk_cuda(q_hnd, k_hnd, v_hnd, topk=self.value)
            hidden_states = o_hnd.permute(0, 2, 1, 3).contiguous()
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def set_sparge_wan(
    model: WanTransformer3DModel,
    mode: str = None,
    value: Optional[float] = None,
):
    for idx, block in enumerate(model.blocks):
        origin_processor = block.attn1.get_processor()
        processor = SpargeWanAttnProcessor(idx=idx, mode=mode, value=value)
        block.attn1.set_processor(processor)
        if not hasattr(block.attn1, "origin_processor"):
            block.attn1.origin_processor = origin_processor
