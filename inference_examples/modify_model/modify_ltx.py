# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
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

import torch
from typing import Optional

from diffusers import LTXVideoTransformer3DModel
from diffusers.utils import is_torch_version, logging
from diffusers.models.attention_dispatch import dispatch_attention_fn

from diffusers.models.transformers.transformer_ltx import LTXAttention, apply_rotary_emb
from spas_sage_attn import spas_sage2_attn_meansim_cuda, spas_sage2_attn_meansim_topk_cuda


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SpargeLTXVideoAttnProcessor:
    r"""
    Processor for implementing attention (SDPA is used by default if you're using PyTorch 2.0). This is used in the LTX
    model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    _attention_backend = None

    def __init__(self, idx: int, mode: str, value: Optional[float] = None):
        self.idx = idx
        self.mode = mode
        self.value = value
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX attention processors require a minimum PyTorch version of 2.0. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "LTXAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Decide whether to use Sparge or full attention
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
            if self.mode == "cdfthreshd":
                out = spas_sage2_attn_meansim_cuda(q_hnd, k_hnd, v_hnd, simthreshd1=-0.1, cdfthreshd=self.value)
            else:
                out = spas_sage2_attn_meansim_topk_cuda(q_hnd, k_hnd, v_hnd, topk=self.value)
            hidden_states = out.permute(0, 2, 1, 3).contiguous()
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
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def set_sparge_ltx(
    model: LTXVideoTransformer3DModel,
    verbose=False,
    mode: str = None,
    value: Optional[float] = None,
):
    for idx, block in enumerate(model.transformer_blocks):
        block.attn1.verbose = verbose
        origin_processor = block.attn1.get_processor()
        processor = SpargeLTXVideoAttnProcessor(idx=idx, mode=mode, value=value)
        block.attn1.set_processor(processor)
        if not hasattr(block.attn1, "origin_processor"):
            block.attn1.origin_processor = origin_processor
