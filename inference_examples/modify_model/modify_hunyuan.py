import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers import HunyuanVideoTransformer3DModel
from spas_sage_attn import spas_sage2_attn_meansim_cuda, spas_sage2_attn_meansim_topk_cuda


class SpargeHunyuanVideoAttnProcessor2_0:
    def __init__(self, idx: int, mode: str, value: Optional[float] = None):
        self.layer_id = idx
        self.mode = mode
        self.value = value
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention
        # Decide whether to use Sparge or full attention on the main branch
        use_sparge = (
            query.is_cuda and key.is_cuda and value.is_cuda
            and query.shape[2] >= 128           # N (seq len)
            and query.shape[3] in (64, 128)     # D (head dim)
            and attention_mask is None          # Sparge path assumes no mask
        )
        if use_sparge and self.mode != "full":
            # Convert to HND explicitly to avoid internal rearrange; Make sure tensors are contiguous
            q_hnd = query.contiguous()  # [B, H, N, D]
            k_hnd = key.contiguous()
            v_hnd = value.contiguous()
            # Call Sparge in HND layout with only the necessary args
            if self.mode == "cdfthreshd":
                hidden_states = spas_sage2_attn_meansim_cuda(
                    q_hnd, k_hnd, v_hnd,
                    # simthreshd1=-0.1,
                    cdfthreshd=self.value,
                )
            else:
                hidden_states = spas_sage2_attn_meansim_topk_cuda(q_hnd, k_hnd, v_hnd, topk=self.value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


def set_sparge_hunyuan(
    model: HunyuanVideoTransformer3DModel,
    mode: str = None,
    value: Optional[float] = None,
):
    """
    Set SpargeHunyuanVideoAttnProcessor2_0 on all attention blocks.
    - `transformer_blocks`: dual-stream (video + text mixed)
    - `single_transformer_blocks`: single-stream (after concat)
    """
    # Dual-stream blocks
    for idx, block in enumerate(model.transformer_blocks):
        if block.attn.processor != None: 
            processor = SpargeHunyuanVideoAttnProcessor2_0(idx=idx, mode=mode, value=value)
            block.attn.set_processor(processor)
    # Single-stream blocks
    for idx, block in enumerate(model.single_transformer_blocks):
        if block.attn.processor != None: 
            processor = SpargeHunyuanVideoAttnProcessor2_0(idx=idx, mode=mode, value=value)
            block.attn.set_processor(processor)
