import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention
from diffusers.models import CogVideoXTransformer3DModel
from spas_sage_attn import spas_sage2_attn_meansim_cuda, spas_sage2_attn_meansim_topk_cuda


class SpargeCogVideoXAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, idx: int, mode: str, value: Optional[float] = None):
        self.idx = idx
        self.mode = mode
        self.value = value
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # query/key/value are [B, H, N, D]
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
            if self.mode == "cdfthreshd":
                hidden_states = spas_sage2_attn_meansim_cuda(q_hnd, k_hnd, v_hnd, simthreshd1=-0.1, cdfthreshd=self.value)
            else:
                hidden_states = spas_sage2_attn_meansim_topk_cuda(q_hnd, k_hnd, v_hnd, topk=self.value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


def set_sparge_cogvideox(
    model: CogVideoXTransformer3DModel,
    verbose=False,
    mode: str = None,
    value: Optional[float] = None,
):
    for idx, block in enumerate(model.transformer_blocks):
        block.attn1.verbose = verbose
        origin_processor = block.attn1.get_processor()
        processor = SpargeCogVideoXAttnProcessor(idx=idx, mode=mode, value=value)
        block.attn1.set_processor(processor)
        if not hasattr(block.attn1, "origin_processor"):
            block.attn1.origin_processor = origin_processor
