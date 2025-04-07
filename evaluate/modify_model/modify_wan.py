import torch
import torch.cuda.amp as amp
from spas_sage_attn.autotune import SparseAttentionMeansim
from wan.modules.model import rope_apply

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
import warnings


def flash_attention(
    inner_attention,
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn('Flash attention 3 is not available, use flash attention 2 instead.')
    # apply attention L H D -> 1 L H D, do not support varlen
    q = q.unflatten(0, (b, lq))
    k = k.unflatten(0, (b, lq))
    v = v.unflatten(0, (b, lq))
    x = inner_attention(q, k, v, scale=softmax_scale, is_causal=causal, tensor_layout="NHD")
    return x.type(out_dtype)


@torch.no_grad()
def sparge_forward(self, x, seq_lens, grid_sizes, freqs):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        seq_lens(Tensor): Shape [B]
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v
    assert self.window_size == (-1, -1), 'cur do not support varlen'
    assert seq_lens.shape[0] == 1
    
    q, k, v = qkv_fn(x)
    if self.use_usp:
        from wan.distributed.xdit_context_parallel import rope_apply as rope_apply_usp
        q = rope_apply_usp(q, grid_sizes, freqs)
        k = rope_apply_usp(k, grid_sizes, freqs)
    else:
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
    
    x = flash_attention(
            self.inner_attention,
            q=q,
            k=k,
            v=v,
        )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


def set_spas_sage_attn_wan(
    model,
    verbose=False,
    l1=0.07,
    pv_l1=0.08,
    tune_pv=True,
    use_usp=False,
):
    for block in model.blocks:
        block.self_attn.verbose = verbose
        block.self_attn.use_usp = use_usp
        block.self_attn.forward = sparge_forward.__get__(block.self_attn, type(block.self_attn))
        if use_usp:
            from xfuser.core.long_ctx_attention import xFuserLongContextAttention, AttnType
            block.self_attn.attn_processor = SparseAttentionMeansim(l1=l1, pv_l1=pv_l1, tune_pv=tune_pv)
            block.self_attn.usp_processor = xFuserLongContextAttention(attn_processor=block.self_attn.attn_processor, attn_type=AttnType.SPARSE_SAGE)
            def inner_attention_usp(q, k, v, scale, is_causal, tensor_layout):
                return block.self_attn.usp_processor(None, q, k, v, softmax_scale=scale, causal=is_causal)
            block.self_attn.inner_attention = inner_attention_usp
        else:
            block.self_attn.inner_attention = SparseAttentionMeansim(l1=l1, pv_l1=pv_l1, tune_pv=tune_pv)
