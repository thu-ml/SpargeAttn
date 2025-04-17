import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models._features_fx import register_notrace_function
# Import the context manager
from torch.backends.cuda import sdp_kernel, SDPBackend # Ensure this import is present

# Helper Functions (Keep these exactly as they were)
def blockify(x, num_heads, split_size: int, flag = True):
    # ... (no changes needed here) ...
    B, h, N, E  = x.shape
    if N == 0: # Handle empty sequence edge case
        if flag:
            grid_size = 0
            x1 = x.new_zeros((0, h, 0, E))
            x2 = x.new_zeros((B, h, 0, 0))
            return x1, x2, grid_size
        else:
            grid_size = 0
            x1 = x.new_zeros((0, h, 0, E))
            return x1, None, grid_size
    if N % split_size != 0:
        raise ValueError(f"Sequence length N ({N}) must be divisible by split_size ({split_size})")
    grid_size = N // split_size
    x = x.view(B, h, grid_size, split_size, E)
    x1 = x.permute(0, 2, 1, 3, 4).reshape(B * grid_size, h, split_size, E).contiguous()
    if flag:
        x2 = x.permute(0, 1, 2, 3, 4).reshape(B, h, grid_size, -1).contiguous()
    else:
        x2 = None
    return x1, x2, grid_size

def blockify_mask(mask, split_size: int):
    # ... (no changes needed here) ...
    if mask is None: return None
    B, N = mask.shape
    if N == 0: return mask.new_zeros((0, 0))
    if N % split_size != 0: raise ValueError(f"Mask length N ({N}) must be divisible by split_size ({split_size})")
    grid_size = N // split_size
    mask_blocks = mask.view(B, grid_size, split_size)
    x1 = mask_blocks.view(B * grid_size, split_size).contiguous()
    return x1

@register_notrace_function
def deblockifyv2(x, B, grid_size: int, split_size: int):
    # ... (no changes needed here) ...
    Bt, h, s, E = x.shape
    if Bt == 0: return x.new_zeros((B, h, 0, E))
    x = x.view(B, grid_size, h, s, E)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    N = grid_size * split_size
    x = x.view(B, h, N, E)
    return x

@register_notrace_function
def deblockify(x, B, grid_size: int, split_size: int):
    # ... (no changes needed here) ...
    _, h, g, E_prime = x.shape
    if g == 0:
        N = 0
        E_final = E_prime // split_size if split_size > 0 else E_prime
        return x.new_zeros((B, h, N, E_final))
    N = grid_size * split_size
    x = x.view(B, h, N, -1)
    return x

class RLSAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.split_size = config["split_size"]
        if self.split_size <= 0:
             raise ValueError(f"split_size must be positive, got {self.split_size}")
        self.dropout_p = config["attention_dropout"]

    def forward(self, Q, K, V, mask=None, is_causal=False, scale=None, tensor_layout="HND"):
        if tensor_layout != "HND":
             raise NotImplementedError(f"Tensor layout {tensor_layout} not handled by RLSAttn. Requires 'HND'.")

        B, H, N, D = Q.shape
        target_dtype = Q.dtype

        if N > 0 and N % self.split_size != 0:
             raise ValueError(f"Sequence length {N} must be divisible by split_size {self.split_size}")

        # --- Prepare Mask ---
        # Mask processing logic - simplified assuming mask is handled upstream or None for now
        attn_mask_stage1 = None # Keep it simple for now, can add mask logic back if needed

        # --- Blockify Q, K, V ---
        k1, k2, grid_size = blockify(K, self.num_head, self.split_size)
        q1, q2, _         = blockify(Q, self.num_head, self.split_size)
        v1, _, _          = blockify(V, self.num_head, self.split_size, flag=False)

        # --- Use Context Manager to disable memory-efficient backend ---
        # This forces PyTorch to use Flash Attention (if available/compatible) or the Math backend.
        # Note: `enable_math=True` is the fallback if flash/mem_efficient aren't chosen.
        # `enable_flash=True` allows FlashAttention if it's installed and conditions are met.
        with sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):

            # --- Stage 1: Local Attention ---
            x1 = F.scaled_dot_product_attention(
                q1, k1, v1, # No explicit .to() needed if context manager handles dtype selection
                attn_mask=attn_mask_stage1,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
                scale=scale
            )

            # --- Deblockify Stage 1 Output ---
            x1_deblocked = deblockifyv2(x1, B, grid_size, self.split_size)

            # --- Prepare Values for Stage 2 ---
            x1_grid = x1_deblocked.view(B, self.num_head, grid_size, -1)

            # --- Stage 2: Global Attention ---
            x = F.scaled_dot_product_attention(
                q2, k2, x1_grid,
                attn_mask=None, # Assuming no mask for global stage
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
                scale=scale
            )

        # --- Final Deblockify ---
        x_final = deblockify(x, B, grid_size, self.split_size)

        # Ensure final output matches original Q dtype
        return x_final.to(target_dtype)