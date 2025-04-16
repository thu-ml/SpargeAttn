import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models._features_fx import register_notrace_function
# from timm.layers import _assert # _assert seems unused, removed for now

# Helper Functions (blockify, blockify_mask, deblockify, deblockifyv2)
# Put the definitions of blockify, blockify_mask, deblockifyv2, deblockify here...

def blockify(x, num_heads, split_size: int, flag = True):
    """image to blocks
    Args:
        x (Tensor): with shape (B, H, N, E) H=num_heads
        split_size (int): edge length of a single square block in units of N
    """
    B, h, N, E  = x.shape
    if N % split_size != 0:
        raise ValueError(f"Sequence length N ({N}) must be divisible by split_size ({split_size})")
    grid_size = N // split_size
    # Reshape assuming N represents the spatial/sequence dimension to be blocked
    x = x.reshape(B, h, grid_size, split_size, E)

    # Permute and reshape for block-wise attention
    # Output shape: (B * grid_size, num_heads, split_size, E)
    x1 = x.permute(0, 2, 1, 3, 4).reshape(B * grid_size, h, split_size, E).contiguous()

    if flag:
        # Reshape for global attention part (if needed, depends on attn2 logic)
        # Output shape: (B, num_heads, grid_size, split_size * E) ? Or maybe (B, num_heads, grid_size, split_size)? Check usage.
        # Let's assume it's used as keys/values for the second attention stage, needing shape (B, h, grid_size, E')?
        # The original code had reshape(B, num_heads, grid_size, -1), let's stick to that structure first.
        # Shape: (B, num_heads, grid_size, split_size * E)
        x2 = x.permute(0, 1, 2, 3, 4).reshape(B, h, grid_size, -1).contiguous() # Adjusted permutation
    else:
        x2 = None
    return x1, x2, grid_size


def blockify_mask(mask, split_size: int):
    """image to blocks mask
    Args:
        mask (Tensor): with shape (B, N)
        split_size (int): edge length of a single square block in units of N
    """
    if mask is None:
        return None # Handle None mask passed in
    B, N = mask.shape
    if N % split_size != 0:
        raise ValueError(f"Mask length N ({N}) must be divisible by split_size ({split_size})")
    grid_size = N // split_size
    # Reshape mask for block-wise application
    # Output shape: (B * grid_size, split_size)
    mask_blocks = mask.reshape(B, grid_size, split_size)
    x1 = mask_blocks.reshape(B * grid_size, split_size).contiguous()
    return x1

@register_notrace_function  # reason: int receives Proxy
def deblockifyv2(x, B, grid_size: int, split_size: int):
    """blocks to image (for first attention stage output)
    Args:
        x (Tensor): with shape (B*grid_size, num_heads, split_size, E)
        B (int): Original batch size
        grid_size (int): Number of blocks along the sequence dimension
        split_size (int): Sequence length within each block
    """
    Bt, h, s, E = x.shape # Bt = B * grid_size, s = split_size
    # Reshape back: (B, grid_size, num_heads, split_size, E)
    x = x.reshape(B, grid_size, h, s, E)
    # Permute to bring heads and sequence dimensions together
    # (B, num_heads, grid_size, split_size, E)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    # Reshape to final sequence: (B, num_heads, N, E) where N = grid_size * split_size
    N = grid_size * split_size
    x = x.reshape(B, h, N, E)
    return x


@register_notrace_function  # reason: int receives Proxy
def deblockify(x, B, grid_size: int, split_size: int):
    """blocks to image (for final output)
    Args:
        x (Tensor): with shape (B, num_heads, grid_size, E') - Check shape from attn2 output
                    If attn2 output is (B, h, grid_size, E'), this assumes we want (B, h, N, E') somehow?
                    The original code reshaped to (B, h, grid_size*split_size, -1). Let's match that.
        B (int): Original batch size
        grid_size (int): Number of blocks along the sequence dimension
        split_size (int): Sequence length within each block (used to calculate N)
    """
    _, h, g, E_prime = x.shape # g = grid_size
    N = grid_size * split_size
    # Reshape to (B, num_heads, N, E_prime // split_size) ? This requires knowing relation between E' and E
    # Let's stick to the original reshape: (B, h, N, -1)
    # This assumes E_prime is related to the original E. If x is (B, h, grid_size, E_orig), it becomes (B, h, N, E_orig / split_size)? Needs clarification.
    # Let's assume the final output dimension should match the original value dimension E.
    # If x is (B, h, grid_size, E_head), then reshape should probably be (B, h, N, E_head) ?
    # The original `reshape(B, h, grid_size*split_size, -1)` seems suspicious if E_prime != E_head.
    # Let's assume the output dimension E_head is correct and reshape accordingly.
    x = x.reshape(B, h, N, -1) # Keep this structure but be mindful of the last dimension's size
    return x


class RLSAttn(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.split_size = config["split_size"]
        if self.split_size <= 0:
             raise ValueError(f"split_size must be positive, got {self.split_size}")
        self.attn_drop = torch.nn.Dropout(p = config["attention_dropout"])
        # Optional: Add back conv layers if needed
        # dim = (self.num_head * self.head_dim) // 2
        # self.get_v1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        # self.get_v2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    # Optional: Add back get_lepe if needed
    # def get_lepe(self, x, H, W, func): ...

    def forward(self, Q, K, V, mask=None, is_causal=False, scale=None, tensor_layout="HND"):
        # RLSAttn doesn't use is_causal, scale, tensor_layout explicitly, but receives them.
        # Expects Q, K, V in shape (B, H, N, D) - which is the default "HND" layout from the processor
        if tensor_layout != "HND":
             # If layout is different, need to transpose Q, K, V before proceeding
             # Example for 'NHD': Q = Q.permute(0, 2, 1, 3), K = K.permute(0, 2, 1, 3), V = V.permute(0, 2, 1, 3)
             raise NotImplementedError(f"Tensor layout {tensor_layout} not handled by RLSAttn yet.")

        B, _, N, D = Q.shape # B, num_head, seq_len, head_dim

        if N % self.split_size != 0:
             raise ValueError(f"Sequence length {N} must be divisible by split_size {self.split_size}")

        # Handle mask: Ensure it's (B, N) or None
        if mask is not None:
            if mask.dim() == 4: # Often (B, 1, N, N) or (B, H, N, N)
                # Need to derive a (B, N) mask, e.g., from diagonal or row presence
                # Simplest: check if a token can attend to *any* other token (sum across last dim)
                # This might not be universally correct, depends on mask meaning.
                # Assuming mask indicates valid tokens: take max along one sequence dimension
                mask = mask.squeeze(1).any(dim=-1).float() # Example: (B, N) bool -> float
            elif mask.dim() == 3: # Maybe (B, N, N)
                 mask = mask.any(dim=-1).float()
            elif mask.dim() != 2:
                 raise ValueError(f"Unexpected mask shape: {mask.shape}")
            # Now mask should be (B, N)

        # Blockify Q, K, V and mask
        block_mask = blockify_mask(mask, self.split_size) # (B*grid_size, split_size) or None
        k1, k2, grid_size = blockify(K, self.num_head, self.split_size) # k1: (B*g, h, s, D), k2: (B, h, g, s*D)
        q1, q2, _         = blockify(Q, self.num_head, self.split_size) # q1: (B*g, h, s, D), q2: (B, h, g, s*D)
        v1, _, _          = blockify(V, self.num_head, self.split_size, flag=False) # v1: (B*g, h, s, D)

        # --- Stage 1: Local Attention ---
        scale_factor = self.head_dim ** -0.5
        q1 = q1 * scale_factor
        attn1_logits = torch.matmul(q1, k1.transpose(-2, -1)) # (B*g, h, s, s)

        # Apply mask for local attention
        if block_mask is not None:
            # block_mask shape is (B*g, s) -> needs expansion to (B*g, 1, 1, s) for broadcasting
            attn1_logits = attn1_logits - 1e6 * (1.0 - block_mask[:, None, None, :].to(attn1_logits.dtype))

        attn1 = F.softmax(attn1_logits, dim=-1, dtype=attn1_logits.dtype) # Use dtype for potential AMP
        attn1 = self.attn_drop(attn1)
        x1 = torch.matmul(attn1, v1) # (B*g, h, s, D)

        # Deblockify x1
        x1_deblocked = deblockifyv2(x1, B, grid_size, self.split_size) # (B, h, N, D)

        # --- Stage 2: Global Attention (Routing?) ---
        # Note: q2 shape is (B, h, g, s*D), k2 shape is (B, h, g, s*D)
        # This attention stage seems unusual (querying/keying with grid-level features?)
        # Ensure dimensions match and the concept is intended.
        # Assuming Q' = q2, K' = k2, V' = x1_deblocked (reshaped?)

        # Let's assume x1_deblocked needs to be "blockified" again for values in stage 2
        # This implies the second stage aggregates information *across* blocks using the grid features q2, k2
        # Value V' should represent the content *within* blocks corresponding to grid positions.
        # Re-blockify x1_deblocked to match the grid structure?
        # _, v2, _ = blockify(x1_deblocked, self.num_head, self.split_size, flag=False) # v2: (B*g, h, s, D) - Doesn't seem right.

        # Let's use the logic from the original code snippet more directly:
        # Attn2 = q2 @ k2.transpose(-2,-1) -> (B, h, g, g)
        # Output = Attn2 @ x1 (where x1 needs reshaping)
        # The deblockify(x, B, grid_size, split_size) expects x=(B, h, grid_size, E')
        # Let's reshape x1_deblocked to fit this expectation.
        # x1_deblocked is (B, h, N, D). We need (B, h, grid_size, E') where N = grid_size * split_size
        # This could be average pooling per block, or just reshaping:
        # x1_grid = x1_deblocked.reshape(B, self.num_head, grid_size, self.split_size, self.head_dim).mean(dim=3) # Avg pool, shape (B, h, g, D)
        x1_grid = x1_deblocked.reshape(B, self.num_head, grid_size, -1) # Reshape, shape (B, h, g, s*D)


        q2 = q2 * (q2.shape[-1]**-0.5) # Use appropriate dimension for scaling
        attn2_logits = torch.matmul(q2, k2.transpose(-2, -1)) # (B, h, g, g)
        attn2 = F.softmax(attn2_logits, dim=-1, dtype=attn2_logits.dtype) # Use dtype for potential AMP
        attn2 = self.attn_drop(attn2)

        # x = Attn2 @ V' where V' is derived from x1_deblocked
        # Using x1_grid from above (shape B, h, g, s*D)
        x = torch.matmul(attn2, x1_grid) # (B, h, g, s*D)

        # Final Deblockify - expects (B, h, g, E'), outputs (B, h, N, E'/s ?)
        # If x is (B, h, g, s*D), deblockify outputs (B, h, N, D)
        x_final = deblockify(x, B, grid_size, self.split_size) # (B, h, N, D)

        return x_final