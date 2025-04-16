import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models import CogVideoXTransformer3DModel

# --- Attempt to import RLSAttn ---
# Ensure rls_attn.py is in the same directory or PYTHONPATH
try:
    from spas_sage_attn.rls_attn import RLSAttn
    # Ensure RLSAttn itself inherits from nn.Module if it has parameters or submodules
    if not issubclass(RLSAttn, nn.Module):
        print("Warning: Imported RLSAttn does not inherit from nn.Module. This might cause issues.")
except ImportError:
    print("*"*40)
    print("Warning: Could not import RLSAttn from rls_attn.py.")
    print("Using a dummy RLSAttn as a fallback.")
    print("Please ensure rls_attn.py exists and is correctly implemented.")
    print("*"*40)
    # Define a dummy class if import fails, to allow script execution
    class RLSAttn(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__() # Still needs super().__init__!
            print("Dummy RLSAttn Initialized.")
        def forward(self, q, k, v, mask=None, is_causal=False, scale=None, tensor_layout="HND"):
            print("WARNING: Executing dummy RLSAttn forward pass (using standard attention).")
            # Fallback to standard attention
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

# --- Original SAGE/SparseAttention Imports (Optional: Keep if needed for comparison/other modes) ---
try:
    from spas_sage_attn.autotune import SparseAttentionMeansim, extract_sparse_attention_state_dict, load_sparse_attention_state_dict
except ImportError:
    # print("Note: spas_sage_attn not found, SAGE attention features unavailable.")
    SparseAttentionMeansim = None # Define as None if not available
    extract_sparse_attention_state_dict = None
    load_sparse_attention_state_dict = None

# --- Original SAGE Processor and Setup Function (Optional: Keep if needed) ---
class SageAttnCogVideoXAttnProcessor(nn.Module): # Ensure this also inherits nn.Module and calls super().__init__ if used
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization. (Original from user code)
    """
    def __init__(self, idx, ):
        super().__init__() # Add if this processor is actually used
        self.idx = idx
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

        assert attention_mask is None, "Attention mask is not supported by original Sage processor example" # Original constraint

        text_seq_length = encoder_hidden_states.size(1)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = hidden_states.shape

        # Attention mask handling would be needed here if supported

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

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention: # CogVideoX attn1 is self-attention
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # Assumes inner_attention calls F.scaled_dot_product_attention or equivalent
        # If SparseAttentionMeansim is set, this calls its forward method
        hidden_states = attn.inner_attention(query, key, value, is_causal=False) # Mask handling missing in original example

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Split output
        encoder_hidden_states_out, hidden_states_out = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states_out, encoder_hidden_states_out


def set_spas_sage_attn_cogvideox(
    model: CogVideoXTransformer3DModel,
    verbose=False,
    l1=0.06,
    pv_l1=0.065
):
    """Sets up SparseAttentionMeansim (Original from user code)."""
    if SparseAttentionMeansim is None:
        print("Cannot set SAGE attention, SparseAttentionMeansim not available.")
        return

    for idx, block in enumerate(model.transformer_blocks):
        if hasattr(block, 'attn1') and isinstance(block.attn1, Attention):
            if verbose: print(f"Setting SAGE attention in block {idx}")
            block.attn1.verbose = verbose # Assuming SparseAttentionMeansim uses this
            # Instantiate SparseAttentionMeansim
            block.attn1.inner_attention = SparseAttentionMeansim(l1=l1, pv_l1=pv_l1)
            # Get the original processor (likely AttnProcessor2_0)
            origin_processor = block.attn1.get_processor()
            # Set the custom SAGE processor
            processor = SageAttnCogVideoXAttnProcessor(idx, )
            block.attn1.set_processor(processor)
            # Store the original one if needed
            if not hasattr(block.attn1, "origin_processor"):
                block.attn1.origin_processor = origin_processor
        else:
             if verbose: print(f"Skipping block {idx}, no attn1 or not Attention type.")


# --- RLSAttn Processor (Corrected) ---

class RLSAttnCogVideoXAttnProcessor(nn.Module): # Inherits from nn.Module
    r"""
    Attention processor for CogVideoX wrapping RLSAttn.
    Adapts the interface, applies RoPE, handles text conditioning concatenation,
    and passes the mask correctly.
    """
    def __init__(self):
        # ***** FIX: Call the parent class initializer *****
        super().__init__()
        # ************************************************

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("RLSAttnCogVideoXAttnProcessor requires PyTorch 2.0 components.")
        # No specific state needed in the processor itself usually

    def __call__(
        self,
        attn: Attention, # The Attention module (e.g., block.attn1)
        hidden_states: torch.Tensor, # Input image features (B, N_img, D_model)
        encoder_hidden_states: Optional[torch.Tensor] = None, # Input text features (B, N_text, D_model)
        attention_mask: Optional[torch.Tensor] = None, # Mask (potentially needs processing)
        image_rotary_emb: Optional[torch.Tensor] = None, # RoPE embeddings
        **kwargs, # Catch potential extra args like scale
    ) -> torch.Tensor:
        """
        Args:
            attn: The diffusers Attention module instance.
            hidden_states: Image token embeddings.
            encoder_hidden_states: Text token embeddings (optional).
            attention_mask: Mask (may be None or need broadcasting).
            image_rotary_emb: Rotary position embeddings for image tokens.

        Returns:
            A tuple containing:
             - hidden_states_out: The output corresponding to the image tokens. (B, N_img, D_model)
             - encoder_hidden_states_out: The output corresponding to the text tokens, or None. (B, N_text, D_model) or None
        """

        input_ndim = hidden_states.ndim
        if input_ndim == 4: # Handle B C H W input format if necessary
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            # Assuming encoder_hidden_states is already (B, N, D)

        batch_size, sequence_length_img, _ = hidden_states.shape

        # --- 1. Handle Text Conditioning ---
        text_seq_length = 0
        if encoder_hidden_states is not None:
            text_seq_length = encoder_hidden_states.shape[1]
            # Concatenate along the sequence dimension
            combined_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        else:
            combined_hidden_states = hidden_states

        # --- 2. Project Q, K, V ---
        query = attn.to_q(combined_hidden_states)
        key = attn.to_k(combined_hidden_states)
        value = attn.to_v(combined_hidden_states)

        # --- 3. Reshape and Norm (if applicable) ---
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # (B, H, N_combined, D_head)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)   # (B, H, N_combined, D_head)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # (B, H, N_combined, D_head)

        if attn.norm_q is not None: query = attn.norm_q(query)
        if attn.norm_k is not None: key = attn.norm_k(key)

        # --- 4. Apply RoPE to Image Part ---
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb # Local import
            # Apply only to the image sequence part (indices >= text_seq_length)
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            # CogVideoX attn1 is self-attention, so apply to keys too
            key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)


        # --- 5. Call the Custom Inner Attention (RLSAttn) ---
        # Ensure Q, K, V are in the expected layout (B, H, N, D_head) - "HND"
        # Pass the attention_mask directly; RLSAttn needs to handle its format.
        hidden_states = attn.inner_attention(
            query, key, value,
            mask=attention_mask, # Pass the original mask
            is_causal=False,     # RLSAttn might ignore this, pass False for self-attn
            tensor_layout="HND"  # Inform RLSAttn of the layout
        )
        # Expected output shape: (B, H, N_combined, D_head)

        # --- 6. Reshape Output and Apply Output Projection ---
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype) # Ensure correct dtype

        hidden_states = attn.to_out[0](hidden_states) # Linear projection
        hidden_states = attn.to_out[1](hidden_states) # Dropout

        # --- 7. Split Output if Text Conditioning Was Used ---
        if encoder_hidden_states is not None:
            # Split based on original text sequence length
            encoder_hidden_states_out, hidden_states_out = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
            # Return image part, text part (matching Sage processor's signature)
            return hidden_states_out, encoder_hidden_states_out
        else:
            # No text conditioning, return image part and None for text part
            return hidden_states, None


# --- Setup Function for RLSAttn ---

def set_rls_attn_cogvideox(
    model: CogVideoXTransformer3DModel,
    split_size: int,
    attention_dropout: float = 0.0,
    verbose: bool = False,
):
    """Replaces the inner attention mechanism in CogVideoX's attn1 blocks with RLSAttn."""
    if verbose:
        print(f"Setting RLSAttn with split_size={split_size}, dropout={attention_dropout}")

    total_attn1_replaced = 0
    for idx, block in enumerate(model.transformer_blocks):
        # Target the self-attention block, typically named 'attn1'
        if hasattr(block, 'attn1') and isinstance(block.attn1, Attention):
            if verbose: print(f"Modifying attn1 in block {idx}")

            # --- Get config needed for RLSAttn ---
            num_heads = block.attn1.heads
            # Infer inner_dim (total dimension before splitting heads) from the projection layer
            inner_dim = block.attn1.to_q.out_features # Or to_k.out_features
            if inner_dim is None: # Fallback if layer structure is unexpected
                 inner_dim = block.attn1.to_out[0].in_features
            head_dim = inner_dim // num_heads

            # Validate dimensions
            if inner_dim % num_heads != 0:
                 raise ValueError(
                     f"Block {idx}: inner_dim ({inner_dim}) must be divisible by "
                     f"num_heads ({num_heads})"
                 )
            if head_dim <= 0:
                 raise ValueError(f"Block {idx}: Calculated head_dim ({head_dim}) is not positive.")

            # --- Create RLSAttn Config ---
            rls_config = {
                "head_dim": head_dim,
                "num_head": num_heads,
                "split_size": split_size,
                "attention_dropout": attention_dropout,
            }
            if verbose: print(f"  RLSAttn config: {rls_config}")

            # --- Instantiate RLSAttn ---
            try:
                rls_attn_instance = RLSAttn(rls_config)
            except Exception as e:
                print(f"ERROR: Failed to instantiate RLSAttn for block {idx} with config {rls_config}")
                raise e

            # --- Replace Inner Attention ---
            # The 'inner_attention' attribute is dynamically used by processors like AttnProcessor2_0
            # We replace the *module* assigned to inner_attention.
            block.attn1.inner_attention = rls_attn_instance

            # --- Set the Correct Processor ---
            # We need our custom processor that knows how to interact with CogVideoX structure (RoPE, text concat)
            # and then call the RLSAttn instance via attn.inner_attention.
            processor = RLSAttnCogVideoXAttnProcessor()
            block.attn1.set_processor(processor)

            total_attn1_replaced += 1
            # Optional: Store original processor if you need to switch back easily
            # if not hasattr(block.attn1, "origin_processor"):
            #     block.attn1.origin_processor = block.attn1.processor # Store the processor itself
        else:
             if verbose: print(f"Skipping block {idx}: No 'attn1' attribute or it's not an Attention instance.")

    if total_attn1_replaced == 0:
        print("WARNING: No 'attn1' blocks were found and replaced with RLSAttn. Check the model structure.")
    elif verbose:
        print(f"Successfully replaced inner attention in {total_attn1_replaced} attn1 blocks with RLSAttn.")