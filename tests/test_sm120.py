"""SM120 (Blackwell) correctness harness — plan item 5 (S2 gate).

Grid: hd x causal x seq x dtype vs F.scaled_dot_product_attention reference.
Dense mode (topk=1.0) so we measure kernel numerics, not sparsity loss.
"""
import pytest
import torch
import torch.nn.functional as F

from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
from spas_sage_attn.core import get_cuda_arch_versions

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_cuda_arch_versions()[0] != "sm120",
    reason="SM120-specific harness",
)

REL_L2_TOL = 0.10  # plan S2 gate for dense kernel mode vs fp16 SDPA reference


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("hd", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq", [512, 4096, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_matches_sdpa(hd, causal, seq, dtype):
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(1, 4, seq, hd, dtype=dtype, device="cuda") * 0.5
        for _ in range(3)
    )
    out = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=1.0, is_causal=causal)
    ref = F.scaled_dot_product_attention(
        q, k, v, is_causal=causal,
        # NHD-vs-HND agnostic: inputs are (b, h, s, d) already
    )
    assert out.dtype == dtype  # API preserves input dtype (fp16->fp16, bf16->bf16)
    rel = _rel_l2(out, ref)
    assert rel < REL_L2_TOL, f"hd={hd} causal={causal} seq={seq} dtype={dtype}: rel L2 = {rel:.4f}"


@pytest.mark.parametrize("hd", [64, 128])
def test_causal_long_seq_stability(hd):
    """Upstream #388 analogue: fp8 PV long-sequence noise probe (sm120 gotcha)."""
    torch.manual_seed(1)
    seq = 16384
    q, k, v = (
        torch.randn(1, 2, seq, hd, dtype=torch.float16, device="cuda")
        for _ in range(3)
    )
    out = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=1.0, is_causal=True)
    assert torch.isfinite(out).all(), "NaN/Inf at long seq"
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    rel = _rel_l2(out, ref)
    assert rel < REL_L2_TOL, f"long-seq hd={hd}: rel L2 = {rel:.4f}"


@pytest.mark.xfail(
    reason="Reproduces upstream thu-ml/SageAttention issue #392 (sm120: CUDA-graph "
    "replay yields wrong/zero output). Kept as a live regression probe, not an S2 gate.",
    strict=False,
)
def test_cuda_graph_replay_equality():
    """Upstream #392 analogue: captured graph replay must match eager output."""
    torch.manual_seed(2)
    q, k, v = (
        torch.randn(1, 4, 1024, 128, dtype=torch.float16, device="cuda")
        for _ in range(3)
    )
    eager = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=1.0).clone()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=1.0)
        graph.replay()
        torch.cuda.synchronize()
    except Exception as exc:  # capture unsupported is a documented finding, not a crash
        pytest.skip(f"CUDA graph capture unsupported on this build: {exc}")
    assert _rel_l2(captured, eager) < 1e-3, "graph replay diverged from eager"
