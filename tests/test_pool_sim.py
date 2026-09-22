import ast
import importlib.util
import inspect
from pathlib import Path

import torch


def test_partial_block_pool_ignores_padded_rows():
    spec = importlib.util.spec_from_file_location(
        "sparge_utils", Path(__file__).resolve().parents[1] / "spas_sage_attn" / "utils.py"
    )
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    for kernel in (
        utils.triton_bmm_pool_sim_simmean,
        utils.triton_bmm_pool_sim_simmean_fuse_quant,
    ):
        loads = [
            node for node in ast.walk(ast.parse(inspect.getsource(kernel.fn)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and any(keyword.arg == "mask" for keyword in node.keywords)
        ]
        assert len(loads) == 1
        assert all(
            any(keyword.arg == "other" and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == 0.0 for keyword in load.keywords)
            for load in loads
        )

    x = torch.ones((1, 1, 100, 64), device="cuda", dtype=torch.float16)
    threshold = torch.zeros(1, device="cuda")
    for pool, _ in (
        utils.get_pool_sim_triton_simmean(x, 64, threshold),
        utils.get_pool_sim_triton_simmean_fuse_quant(x, None, 64, threshold)[:2],
    ):
        torch.testing.assert_close(pool, torch.ones_like(pool))
