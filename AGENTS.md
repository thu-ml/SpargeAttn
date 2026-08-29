# AGENTS.md — SM120 (Blackwell) Support Work

Active plan: `.omo/plans/sm120-blackwell-support.md` (under Momus review; goal is **maximize SM120 hardware performance**, not merely compile-and-run).

## Hard user directives (2026-08-29, binding for all agents)

- Confine ALL work to `/work/SpargeDev`.
- Python: `/work/SpargeDev/.venv` ONLY. venv package installs are pre-approved (ninja, pytest, etc.).
- **NEVER downgrade torch or CUDA.** Local stack is fixed: torch 2.13.0+cu130, CUDA toolkit 13.2.
- GPU builds/runs on this box: standing approval while user is away ("go as far as you can").
- Fill knowledge gaps with MCP tools (cuda-docs, context7, deepwiki) — do not guess, and do not give up on a lookup; retry with corrected names.
- **SageAttention3 does NOT work on sm120 per user's firsthand experience** — treat any upstream claim of SA3-on-consumer-Blackwell as unverified; FP4 work is decision-only (Phase 3) and evidence must come from local probes.

## Local environment (preflight-verified 2026-08-29)

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition — CC **12.0**, **188 SM**, 96 GB, driver 610.43.02 |
| Clocks | torch `clock_rate` 2617 MHz; whitepaper boost ~2417 MHz |
| smem | 100 KB/SM, 99 KB/block opt-in (torch reports 101376 B) |
| Python | `.venv` py 3.12.3; torch **2.13.0+cu130**; triton 3.7.1; pytest NOT installed (install into .venv) |
| CUDA | nvcc/ptxas/cuobjdump **13.2** at `/usr/local/cuda`; accepts `sm_120/120a/120f/121/121a` (compile-probed OK) |
| Env | `TORCH_CUDA_ARCH_LIST=12.0` already exported — current setup.py rejects it at configure (expected pre-fix) |
| Build state | none — `build/` absent, `spas_sage_attn._qattn` unimportable |

**CUDA 13 gotcha (probe-verified):** `csrc/mma.cuh` etc. fail standalone compile on CUDA 13.2 — CUDA 13 dropped transitive `<cstdint>` (`uint32_t` undefined, `cuda_fp8.h` internals break). Fix = explicit `#include <cstdint>` (plan item 2b). Related upstream: SpargeAttn issue #115 (CUDA 13 build failure), PR #45 (assert.h workaround pattern).

**Instruction probes (sm_120 + sm_120a both OK, real SASS emitted):** `IMMA.16832.S8.S8` (int8 QK), `QMMA.16832.F16.E4M3.E4M3` (fp16-accum fp8 PV — Sage2++ peak-rate path), `QMMA.16832.F32.E4M3.E4M3` (half-rate). `120a` suffix needed only for FP4.

## MCP resources — USE THEM, don't assume they're unavailable

- **deepwiki**: this repo mirrors `thu-ml/SpargeAttn` (exact name — NOT "SpargeAttention"): https://deepwiki.com/thu-ml/SpargeAttn — also `thu-ml/SageAttention`. Tools: `ask_question`, `read_wiki_structure`, `read_wiki_contents`.
- **cuda-docs**: `search_cuda_docs` — neural search over current NVIDIA CUDA/PTX ISA corpus (covers sm_120/sm_120a instructions that post-date training cutoffs).
- **context7**: `resolve-library-id` → `query-docs` for torch/setuptools API docs.

## Verified code anchors (Momus-checked, ±4 lines)

| Location | What |
|---|---|
| `setup.py` L52 | `SUPPORTED_ARCHS = {"8.0","8.6","8.7","8.9","9.0"}` — no `12.0` |
| `setup.py` L88–133 | `get_torch_arch_list()` + capability detection |
| `setup.py` L142–152 | CUDA-toolkit version-gate pattern (raise RuntimeError if too old) — sm120 needs an added gate, `compute_120` requires **CUDA >= 12.8** |
| `setup.py` L153–184 | gencode loop, `HAS_SM90` / `-DHAS_SM90`, source assembly; `SAGE2PP_ENABLED` at ~L149 |
| `csrc/mma.cuh` L44–48 | `MMA_F8F8F32_M16N8K16_ENABLED` — requires CUDA >= 12.4 && `__CUDA_ARCH__ >= 890` |
| `spas_sage_attn/autotune.py` L136–144 | `kernel_selection`: any `sm >= 89` → fp8 (SM89-style) kernel |
| `csrc/qattn/decl.cuh` L24–46 | `SpargeAttentionSM8{0,9}Dispatched` decls |
| `csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh` L758 | `SpargeAttentionSM89Dispatched` def |
| `csrc/qattn/instantiations_sm89/autogen.py` | instantiation generation; flows through shared `NVCC_FLAGS` (setup.py ~L180) |
| `csrc/wgmma.cuh` L322–337 | `SM90_ENABLED`-gated wgmma (Hopper-only) |

## Established technical facts

- SM120 = Blackwell **consumer/workstation**; implements **neither** `wgmma` (sm_90a) **nor** `tcgen05` (sm_100) → correct kernel family is the **SM89-style `mma.sync` int8-QK/fp8-PV** path.
- `__CUDA_ARCH__` on sm_120 is `1200`, so existing `>= 890` guards already pass — check guards for upper-bounds (`< 900`-style) that would exclude it instead.
- Upstream README already claims "CUDA >= 12.8 for Blackwell"; upstream has `instantiations_sm90/` + `HAS_SM90` machinery but (per deepwiki snapshot) no `instantiations_sm120/` — re-verify upstream for newer sm120 work before porting.
- README base env: `python>=3.9`, `torch>=2.3.0`; build via `python setup.py install` (local `.venv`, never system-wide python).

## Research findings (all complete 2026-08-29 — details in plan's Background section)

1. **Upstream**: `thu-ml/SpargeAttn` has NO sm120 support (unmerged build-only PR #45). `thu-ml/SageAttention` solved it: merged PR #109 = SUPPORTED_ARCHS+12.0/12.1, CUDA≥12.8 gate, `compute_120a` gencode, sm89 kernel family compiled for sm_120, dispatch rule sm120→int8-QK/fp8-PV per-warp. Bug watch-list: #388 (fp8 noise >160k seq), #392 (CUDA graph), #378/#379 (hardcoded SM-count grid — NEVER hardcode).
2. **HW spec**: sm_120 per-SM-per-clk == sm_89 for fp16/int8/fp8; FP4 (2× fp8) is new, needs sm_120a; fp8 with fp32 accumulator = HALF rate on both archs → use fp16-accum PV (`use_pv_fp16_accu` flag, `MMA_F8F8F16_M16N8K16`). wgmma/tcgen05 absent on sm_120.
3. **Repo**: tile knobs duplicated between autogen.py and .cu wrappers (`qk_int_sv_f8_cuda_sm89.cu:140-143/326-329/521-524`); autotune.py tunes sparsity thresholds only, never tiles; `>= 900` guards in wgmma/sm90 .cuh numerically admit 1200 (clamp to <1000); guard census all-pass table in plan.
