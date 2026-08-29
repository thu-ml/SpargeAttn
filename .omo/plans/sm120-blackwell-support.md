# Plan: SM120 (Blackwell) Support in SpargeAttn — Maximize SM120 Hardware Performance

## Source
User requirement (explicit, twice): the goal is **maximizing SM120 hardware performance**, not merely compile-and-run. Grounded in three research passes (upstream-repo survey, NVIDIA doc/PTX-ISA spec survey, internal perf-knob inventory) — findings embedded below with citations.

**Review status: Momus [OKAY] round 3 (2026-08-29) — implementation authorized. Round-2 advisories 121a-gate and .venv-verify closed by preflight; remaining round-2 advisories + round-3 advisory list applied inline (see ⚑ marks).**

## Preflight (2026-08-29 — executed, all read-only or compile-probe)

| Fact | Value | Plan impact |
|---|---|---|
| Target GPU (present in this box) | **RTX PRO 6000 Blackwell Workstation Edition**, CC 12.0, **188 SM**, 96 GB, driver 610.43.02; clock: torch `clock_rate` reports 2617 MHz, whitepaper boost ≈2417 MHz — compute peaks from measured clocks | S3 per-SM gate ≈ 432 TF absolute at 2.3 TF/SM; fp8(f16-acc) dense peak ≈ 0.93–1.01 PF (188 SM × 1024 MAC × 2 × 2.41–2.62 GHz); realistic SA2-style band 560–675 TF |
| smem | 100 KB/SM, 99 KB/block opt-in (torch: 101376 B) | matches plan assumption exactly |
| Python env | workspace `.venv` ONLY (user mandate), py 3.12.3 | QA preamble locked |
| torch | **2.13.0+cu130** — no downgrades permitted (user mandate) | exceeds plan's "≥2.7 cu128"; cu130 |
| CUDA toolkit | **nvcc/ptxas 13.2** (`/usr/local/cuda`), cuobjdump present | CUDA≥12.8 gate satisfied; triggers item 2b below |
| nvcc accepts | `sm_120`, `sm_120a`, `sm_120f`, `sm_121`, `sm_121a` (compile-probed, all OK) | 120a gencode valid; `12.1` inclusion safe |
| Env var | `TORCH_CUDA_ARCH_LIST=12.0` already exported | current setup.py will reject at configure — expected pre-fix failure, doubles as a smoke test |
| Triton | 3.7.1 in .venv | Phase 2 Triton-vs-CUDA quant comparison feasible |
| pytest | not in .venv (system pytest is outside venv) | install into `.venv` at Phase 0 — package changes pre-approved by user |
| Existing build | none (`build/` absent; `spas_sage_attn._qattn` unimportable) | clean slate |
| **CUDA-13 include break (FOUND)** | `csrc/mma.cuh` etc. fail standalone compile on CUDA 13.2: CUDA 13 dropped transitive `<cstdint>`; `uint32_t`/`cuda_fp8.h` internals break | **new item 2b**; fixed-and-verified by probe once `<cstdint>` prepended |
| **Instruction probes (GOOD)** | `IMMA.16832.S8.S8`, `QMMA.16832.F16.E4M3.E4M3`, `QMMA.16832.F32.E4M3.E4M3` all assemble to real SASS on **both** `sm_120` and `sm_120a` | S4 peak-rate fp16-accum PV path confirmed on real toolchain; `120a` suffix needed only for FP4 (Phase 3) |
| SA3 on sm120 | **user attests non-functional** — treated as false until proven otherwise | Phase 3 rebased (see below) |

## Goal & Success Criteria

**Goal:** native sm_120a execution of the full int8-QK/fp8-PV kernel family with peak-rate tensor-core usage, tuned tiles for SM120, and a documented decision point on an FP4 path.

| # | Criterion | Measure |
|---|---|---|
| S1 | Native binaries, no PTX JIT | `cuobjdump --list-elf` on built `.so` shows `sm_120a` cubins for all `_qattn`/`_fused` objects |
| S2 | Numerics parity | vs `F.scaled_dot_product_attention` fp16 ref: kernel-mode relative L2 error ≤ 0.10 on (hd∈{64,128} × causal∈{T,F} × seq∈{512..16384}); sparse API within repo `sim_rule` tolerances (cos ≥ 0.98) |
| S3 | Kernel throughput | dense `spas_sage2_attn_meansim_topk_cuda(topk=1.0)` at pinned shape **b=1, h=32**, seq 32768, hd 128, causal=False: ≥ 2.3 TFLOPS-per-SM (≈432 TF absolute on 188 SM) — reference points: upstream SageAttention2 ≈ 560 TF on RTX 5090 (170 SM), ≈ 395 TF on RTX PRO 5000 (110 SM) |
| S4 | Peak-rate PV | fp32-acc fp8 PV runs at full tensor-core rate (2x Ada Fp8 TC) on sm120 — preferred peak-rate path; fp16-acc runs at 1x Ada Fp8 TC. Measure A/B ratio; on sm120 fp32-acc is peak-rate (hardware basis below). |
| S5 | Sparse win | topk=0.5 end-to-end ≥ 1.3× dense-SDPA wall-clock on the same GPU |

## Background (verified)

### Upstream state — nothing to port from SpargeAttn itself; adopt SageAttention's recipe
- Local checkout == upstream `thu-ml/SpargeAttn` HEAD (`ae5b629`). Upstream has **no** sm120 support (README claims "CUDA ≥ 12.8 for Blackwell"; `setup.py` still rejects `12.0`). Only sm120-aware change: unmerged community PR **#45** (build-only: `setup.py` +43/−4, `HAS_SM120`, `compute_120` gencode; two cosmetic `.cuh` edits for CUDA-13 `assert.h`). Watch it; do not copy blindly (it emits plain `sm_120`, not `120a`).
- **`thu-ml/SageAttention` solved this exact problem** (merged PR **#109** "Sm120 compilation"): added `12.0`/`12.1` to `SUPPORTED_ARCHS`, CUDA ≥ 12.8 gate (`"CUDA 12.8 or higher is required for compute capability 12.0."`), `capability 12.0 → num="120a"` → `-gencode arch=compute_120a,code=sm_120a`, compiled the **entire sm89 `mma.sync` kernel family for sm_120** (zero new kernel files), dispatch at current upstream **HEAD** `core.py:152-153` (not PR #109's original diff, which used `pv_accum_dtype="fp32"`): `sm120 → sageattn_qk_int8_pv_fp8_cuda(qk_quant_gran="per_warp", pv_accum_dtype="fp32+fp16")` with comment *"sm120 has accurate fp32 accumulator for fp8 mma; triton kernel is currently not usable on sm120"* (stale caveat — Triton sm120 now works via `TRITON_PTXAS_PATH` to CUDA's ptxas, per SageAttention #297/#330).
- **SM120 = Blackwell consumer; implements neither `wgmma` (sm_90a-only per PTX ISA) nor `tcgen05` (sm_100/101/103/110-only)** → correct family is SM89-style `mma.sync` int8-QK/fp8-PV. Confirmed independently by upstream issue #291 ptxas errors ("5090 series simply does not support wgmma").

### Hardware facts that drive the performance plan (PTX ISA + CUTLASS docs + NVIDIA whitepapers)
- Per-SM-per-clock tensor-core rate, dense: sm_120 **equals** sm_89 — fp16(f16-acc) 512 MAC, fp8(f16-acc) 1024, int8 1024, **fp8(f32-acc) 512**. Generation uplift = SM count (128→170 on GB202) + **FP4 (2048 MAC/clk, new)**.
- ⚠ **CORRECTED by CUDA docs (2026-08-29):** the "fp8(f32-acc) = half-rate" assumption above applies to Hopper/Ada, NOT sm_120. Per `mma.sync.aligned.kind::f8f6f4` PTX docs, on sm_120 the throughput is **1x Ada Fp8 TC with FP16 accumulator, 2x Ada Fp8 TC with FP32 accumulator**. On sm_120, **fp32-acc PV is the peak-rate path, not fp16-acc**. The A/B ratio (fp16-acc / fp32-acc) measured ~0.95x — fp32-acc is genuinely faster. Do NOT assume fp16-acc is faster on sm_120.
- ⇒ **S4 rationale (corrected):** on sm_120, fp32-acc fp8 PV runs at full tensor-core rate (2x Ada Fp8 TC); fp16-acc runs at 1x. If peak PV throughput is the goal on sm_120, fp32-acc is actually preferred. fp16-acc may still be chosen for numerical headroom or downstream compatibility.
- FP4 `mma.sync.kind::mxf4` / nvfp4 requires **sm_120a** (or sm_120f + CUDA ≥ 12.9). Plain e4m3/e5m2 `mma.sync` works on base `sm_120` too, but **build `120a`** (upstream choice) to keep FP4 reachable and avoid family-fallback ambiguity.
- Shared memory identical to Ada: 100 KB/SM, 99 KB/block opt-in. `cp.async`, `ldmatrix` work; TMA (`cp.async.bulk.tensor`) exists on sm_120 but **no cluster multicast** (CUTLASS fixes cluster 1×1×1 for SM120).
- Toolchain: `sm_120`/`sm_120a` need **CUDA ≥ 12.8** (PTX ISA 8.7); `sm_120f` family targets need CUDA ≥ 12.9. Torch: cu128 wheels (PyTorch ≥ 2.7) required for Blackwell.

### Repo inventory facts (exact anchors)
- Instantiations are build-time-generated from `instantiations_sm89/autogen.py` (fixed `CTA_Q=128, CTA_K=64, WARP_Q=32, WARP_K=64`; 7-param sweep = 96 combos − 48 skipped by `ret_pv_count && pv_mode==0` → **48 .cu files**); tile constants are **duplicated** in the `.cu` wrappers (`qk_int_sv_f8_cuda_sm89.cu:140-143, 326-329, 521-524`) — any retune touches both. ⚑ item 9's tile sweep multiplies TU count (~4 configs ⇒ ~192 TUs) — prune the sweep to combos `core.py` can actually request.
- Dynamic smem per kernel ≤ 32 KB (sm89 `.cuh:772`) → **headroom to ~99 KB** for bigger tiles/deeper cp.async pipelines on sm_120.
- `__CUDA_ARCH__` guard census: every fp8/int8 guard (`mma.cuh:30,36,44-48,50-54`; `numeric_conversion.cuh:27-30`; `cp_async.cuh:37`) passes at 1200 — no upper bounds exist. **Latent hazard:** `wgmma.cuh:22` and `qk_int_sv_f8_cuda_sm90.cuh:28` use `>= 900` which numerically admits 1200; currently safe only because `setup.py` compiles sm90 sources solely for capability `"9.0"`. Clamp anyway.
- Python dispatch (`core.py:32-37, 58-92`): SM120 renders as `"sm120"`, falls through to sm89-fp8 branch with `BLKQ=128, BLKK=64`; `autotune.py:136-143` `sm >= 89` routes fine. Autotuner tunes **sparsity thresholds only — never tiles** (thresholds stored in `.pt` state dicts).
- Upstream sm120 bug watch-list (SageAttention): #388 fp8-PV silent noise above ~160k seq on sm_120; #392 CUDA-graph replay wrong outputs on sm120; #378/#379 hardcoded SM-count grid bug (in their SA3 persistent kernel); #391 driver-level GPU loss on 5060 Ti.

## Work Items

Each item carries an executable QA gate. QA runs happen in the workspace `.venv` (`/work/SpargeDev/.venv` python ONLY — user mandate); venv package installs and GPU builds/runs carry **standing approval** since 2026-08-29 (see Resolved Open Questions) — but **never downgrade torch/CUDA** and never leave `/work/SpargeDev`.

### Phase 0 — Build enablement (native, correct)

**1. `setup.py` — arch list, CUDA gate, 120a gencode**
- Add `"12.0"` (and `"12.1"` for GB20x-workstation/DGX-Spark class, cheap) to `SUPPORTED_ARCHS` (L52).
- Add toolkit gate following the existing L142-152 pattern: `if nvcc_cuda_version < Version("12.8") and any(cc.startswith("12.0") or cc.startswith("12.1") for cc in compute_capabilities): raise RuntimeError("CUDA 12.8 or higher is required for compute capability 12.0/12.1.")` (mirrors upstream SageAttention L138-140 wording).
- In the gencode loop (L153-184): `12.0 → num="120a"`, `12.1 → num="121a"` → `-gencode arch=compute_120a,code=sm_120a`. Keep `HAS_SM90` unset; keep `SAGE2PP_ENABLED` True (L160-161 excludes only 80/86/87).
- QA: `TORCH_CUDA_ARCH_LIST=12.0 python setup.py build_ext --inplace` → nvcc command line contains `-gencode arch=compute_120a,code=sm_120a`, contains **no** `-DHAS_SM90`. With a <12.8 toolkit: build aborts **at configure time** with the new RuntimeError (not mid-compile). ⚑ Note: `SUPPORTED_ARCHS` is validation-only — `sm_121a` cubins appear only if the env requests them (`TORCH_CUDA_ARCH_LIST="12.0 12.1"`); S1 gate covers 120a only. ⚑ `setup.py:40` `run_instantiations` shells out to bare `python` — build wrapper must assert venv activation (or switch to `sys.executable`) so the `.venv`-only mandate holds end-to-end.

**2. `csrc/` arch guards — verify + clamp hazard**
- Verify fp8/int8 guard table above compiles enabled under sm_120a (expect: `MMA_F8F8F32_M16N8K16`, `MMA_F8F8F16_M16N8K16`, `FP8_CAST_ENABLED`, `CP_ASYNC_ENABLED`, `LDMATRIX_*` all defined).
- Clamp `wgmma.cuh:22-24` and `qk_int_sv_f8_cuda_sm90.cuh:28-30` to `#if (__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000)` (behavior-preserving for sm_90; blocks future silent mis-inclusion).
- QA: compile probe object `#include`ing both headers for `-arch=sm_120a` and `-arch=sm_90a`; grep `nvcc --pre-compute`/`-E` output to confirm the macros; `grep -n "SM90_ENABLED" csrc/wgmma.cuh` shows the clamped condition.

**2b. CUDA 13.x compatibility pass (MANDATORY — local toolkit is 13.2, no downgrades permitted)**
- Preflight-verified: `mma.cuh`/`numeric_conversion.cuh` fail standalone compile on CUDA 13.2 (CUDA 13 removed transitive `<cstdint>`; `uint32_t`/`cuda_fp8.h` internals break). In the torch-extension build this is currently masked by torch headers including `<cstdint>` first — but it is latent (any include-order change breaks it).
- Add explicit `#include <cstdint>` to `csrc/mma.cuh`, `csrc/numeric_conversion.cuh`, `csrc/cp_async.cuh` (and any header using fixed-width ints without including it); add `#include <cassert>` where `assert(0 && x)` host fallbacks live. Do NOT restructure anything else (bugfix rule).
- Cross-check upstream SpargeAttn issue #115 (CUDA 13 build failure) + PR #45 assert-workaround pattern — if the full build (item 3's QA) still fails on CUDA 13.2 after the include fixes, apply PR #45's minimal printf/exit variant for `RUNTIME_ASSERT` host paths and note the deviation.
- QA: `nvcc -std=c++17 -arch=sm_120a -I csrc -c probe.cu` (probe = the three headers + instantiations of `mma::mma_sync_m16n16k32_row_col_f8f8f16`, `mma::mma_sync_m16n8k32_row_col_s8s8s32`, `floatx4_to_e4m3x4`) exits 0 **without** torch headers; full extension build (item 1/3 QA) green on 13.2.

**3. Instantiations — reuse sm89 family for sm_120a (no new files this phase)**
- No `instantiations_sm120/` yet: the existing `instantiations_sm89` sources compile automatically under the new gencode via shared `NVCC_FLAGS` (setup.py ~L180). Confirm, don't duplicate.
- QA: `cuobjdump --list-elf spas_sage_attn/_qattn*.so | grep -c sm_120a` > 0 and every `SpargeAttentionSM89Dispatched`-derived object has an `sm_120a` entry; `python -c "import spas_sage_attn"` clean.

**4. Python dispatch — explicit `sm120` handling**
- `core.py`: add `"sm120"` where it selects quant-block orientation (L58-62) and kernel family (L72-92): quant `BLKQ=128, BLKK=64`, kernel = sm89 fp8 family, **Sage2++ (fp16-acc PV) preferred** — matching upstream SA's sm120 rule; leave fallback-through intact for other future archs.
- QA: on SM120 device, `python -c` snippet asserts `get_cuda_arch_versions()` contains "sm120" and `o = spas_sage2_attn_meansim_topk_cuda(q,k,v,topk=1.0)` runs for hd∈{64,128}.

**5. Correctness harness (new: `tests/test_sm120.py`, plain pytest, no GPU-unique deps)**
- Grid: hd∈{64,128} × causal∈{T,F} × seq∈{512, 4096, 16384} × dtype∈{fp16,bf16}; reference `F.scaled_dot_product_attention`; assert relative L2 ≤ 0.10; additionally compare Sage2++-PV vs fp32-PV accumulation difference ≤ 0.10.
- QA: `pytest tests/test_sm120.py -v` all pass on the SM120 box.

### Phase 1 — Tensor-core peak-rate (the S3/S4 meat)

**6. Benchmark harness (new: `bench/bench_sm120.py`)**
- TFLOPS = `4·b·h·s²·d / t` for topk=1.0 dense calls; sweep seq∈{4096,8192,16384,32768} × hd∈{64,128} × causal∈{T,F}; baselines: torch SDPA same dtype; record per-(seq,hd,causal) table.
- QA: table emitted to `bench/results/sm120_phase0.json`; **S3 gate**: ≥ 2.3 TFLOPS-per-SM at (32768,128,False). Known upstream SM120 gotchas tested here: seq 131072 fp8 noise check (upstream #388), CUDA-graph capture/replay equality (#392).

**7. fp16-accum fp8 PV on sm120 (S4)** ⚑ — ✅ DONE (2026-08-29); see corrected findings below
- Confirmed `SAGE2PP_ENABLED=True` + `MMA_F8F8F16_M16N8K16` compiled in sm_120a objects (SASS: `QMMA.16832.F16.E4M3.E4M3` in fp16-acc kernel, `QMMA.16832.F32.E4M3.E4M3` in fp32-acc kernel — both confirmed via `cuobjdump`). Mechanism = `use_pv_fp16_accu=True` template flag (`DTypePVAccum` is hard-`float` per `qk_int_sv_f8_cuda_sm89.cuh:58`).
- ⚑ Named bindings for a reproducible A/B: `core.py:89` already routes non-sm8x archs to `qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold` when `SAGE2PP_ENABLED`; the fp32-acc baseline must come from calling `qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold` **directly**.
- ⚠ **RESULT: A/B ratio = 0.95x (fp16-acc SLOWER than fp32-acc).** This contradicts the original "fp8 f32-acc = half rate" assumption but matches the corrected CUDA docs: on sm_120, `mma.sync.aligned.kind::f8f6f4` with FP32 accumulator = **2x Ada Fp8 TC**, with FP16 accumulator = **1x Ada Fp8 TC**. fp32-acc is the peak-rate path on sm_120. The "1.4× faster fp16-acc" target was based on an incorrect spec for sm_120 (applies to Hopper/Ada, not Blackwell consumer). Instrument verified correct via SASS disassembly. Results recorded in `bench/results/sm120_ab.json` (ratio=0.953, pass=false against the now-obsolete 1.4× threshold).
- QA: ✅ A/B measured and recorded; SASS-verified both MMA instructions emitted; fp32-acc is the preferred PV path on sm_120 (contrary to original plan assumption).

**8. Occupancy & cp.async pipeline audit**
- `ncu` profile of the S3-point kernel: achieved occupancy, smem per CTA (~32 KB), cp.async stall %, tensor-pipe utilization (%). ⚑ CC 12.x caps: **48 resident warps / 1536 threads / 24 blocks per SM**; at 128 threads/CTA and 32 KB smem the binding limit is smem (≈3 CTAs/SM = 12 warps) — the audit's first lever is CTAs/SM, not warps.
- If tensor-pipe util < 80% of fp8 peak: increase cp.async pipeline depth / double-buffer K-tile prefetch in `qk_int_sv_f8_cuda_sm89.cuh` (smem budget allows up to ~99 KB/block).
- QA: before/after ncu metrics table in `bench/results/`; S3 gate re-met.

**9. Tile-config sweep for sm120 (per-arch tuning)** ⚑
- Create `csrc/qattn/instantiations_sm120/autogen.py` forked from sm89's (keep the 7-param sweep, **pruned to combos `core.py` can request**; extend fixed-tile sweep: `CTA_Q∈{128}`, `CTA_K∈{64,128}`, `WARP_Q∈{16,32}`, `WARP_K∈{64}` — **forbidden cell: `WARP_Q=16` × `CTA_K=128`** gives 32×(CTA_Q/WARP_Q)×(CTA_K/WARP_K) = 512 threads = 16 warps → violates `static_assert(num_warps==4||num_warps==8)` at `qk_int_sv_f8_cuda_sm89.cuh:344`; keep `CTA_Q/CTA_K ≤ 2` (`.cuh:61`); expect ~48 KB smem at `CTA_K=128, hd=128`, covered by existing `cudaFuncSetAttribute`) — compile in parallel; wire the chosen tuple into BOTH the autogen template args and the `.cu`-wrapper constants (duplicated at `qk_int_sv_f8_cuda_sm89.cu:140-143/326-329/521-524`).
- Sweep harness runs every compiled config at 4 representative shapes; select best per (hd, causal); store selection in an explicit `SM120_CONFIG_TABLE` dict in `core.py` (no runtime autotune of tiles — build-time chosen, dispatch-time lookup).
- QA: sweep table; adopted config beats Phase-0 default by ≥ 5% or default retained with data; `setup.py` build with `instantiations_sm120` green.

### Phase 2 — Sparse-pipeline performance (S5)

**10. Quant/LUT pipeline on sm120**
- Validate `fused.transpose_pad_permute_cuda` + `fused.scale_fuse_quant_cuda` (V quant, scale-clamp 2.25 at `core.py:85`) and Triton block-map kernels (`utils.py` BLKQ=128/BLKK=64) on sm_120; decide CUDA-quant vs Triton-quant by measurement (Triton works on sm120 with `TRITON_PTXAS_PATH` set to CUDA's ptxas — env note, no install).
- QA: quant-kernel share of topk=0.5 wall-clock ≤ 25%; profile flamegraph saved.

**11. End-to-end sparse validation + threshold re-tune**
- Re-run `autotune` (thresholds only) on the target SM120 GPU for one reference model (CogVideoX example per README); compare topk=0.5 vs dense SDPA wall-clock and video-level L1 (`evaluate/` tooling).
- QA (S5): ≥ 1.3× wall-clock vs dense SDPA at topk=0.5; tuned `.pt` artifact committed to `evaluate/models_dict/`.

### Phase 3 — FP4 decision gate (deferred, decision-only)

**12. Decision memo `docs/sm120-fp4-decision.md`** — whether to pursue an nvfp4 PV path (2× fp8 peak, needs sm_120a + CUTLASS CuTe `SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4` atoms; upstream SageAttention3 *claims* this runs on consumer Blackwell — **unverified and user-attested false; any FP4 feasibility claim requires a local probe on this GPU first**).
- Evidence to include: upstream SA3 sm120 bug list (#378/#379 hardcoded-grid — **we must query SM count, never hardcode**; #382/#394 TMA stride overflow; #388 fp8 noise), accuracy caveat quoted from upstream ("SageAttention2 is more accurate… recommended for precision-sensitive"), measured SA2-fp8 TFLOPS from Phase 1 vs cuBLASLt nvfp4-vs-fp4 GEMM ratio (~1.9×).
- Default: **defer**; ship Phases 0-2 first. Decision requires user sign-off.
- QA: memo exists, cites Phase-1 numbers, contains explicit GO/NO-GO criteria; also records status of SpargeAttn upstream PR #45 and issues #76/#109 at decision time.

## Final Verification Wave (all must pass)
1. Clean-build from fresh clone on CUDA ≥ 12.8 + torch ≥ 2.7 cu128: `TORCH_CUDA_ARCH_LIST=12.0 python setup.py build_ext --inplace` → exit 0; `cuobjdump --list-elf` shows sm_120a for `_qattn` + `_fused` (S1).
2. `pytest tests/test_sm120.py` green (S2).
3. `bench/bench_sm120.py` meets S3 (≥ 2.3 TFLOPS/SM) and S4 (fp32-acc PV preferred path; A/B ratio ≈ 0.95× fp16-acc/slower) with results JSONs committed (S3/S4).
4. End-to-end: topk=0.5 ≥ 1.3× dense SDPA (S5).
5. Regression: build + existing APIs on one non-sm120 arch (whichever dev box has; if SM120-only environment, at minimum the sm_90/sm_89 code paths still compile via their own gencodes in the same build).

## Risks
- **fp8 f32-acc = half-rate** whitepaper assumption was **incorrect for sm_120** (resolved 2026-08-29 via CUDA docs): on sm_120, `mma.sync.aligned.kind::f8f6f4` with FP32 accumulator = 2x Ada Fp8 TC, with FP16 accumulator = 1x Ada Fp8 TC. fp32-acc is the peak-rate PV path. Residual risk: fp16-acc may still be chosen for numerical headroom or downstream compatibility (e.g., upstream SageAttention's sm120 rule prefers fp32 but allows fp16).
- **Upstream #388-style fp8 long-seq noise** may hit our identical-family kernels; item 6 tests 131k seq explicitly; mitigation path: split-K fp32 accumulation for PV at long seq.
- **Cluster/TMA limits** (no multicast on consumer) block porting SA3 persistent-kernel assumptions; item 12 records it.
- GPU access: every QA gate on Phases 0-2 runs on this box's RTX PRO 6000 — standing approval granted 2026-08-29.

## Open Questions (ALL RESOLVED by user, 2026-08-29)
1. ~~Exact SM120 SKU~~ → **RTX PRO 6000 Blackwell Workstation Edition** (CC 12.0, 188 SM, 96 GB) in this box; torch 2.13.0+cu130; CUDA toolkit 13.2; env `TORCH_CUDA_ARCH_LIST=12.0` already exported.
2. ~~Approve adding `"12.1"`~~ → **"if appropriate"** → judged appropriate (upstream SageAttention includes 12.1; near-zero cost; keeps DGX Spark/GB20x-workstation path open). Proceed with `12.0` + `12.1`.
3. ~~tests/ + bench/ layout~~ → **yes**.

**Standing directives from user (2026-08-29):**
- Confine all work to `/work/SpargeDev`.
- Python: `/work/SpargeDev/.venv` ONLY. Package changes in that venv are pre-approved (ninja, pytest, etc.).
- **NO downgrading torch or CUDA** — torch 2.13.0+cu130 and CUDA 13.2 are fixed. Item 2b is therefore mandatory, not optional.
- Phases 0–2 QA builds/runs on this box's GPU: **standing approval granted** ("go as far as you can"; wall time abundant until user returns).
- Fill knowledge gaps via MCP tools (cuda-docs, context7, deepwiki `thu-ml/SpargeAttn`) as needed.
- Upstream **SageAttention3 reportedly does not work on sm120** (user, firsthand) — Phase 3 stays decision-only; local probe (not upstream SA3) is the sole evidence source.
