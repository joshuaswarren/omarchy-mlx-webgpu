# GPU-matrix test inventory — 2026-11-27

What the parity surface looks like today, which GPU cells have actually run it,
and which cells no committed test covers. Re-verified against main after the
gpu-matrix suite landed (#14/#15); the 2026-11-26 snapshot claimed zero
Playwright and a dangling `test:webgpu` runner, both since fixed.

## Automated tests committed

| Path | Runner | Coverage |
|---|---|---|
| `tests/cpu_reference.test.ts` | `bun test` (`npm run test:cpu`) | 12 `it()` CPU reference math tests — no GPU touched |
| `tests/tuning_lab.test.ts` | `bun test` (`npm run test:cpu`) | 10 `it()` — WGSL variant/grid math, tuning-config + URL round-trip, parity-gated scoring, hill climber |
| `tests/gpu-matrix/` (`adapter_class.spec.ts`, `gpu_test.ts`, `matrix.ts`) | Playwright (`npm run test:gpu`; `test:webgpu` = `--project=swiftshader`) | per-adapter-class projects; spec asserts the answering adapter matches the project class, all 12 clean parity legs pass, and every fault leg is detected. Optional `GPU_MATRIX_GENERATE=1` adds a 24-token generation smoke per project |
| `playwright.config.ts` | — | projects built from `GPU_MATRIX`: `swiftshader`, `apple-metal-3`, `intel-anv`, `nvidia-vulkan` (a commented `amd-radv` row shows the add-a-class shape) |

`test:webgpu` is live: it runs the `swiftshader` project of the committed
gpu-matrix suite. Real-GPU classes run on demand via `GPU_BASE_URL` /
`GPU_CDP_<PROJECT>` (remote CDP) or `VK_ICD_FILENAMES` (local launch);
`workers: 1`, no CI workflow.

## Parity checks that exist (in-page, browser-run)

`src/main.ts`'s `runFullParitySuite` runs 12 parity checks at boot —
`sdpa_online`, `sdpa_split` (p1+p2), `qmm_vec`, `fast_norm`, `fast_rope_norm`,
`qmm_vec_swiglu`, `fast_rope`, `swiglu`, `embed_q4`, `residual_add`,
`kv_cache_write`, `argmax` — then repeats the suite with every fault family
injected and requires each check to *fail* (`allFaultsDetected`). The verdict
is published on `window.__PARITY_REPORT__` and the terminal title prefix
(`PARITY_PASS`/`PARITY_FAIL`), which is exactly what the gpu-matrix spec
asserts.

`verifyAllKernelsParity` (the ad-hoc suite in `parity_kernels.ts`) adds the
LM-head-scale `qmm_vec` leg (`N=151936`, ~78 MB) on top.

## GPU-matrix cells run so far

From the merge history / lane journal (not reproducible from a committed test):

| Cell | Result |
|---|---|
| Apple Silicon Metal-3 WebGPU (Chrome, real GPU, one host) | 10/10 parity + live perf runs (61.6 → 77.0 → 82.4 tok/s across the optimization toggles) |
| Chrome + Dawn/SwiftShader (CPU fallback adapter, headless) | 10/10 parity, extra QMM shapes (o_proj/down_proj/N=1/odd-N g32·g128/swiglu-odd), fault checks |

The committed matrix names four classes (`swiftshader`, `apple-metal-3`,
`intel-anv`, `nvidia-vulkan`); which real-GPU classes have produced a run is
a receipts question, not visible from the repo.

## Uncovered matrix cells

- **Real non-Apple GPU adapters**: `intel-anv` and `nvidia-vulkan` projects
  exist and run on demand, but no run of either is recorded here — treat as
  unverified until a receipt lands. The WGSL is written to be portable
  (f32 only, ≤16 KB workgroup memory, no subgroup ops).
- **Other Apple GPUs**: the `apple-metal-3` project pins one adapter class;
  other chip families (different core counts / Metal feature sets)
  unverified.
- **Other browsers**: Safari WebGPU and Firefox WebGPU — the matrix is
  Chromium-only; never run.
- **`forceFallbackAdapter` on a real-GPU machine**: `device.ts` has the
  fallback path, but every fallback run so far was on a machine where the
  fallback was the *only* adapter; the real-GPU-then-fallback branch is
  untested.
- **Mobile / Android Chrome**: no matrix class; never run.

## Gaps in the kernel surface itself

- `sdpa_decode_split` parity covers the p1 partial pipeline label; p2's
  combine step is exercised inside the same check — a p2-only failure mode
  would surface as a p1+p2 failure, not a named cell.
- The gpu-matrix projects run the *boot* parity suite only; the heavier
  ad-hoc suite (`verifyAllKernelsParity`, incl. the LM-head leg) has no
  committed per-adapter-class run.
