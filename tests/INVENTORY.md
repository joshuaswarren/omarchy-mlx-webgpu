# GPU-matrix test inventory — 2026-11-26

What the parity surface looks like today, which GPU cells have actually run it,
and which cells no committed test covers.

## Automated tests committed

| Path | Runner | Coverage |
|---|---|---|
| `tests/cpu_reference.test.ts` | `bun test` (`npm run test:cpu`) | 17 tests, CPU reference math only — no GPU touched |
| `tests/run_webgpu_parity.mjs` | referenced by `npm run test:webgpu` | **missing** — the script target does not exist on any branch |
| Playwright (config/specs/dep) | none | **zero** — no `playwright.config.*`, no `*.spec.ts`, no dependency |

## Parity checks that exist (in-page, browser-run)

`src/webgpu/parity.ts` + `src/webgpu/parity_kernels.ts` expose 13 parity checks
behind the UI's run-parity path (`onRunParity`). Each compares one WGSL kernel's
output against `src/reference/cpu.ts` with a per-kernel tolerance:

`fast_norm`, `sdpa_decode_online`, `sdpa_decode_split` (p1+p2), `fast_rope`,
`fast_rope_norm`, `qmm_vec`, `qmm_vec` at LM-head scale (`N=151936`),
`qmm_vec_swiglu`, `swiglu`, `embed_q4`, `residual_add`, `kv_cache_write`,
`argmax`.

These are runtime checks, not test files: they execute only when a real
browser session runs them (the page UI, or a lane's ad-hoc headless-Chrome
invocation — none of which is committed).

## GPU-matrix cells run so far

From the merge history / lane journal (not reproducible from a committed test):

| Cell | Result |
|---|---|
| Apple Silicon Metal-3 WebGPU (Chrome, real GPU, one host) | 10/10 parity + live perf runs (61.6 → 77.0 → 82.4 tok/s across the optimization toggles) |
| Chrome + Dawn/SwiftShader (CPU fallback adapter, headless) | 10/10 parity, extra QMM shapes (o_proj/down_proj/N=1/odd-N g32·g128/swiglu-odd), fault checks |

## Uncovered matrix cells

- **Any real non-Apple GPU adapter**: NVIDIA / AMD / Intel dGPU or iGPU under Dawn
  (Chrome on Linux/Windows) — never run; the WGSL is written to be portable
  (f32 only, ≤16 KB workgroup memory, no subgroup ops) but no adapter other
  than the one Apple host has verified it.
- **Other Apple GPUs**: only one host's Metal-3 adapter has run; other chip
  families (different core counts / Metal feature sets) unverified.
- **Other browsers**: Safari WebGPU and Firefox WebGPU — never run.
- **`forceFallbackAdapter` on a real-GPU machine**: `device.ts` has the
  fallback path, but every fallback run so far was on a machine where the
  fallback was the *only* adapter; the real-GPU-then-fallback branch is
  untested.
- **Mobile / Android Chrome**: never run.

## Gaps in the kernel surface itself

- `sdpa_decode_split` parity covers the p1 partial pipeline label; p2's
  combine step is exercised inside the same check — a p2-only failure mode
  would surface as a p1+p2 failure, not a named cell.
- The fault-injection path (`faultDiff` / fault switch) is exercised by the
  UI only; no check asserts that a faulted run is *detected* as a parity
  failure rather than silently passing.
- `test:webgpu` is a dangling script — either land the runner or drop the
  script so `npm run` doesn't advertise a harness that 404s.
