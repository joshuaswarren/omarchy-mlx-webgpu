# Fault-injection + toggle coverage table — 2026-11-26 (SUP-520)

Consolidates the measured SUP-488 fault-injection run and SUP-497 #997-toggle
run on this server (Chrome headless, Dawn **SwiftShader** CPU adapter —
`hasShaderF16=false`, `hasSubgroups=true`, Playwright chromium).
Update of the SUP-291 inventory (`w7Z/sup291-test-inventory`, unmerged): the
Playwright matrix (`tests/gpu-matrix/`, `playwright.config.ts`) now exists on
main, and boot parity runs 10 of the 13 in-page checks.

## SUP-488 — fault-injection per class (SwiftShader leg)

Boot report `__PARITY_REPORT__`: `allCleanPassed=true`, `allFaultsDetected=true`.
UI switch sweep (`sup488-final.mjs`) per `faultTarget`:

| Fault class | Injected family | Caught | Notes |
|---|---|---|---|
| `attention` | SDPA | 2/2 | both `sdpa_decode_online` and `sdpa_decode_split` baseline fault — injection hits the whole SDPA family |
| `qmm` | Q4 GEMV | 2/2 | `qmm_vec` + `qmm_vec_swiglu` |
| `rope` | RoPE | 2/2 | `fast_rope_norm` + `fast_rope` |
| `norm` | norm | 2/2 | `fast_norm` + `fast_rope_norm` |
| clean | — | 10/10 PASS | all parity rows clean leg |

Detection is parity-diff only (maxAbsDiff / cosine vs CPU reference), which is
the contract SUP-294 named adoptable.

## SUP-497 — #997 toggle, per reachable GPU (16 greedy tokens)

| Adapter | `attn=1` (1-dispatch online SDPA) | `attn=0` (2-dispatch split-K) | token ids |
|---|---|---|---|
| SwiftShader (google, CPU) | 0.27 tok/s · 397 disp/tok · parity 16/16 | 0.38 tok/s · 425 disp/tok · parity 16/16 | identical both legs |

Dispatch delta −28/tok confirmed; **on SwiftShader the 1-dispatch path is
slower** — a CPU rasterizer is compute-bound, so the dispatch saving does not
pay there. The measured win exists only on the one Apple Metal-3 host so far.

## Coverage gaps (for w8H)

1. **Fault classes cover 8 of 13 in-page checks.** `faultTarget` has
   {attention,qmm,rope,norm}; no injection exists for `embed_q4`, `swiglu`
   standalone, `argmax`, `residual_add`, `kv_cache_write`, or the LM-head-scale
   `qmm_vec` (N=151936) check.
2. **Boot parity runs 10 of 13 checks** — `residual_add`, `kv_cache_write`,
   and the LM-head `qmm_vec` never run at boot; my leg did not exercise them
   either.
3. **One fault shape per class** — a single perturbation each; no magnitude
   ladder, and detection is parity-diff only (no error-class assertion).
4. **Adapter breadth is one cell**: every measured leg on this server is
   SwiftShader. Non-Apple real-GPU adapters (NVIDIA/AMD/Intel under Dawn),
   other Apple chips, Safari/Firefox WebGPU, and `forceFallbackAdapter` on a
   real-GPU machine all remain unrun (inventory list still stands).
5. **Toggle perf unproven off Apple**: parity identical on SwiftShader, but
   the −28 disp/tok translates to a win only where dispatch cost matters —
   no non-Apple adapter leg exists.
6. `npm run test:webgpu` still points at a missing script (dangling since
   SUP-291).

Run provenance: `vite preview` of `main` build (dist asset index-C2slLG5y.js),
headless chromium `--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader`,
runner `tests/gpu-matrix/sup497-toggle.mjs` + `sup488-final.mjs`/`sup488-report.mjs`.
