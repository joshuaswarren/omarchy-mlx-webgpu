# Agent Tuning Guide (`omarchy-mlx-webgpu`)

This page runs `mlx-community/Qwen3-0.6B-4bit` directly on the visitor's WebGPU device and exposes a structured tuning interface so browser agents (Playwright, Chrome DevTools Protocol, or custom scripts) can search GPU execution levers and hill-climb decode throughput autonomously.

Live page: `https://joshuaswarren.github.io/omarchy-mlx-webgpu/?tab=tuning`

## Scoring Rule

Every candidate configuration is scored against the canonical IEEE `f32` greedy reference token sequence for the active prompt:

- `parityPass`: `true` if and only if every generated token ID matches the reference sequence (`parityMatched === parityTotal`).
- `score`:
  - `tokPerSec` (rounded to 2 decimal places) when `parityPass === true`
  - `0` whenever even a single output token diverges (`parityPass === false`)

Speed gains that change output tokens always score `0` and are rejected by the hill climber.

## URL Parameters

Pass any subset of levers in the query string to load a specific configuration directly or share a winning configuration:

| Param | Config key | Allowed values | Meaning |
|---|---|---|---|
| `tab` | — | `tuning` (or `4`) | Open directly to the Tuning Lab tab |
| `attn` | `oneDispatchAttention` | `1`, `0` | PR #997 1-dispatch online-softmax SDPA vs 2-dispatch split-K (`-28` disp/tok) |
| `rope` | `fusedRopeNorm` | `1`, `0` | Fused per-head QK RMSNorm + RoPE (`-56` disp/tok) |
| `swiglu` | `fusedSwigluQmm` | `1`, `0` | Fused Gate+Up Q4 GEMV + SwiGLU (`-56` disp/tok) |
| `wg` | `qmmWorkgroupSize` | `64`, `128`, `256` | QMM threads per workgroup (`4`, `8`, or `16` output rows/WG) |
| `prec` | `qmmPrecision` | `f32`, `f16`, `f16_accum` | IEEE f32, `shader-f16` inner dot + f32 accum, or full f16 K-loop accum |
| `sg` | `useSubgroups` | `1`, `0` | Hardware `subgroupAdd` reduction (requires adapter `subgroups` feature) |
| `batch` | `dispatchBatching` | `single_pass`, `per_layer_pass`, `chunked_submit` | 1 compute pass vs 30 per-layer passes vs 4 queue submits per token |
| `pace` | `decodePacing` | `every_token`, `every_4`, `unthrottled` | UI timer-yield and trace-readback cadence |
| `chain` | `gpuTokenChain` | `1`, `4` | Chain 1 or 4 decode tokens in one GPU submit via indirect `argmax -> embed_q4_indirect` (cuts `mapAsync` syncs 4x) |
| `fault` | `faultEnabled` | `0`, `1` | Inject a TDR-safe kernel fault (triggers parity fail -> score `0`) |
| `faultTarget` | `faultTarget` | `attention`, `qmm`, `rope`, `norm` | Target kernel family when `fault=1` |
| `autorun` | — | `1`, `0` | Automatically benchmark the URL config once boot parity passes |
| `prompt` | — | URL-encoded string | Override the default benchmark prompt |

Example share URL:

```text
https://joshuaswarren.github.io/omarchy-mlx-webgpu/?tab=tuning&attn=1&rope=1&swiglu=1&wg=256&prec=f16&sg=1&batch=single_pass&pace=unthrottled
```

When a non-default benchmark prompt is active, generated share URLs automatically preserve `&prompt=<encoded>` alongside the lever parameters.

Unsupported adapter capabilities (for example `prec=f16` on an adapter without `shader-f16`, or `sg=1` without `subgroups`) are clamped safely to supported values during normalization.

## Page JSON API (`window.omarchyLab`)

During boot, `document.title` resolves to one of four terminal prefixes:

- `PARITY_PASS` — WebGPU initialized and all 10 boot kernels passed CPU parity (ready to run `window.omarchyLab`)
- `PARITY_FAIL` — WebGPU initialized, but a boot parity check failed
- `PARITY_ERROR` — WebGPU initialized, but device setup or shader compilation threw an error
- `WebGPU unavailable` — browser lacks WebGPU (`navigator.gpu` missing or no adapter); stop immediately instead of waiting for `PARITY_PASS`

Once `document.title` starts with `PARITY_PASS`, `window.omarchyLab` is ready:

| Method | Signature | Returns |
|---|---|---|
| `getState()` | `() => LabState` | `{ ready, adapter, model, config, levers, last, best, history, shareUrl }` |
| `getLevers()` | `() => LeverSpec[]` | Every lever, all values, `supportedValues` on this GPU, and its `urlParam` |
| `set(patch)` | `(patch: Partial<TuningConfig>) => TuningConfig` | Validates, normalizes, syncs UI & URL, returns active `TuningConfig` |
| `run(opts?)` | `(opts?: { config?, prompt?, tokens?, label?, trials? }) => Promise<TuningEvalResult>` | Evaluates one config against reference tokens (adaptive median across `trials`, default `1`) and returns scored JSON |
| `hillClimb(opts?)` | `(opts?: { start?: "baseline" \| "current", prompt?, tokens?, maxPasses?: number, trials?: number, minGainPct?: number }) => Promise<HillClimbReport>` | Runs multi-pass coordinate ascent (default `maxPasses: 2`, adaptive median-of-3 `trials`, `minGainPct: 1.0`) and returns full path JSON |
| `read()` | `() => { adapter, config, last, best, history, shareUrl }` | Latest evaluation, best winning config, full history, and share link as plain JSON |
| `shareUrl(cfg?, prompt?)` | `(cfg?: Partial<TuningConfig>, prompt?: string) => string` | Shareable URL for the best (or specified) config, preserving custom `prompt=` when set |
| `stop()` | `() => void` | Cancels an in-flight benchmark or hill climb |

After every state update, the page also writes the latest JSON snapshot into `<script type="application/json" id="omarchy-lab-state">` in `<head>`, so DOM-only tools can read it without evaluating custom scripts.

### `TuningEvalResult` Shape

```json
{
  "step": 4,
  "label": "wg=256",
  "config": {
    "oneDispatchAttention": true,
    "fusedRopeNorm": true,
    "fusedSwigluQmm": true,
    "qmmWorkgroupSize": 256,
    "qmmPrecision": "f16",
    "useSubgroups": true,
    "dispatchBatching": "single_pass",
    "decodePacing": "unthrottled",
    "faultEnabled": false,
    "faultTarget": "attention"
  },
  "tokPerSec": 118.4,
  "msPerTok": 8.45,
  "dispatchesPerTok": 397,
  "parityMatched": 16,
  "parityTotal": 16,
  "parityPass": true,
  "score": 118.4,
  "status": "accepted",
  "shareUrl": "https://joshuaswarren.github.io/omarchy-mlx-webgpu/?tab=tuning&..."
}
```

## Example: Run an Agent Hill Climb with Playwright

```js
import { chromium } from "@playwright/test";

const browser = await chromium.launch({
  args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage();
await page.goto("https://joshuaswarren.github.io/omarchy-mlx-webgpu/?tab=tuning");
await page.waitForFunction(
  () => /^(PARITY_|WebGPU unavailable)/.test(document.title),
  null,
  { timeout: 120_000 },
);
const title = await page.title();
if (!title.startsWith("PARITY_PASS")) {
  throw new Error(`Tuning Lab cannot run on this browser: ${title}`);
}

// Option A: run the built-in coordinate-ascent hill climber
const report = await page.evaluate(() =>
  window.omarchyLab.hillClimb({ start: "baseline", tokens: 16 }),
);
console.log("Best score:", report.best.score, "Share URL:", report.shareUrl);

// Option B: custom agent search loop using set() + run() + read()
const levers = await page.evaluate(() => window.omarchyLab.getLevers());
for (const wg of levers.find((l) => l.key === "qmmWorkgroupSize").supportedValues) {
  await page.evaluate((w) => window.omarchyLab.set({ qmmWorkgroupSize: w }), wg);
  const res = await page.evaluate(() => window.omarchyLab.run({ tokens: 16 }));
  console.log({ wg, tokPerSec: res.tokPerSec, parityPass: res.parityPass, score: res.score });
}

const summary = await page.evaluate(() => window.omarchyLab.read());
console.log(JSON.stringify(summary.best, null, 2));
await browser.close();
```
