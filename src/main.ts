import { QWEN3_0_6B_4BIT } from "./model/config";
import type { ParityResult } from "./kernels/types";
import { initWebGpu } from "./webgpu/device";
import {
  verifyFastNormParity,
  verifySdpaOnlineParity,
  verifySdpaSplitParity,
} from "./webgpu/parity";

declare global {
  interface Window {
    __PARITY_REPORT__?: {
      adapter: unknown;
      model: string;
      results: ParityResult[];
      faultResults: ParityResult[];
      allCleanPassed: boolean;
      allFaultsDetected: boolean;
    };
  }
}

async function boot() {
  const root = document.getElementById("app");
  if (!root) return;
  root.innerHTML = `<main><h1>omarchy-mlx WebGPU</h1><p>Initializing WebGPU adapter…</p></main>`;

  try {
    const ctx = await initWebGpu();
    const cleanResults = [
      await verifyFastNormParity(ctx.device, 16, 1024, 1e-6, false),
      await verifySdpaOnlineParity(ctx.device, 16, 8, 192, false),
      await verifySdpaSplitParity(ctx.device, 16, 8, 192, false),
    ];
    const faultResults = [
      await verifySdpaOnlineParity(ctx.device, 16, 8, 192, true),
    ];

    const allCleanPassed = cleanResults.every((r) => r.passed);
    const allFaultsDetected = faultResults.every((r) => !r.passed);

    window.__PARITY_REPORT__ = {
      adapter: ctx.info,
      model: QWEN3_0_6B_4BIT.repoId,
      results: cleanResults,
      faultResults,
      allCleanPassed,
      allFaultsDetected,
    };

    const rows = [...cleanResults, ...faultResults]
      .map(
        (r) => `<tr>
          <td>${r.label}${r.faultInjected ? " [FAULT ON]" : ""}</td>
          <td><code>${r.shape}</code></td>
          <td><code>${r.maxAbsDiff.toExponential(2)}</code></td>
          <td><code>${r.cosineSimilarity.toFixed(6)}</code></td>
          <td><code>${r.gpuMs.toFixed(2)} ms</code></td>
          <td class="${r.passed ? "pass" : "fail"}">${r.passed ? "PASS" : "FAIL"}</td>
        </tr>`,
      )
      .join("");

    root.innerHTML = `<main>
      <h1>omarchy-mlx → WebGPU</h1>
      <div class="card">
        <strong>Adapter:</strong> <code>${ctx.info.vendor} / ${ctx.info.architecture}</code> |
        <strong>Model:</strong> <code>${QWEN3_0_6B_4BIT.repoId}</code> (28L, HD=128, MLX 4-bit g64)
      </div>
      <div class="card">
        <h2>WebGPU vs CPU Parity</h2>
        <table style="width:100%; border-collapse:collapse; text-align:left;">
          <thead><tr><th>Kernel</th><th>Shape</th><th>Max Abs Diff</th><th>Cosine</th><th>GPU Time</th><th>Status</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </main>`;
    document.title = allCleanPassed && allFaultsDetected ? "PARITY_PASS" : "PARITY_FAIL";
  } catch (err) {
    root.innerHTML = `<main><h1>WebGPU Init Error</h1><pre>${String(err)}</pre></main>`;
    document.title = "PARITY_ERROR";
  }
}

boot();
