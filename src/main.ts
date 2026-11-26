import type { FaultConfig, OptFlags, ParityResult } from "./kernels/types";
import { QWEN3_0_6B_4BIT } from "./model/config";
import { mountApp } from "./ui/app";
import { initWebGpu, type WebGpuContext } from "./webgpu/device";
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

async function runParitySuite(device: GPUDevice, faultEnabled: boolean): Promise<ParityResult[]> {
  return [
    await verifyFastNormParity(device, 16, 1024, 1e-6, false),
    await verifySdpaOnlineParity(device, 16, 8, 192, faultEnabled),
    await verifySdpaSplitParity(device, 16, 8, 192, faultEnabled),
  ];
}

async function boot() {
  const root = document.getElementById("app");
  if (!root) return;

  let gpu: WebGpuContext | null = null;
  let currentOpts: OptFlags | null = null;
  let currentFault: FaultConfig = { enabled: false };

  const ui = mountApp(root, {
    onOptChange(opts) {
      currentOpts = opts;
      void currentOpts;
    },
    onFaultChange(fault) {
      currentFault = fault;
      if (gpu) {
        void runParitySuite(gpu.device, fault.enabled).then((results) => {
          ui.setParityResults(results);
        });
      }
    },
    async onRunParity(faultEnabled) {
      if (!gpu) return [];
      ui.setStatus("Running WebGPU vs CPU parity…", "busy");
      const results = await runParitySuite(gpu.device, faultEnabled);
      ui.setParityResults(results);
      ui.setStatus(
        results.every((r) => r.passed) ? "All kernels PASS CPU parity" : "Fault detected vs CPU reference",
        results.every((r) => r.passed) ? "ok" : "error",
      );
      return results;
    },
    async onGenerate() {
      ui.setStatus("Connecting Qwen3-0.6B-4bit runtime kernels…", "busy");
    },
  });

  try {
    ui.setStatus("Initializing WebGPU…", "busy");
    gpu = await initWebGpu();
    ui.setAdapterInfo(gpu.info);

    const cleanResults = await runParitySuite(gpu.device, false);
    const faultResults = [await verifySdpaOnlineParity(gpu.device, 16, 8, 192, true)];
    ui.setParityResults(cleanResults);

    const allCleanPassed = cleanResults.every((r) => r.passed);
    const allFaultsDetected = faultResults.every((r) => !r.passed);
    window.__PARITY_REPORT__ = {
      adapter: gpu.info,
      model: QWEN3_0_6B_4BIT.repoId,
      results: cleanResults,
      faultResults,
      allCleanPassed,
      allFaultsDetected,
    };

    ui.setStatus(
      allCleanPassed ? "WebGPU parity PASS — ready" : "WebGPU parity check failed",
      allCleanPassed ? "ok" : "error",
    );
    document.title = allCleanPassed && allFaultsDetected ? "PARITY_PASS — omarchy-mlx WebGPU" : "PARITY_FAIL — omarchy-mlx WebGPU";
    void currentFault;
  } catch (err) {
    ui.setError(err instanceof Error ? err.message : String(err));
    document.title = "PARITY_ERROR — omarchy-mlx WebGPU";
  }
}

boot();
