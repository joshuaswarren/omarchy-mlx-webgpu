import {
  DEFAULT_OPT_FLAGS,
  type FaultConfig,
  type OptFlags,
  type ParityResult,
} from "./kernels/types";
import { QWEN3_0_6B_4BIT } from "./model/config";
import { Qwen3WebGpuEngine } from "./model/qwen3";
import { Qwen3Tokenizer } from "./model/tokenizer";
import { loadQwen3Weights } from "./model/weights";
import { mountApp } from "./ui/app";
import { initWebGpu, type WebGpuContext } from "./webgpu/device";
import {
  verifyFastNormParity,
  verifySdpaOnlineParity,
  verifySdpaSplitParity,
} from "./webgpu/parity";
import {
  verifyArgmaxParity,
  verifyEmbedQ4Parity,
  verifyFastRopeNormParity,
  verifyFastRopeParity,
  verifyQmmVecParity,
  verifyQmmVecSwigluParity,
  verifySwigluParity,
} from "./webgpu/parity_kernels";

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
    __SMOKE_GENERATE__?: (
      prompt: string,
      maxTokens?: number,
      opts?: Partial<OptFlags>,
      fault?: boolean,
    ) => Promise<{
      tokens: number[];
      text: string;
      avgTokPerSec: number;
      totalDispatches: number;
      attentionDispatches: number;
    }>;
  }
}

async function runFullParitySuite(
  device: GPUDevice,
  faultEnabled: boolean,
): Promise<ParityResult[]> {
  return [
    await verifyFastNormParity(device, 16, 1024, 1e-6, false),
    await verifyFastRopeParity(device, 16, 128, 42, 1_000_000, false),
    await verifyFastRopeNormParity(device, 16, 128, 42, 1e-6, 1_000_000, false),
    await verifyQmmVecParity(device, 1024, 1024, 64, false),
    await verifyQmmVecSwigluParity(device, 512, 1024, 64, false),
    await verifySwigluParity(device, 3072, false),
    await verifySdpaOnlineParity(device, 16, 8, 192, faultEnabled),
    await verifySdpaSplitParity(device, 16, 8, 192, faultEnabled),
    await verifyEmbedQ4Parity(device, 4096, 1024, 1337, 64),
    await verifyArgmaxParity(device, 151936),
  ];
}

async function boot() {
  const root = document.getElementById("app");
  if (!root) return;

  let gpu: WebGpuContext | null = null;
  let engine: Qwen3WebGpuEngine | null = null;
  let loadingPromise: Promise<Qwen3WebGpuEngine> | null = null;
  let optFlags: OptFlags = { ...DEFAULT_OPT_FLAGS };
  let faultConfig: FaultConfig = { enabled: false };
  let stopRequested = false;

  const ensureEngine = async (): Promise<Qwen3WebGpuEngine> => {
    if (engine) return engine;
    if (loadingPromise) return loadingPromise;
    if (!gpu) throw new Error("WebGPU device is not initialized");
    const device = gpu.device;

    loadingPromise = (async () => {
      ui.setLoadProgress({
        stage: "tokenizer",
        loadedBytes: 0,
        totalBytes: QWEN3_0_6B_4BIT.weightsBytes,
        message: "Loading Qwen3-0.6B-4bit tokenizer…",
      });
      const tokenizer = await Qwen3Tokenizer.load(QWEN3_0_6B_4BIT);
      const weights = await loadQwen3Weights(device, QWEN3_0_6B_4BIT, (p) => {
        ui.setLoadProgress(p);
      });
      ui.setStatus("Compiling omarchy-mlx WGSL pipelines…", "busy");
      const created = await Qwen3WebGpuEngine.create(device, weights, tokenizer);
      engine = created;
      return created;
    })();

    try {
      return await loadingPromise;
    } catch (err) {
      loadingPromise = null;
      throw err;
    }
  };

  const ui = mountApp(root, {
    onOptChange(next) {
      optFlags = { ...next };
    },
    onFaultChange(next) {
      faultConfig = { ...next };
      if (gpu) {
        void runFullParitySuite(gpu.device, next.enabled).then((results) => {
          ui.setParityResults(results);
        });
      }
    },
    onStop() {
      stopRequested = true;
    },
    onReset() {
      stopRequested = true;
      ui.clearTokens();
      ui.setStatus(engine ? "Model ready" : "WebGPU parity PASS — click Generate to load model", "ok");
    },
    async onRunParity(faultEnabled) {
      if (!gpu) return [];
      ui.setStatus("Running 10-kernel WebGPU vs CPU parity…", "busy");
      const results = await runFullParitySuite(gpu.device, faultEnabled);
      ui.setParityResults(results);
      const allPass = results.every((r) => r.passed);
      ui.setStatus(
        allPass
          ? `All ${results.length} kernels PASS CPU parity`
          : "Fault detected — GPU diverged from CPU reference",
        allPass ? "ok" : "error",
      );
      return results;
    },
    async onGenerate(prompt, opts, fault) {
      optFlags = { ...opts };
      faultConfig = { ...fault };
      stopRequested = false;
      ui.setError(null);
      ui.clearTokens();
      ui.setGenerating(true);

      try {
        const eng = await ensureEngine();
        await eng.generate(
          prompt,
          96,
          () => optFlags,
          () => faultConfig,
          (trace) => ui.pushTokenTrace(trace),
          () => stopRequested,
        );
      } catch (err) {
        ui.setError(err instanceof Error ? err.message : String(err));
      } finally {
        ui.setGenerating(false);
      }
    },
  });

  window.__SMOKE_GENERATE__ = async (prompt, maxTokens = 24, optsOverride, faultOn = false) => {
    optFlags = { ...DEFAULT_OPT_FLAGS, ...optsOverride };
    faultConfig = { enabled: faultOn };
    ui.setOptFlags(optFlags);
    ui.setFaultConfig(faultConfig);
    stopRequested = false;
    ui.clearTokens();
    ui.setGenerating(true);
    let lastDispatches = 0;
    let lastAttn = 0;
    try {
      const eng = await ensureEngine();
      const out = await eng.generate(
        prompt,
        maxTokens,
        () => optFlags,
        () => faultConfig,
        (trace) => {
          lastDispatches = trace.totalDispatches;
          lastAttn = trace.attentionDispatches;
          ui.pushTokenTrace(trace);
        },
        () => stopRequested,
      );
      return {
        ...out,
        totalDispatches: lastDispatches,
        attentionDispatches: lastAttn,
      };
    } finally {
      ui.setGenerating(false);
    }
  };

  try {
    ui.setStatus("Initializing WebGPU…", "busy");
    gpu = await initWebGpu();
    ui.setAdapterInfo(gpu.info);

    const cleanResults = await runFullParitySuite(gpu.device, false);
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
      allCleanPassed
        ? `${cleanResults.length}/${cleanResults.length} kernels PASS CPU parity — click Generate`
        : "WebGPU parity check failed",
      allCleanPassed ? "ok" : "error",
    );
    document.title =
      allCleanPassed && allFaultsDetected
        ? "PARITY_PASS — omarchy-mlx WebGPU"
        : "PARITY_FAIL — omarchy-mlx WebGPU";
  } catch (err) {
    ui.setError(err instanceof Error ? err.message : String(err));
    document.title = "PARITY_ERROR — omarchy-mlx WebGPU";
  }
}

boot();
