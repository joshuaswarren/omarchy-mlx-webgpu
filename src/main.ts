import {
  DEFAULT_OPT_FLAGS,
  type AbRaceResult,
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
      fault?: boolean | FaultConfig,
    ) => Promise<{
      tokens: number[];
      text: string;
      avgTokPerSec: number;
      totalDispatches: number;
      attentionDispatches: number;
    }>;
    __RUN_AB_RACE__?: (prompt?: string, tokenBudget?: number) => Promise<AbRaceResult | null>;
  }
}

async function runFullParitySuite(
  device: GPUDevice,
  fault: FaultConfig | boolean,
): Promise<ParityResult[]> {
  const fc: FaultConfig =
    typeof fault === "boolean"
      ? { enabled: fault, target: "attention" }
      : { enabled: fault.enabled, target: fault.target ?? "attention" };
  const target = fc.target ?? "attention";
  const attnF = fc.enabled && target === "attention";
  const qmmF = fc.enabled && target === "qmm";
  const ropeF = fc.enabled && target === "rope";
  const normF = fc.enabled && target === "norm";
  return [
    await verifySdpaOnlineParity(device, 16, 8, 192, attnF),
    await verifySdpaSplitParity(device, 16, 8, 192, attnF),
    await verifyQmmVecParity(device, 1024, 1024, 64, qmmF),
    await verifyFastNormParity(device, 16, 1024, 1e-6, normF),
    await verifyFastRopeNormParity(device, 16, 128, 42, 1e-6, 1_000_000, ropeF || normF),
    await verifyQmmVecSwigluParity(device, 512, 1024, 64, qmmF),
    await verifyFastRopeParity(device, 16, 128, 42, 1_000_000, ropeF),
    await verifySwigluParity(device, 3072, false),
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
  let faultConfig: FaultConfig = { enabled: false, target: "attention" };
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

  const runAbRace = async (prompt: string, tokenBudget = 24): Promise<AbRaceResult | null> => {
    stopRequested = false;
    faultConfig = { enabled: false, target: faultConfig.target ?? "attention" };
    ui.setFaultConfig(faultConfig);
    ui.setError(null);
    ui.clearTokens();
    ui.setGenerating(true);

    const baseOpts: OptFlags = {
      oneDispatchAttention: false,
      fusedRopeNorm: false,
      fusedSwigluQmm: false,
    };
    const optAll: OptFlags = {
      oneDispatchAttention: true,
      fusedRopeNorm: true,
      fusedSwigluQmm: true,
    };

    try {
      const eng = await ensureEngine();

      // Short silent GPU warmup (3 tokens) so shader JIT and GPU clocks are identical for both legs
      await eng.generate(prompt, 3, () => baseOpts, () => ({ enabled: false }), () => {}, () => stopRequested);
      if (stopRequested) return null;

      // Leg A: Unfused baseline (537 dispatches/tok)
      optFlags = { ...baseOpts };
      ui.setOptFlags(baseOpts);
      ui.setAbRace({ prompt, phase: "baseline", tokenBudget });
      ui.setStatus("A/B Race 1/2: Unfused baseline (537 dispatches/tok)…", "busy");
      let baseDispatches = 537;
      const legA = await eng.generate(
        prompt,
        tokenBudget,
        () => baseOpts,
        () => ({ enabled: false }),
        (trace) => {
          baseDispatches = trace.totalDispatches;
          ui.pushTokenTrace(trace);
        },
        () => stopRequested,
      );
      if (stopRequested || legA.tokens.length === 0) {
        ui.setAbRace(null);
        return null;
      }
      const baseline = {
        tokens: legA.tokens,
        text: legA.text,
        avgTokPerSec: legA.avgTokPerSec,
        msPerTok: legA.avgTokPerSec > 0 ? 1000 / legA.avgTokPerSec : 0,
        dispatchesPerTok: baseDispatches,
      };

      // Leg B: Optimized (PR #997 + fusions, 397 dispatches/tok)
      optFlags = { ...optAll };
      ui.clearTokens();
      ui.setOptFlags(optAll);
      ui.setAbRace({ prompt, phase: "optimized", tokenBudget, baseline });
      ui.setStatus("A/B Race 2/2: PR #997 + fused kernels (397 dispatches/tok)…", "busy");
      let optDispatches = 397;
      const legB = await eng.generate(
        prompt,
        tokenBudget,
        () => optAll,
        () => ({ enabled: false }),
        (trace) => {
          optDispatches = trace.totalDispatches;
          ui.pushTokenTrace(trace);
        },
        () => stopRequested,
      );
      if (stopRequested || legB.tokens.length === 0) {
        ui.setAbRace(null);
        return null;
      }
      const optimized = {
        tokens: legB.tokens,
        text: legB.text,
        avgTokPerSec: legB.avgTokPerSec,
        msPerTok: legB.avgTokPerSec > 0 ? 1000 / legB.avgTokPerSec : 0,
        dispatchesPerTok: optDispatches,
      };

      const comparedCount = Math.min(baseline.tokens.length, optimized.tokens.length);
      let exactMatchCount = 0;
      for (let i = 0; i < comparedCount; i++) {
        if (baseline.tokens[i] === optimized.tokens[i]) exactMatchCount++;
      }
      const speedupPct =
        baseline.avgTokPerSec > 0
          ? ((optimized.avgTokPerSec - baseline.avgTokPerSec) / baseline.avgTokPerSec) * 100
          : 0;

      const result: AbRaceResult = {
        prompt,
        phase: "done",
        tokenBudget,
        baseline,
        optimized,
        speedupPct,
        exactMatchCount,
        comparedCount,
      };
      ui.setAbRace(result);
      const sign = speedupPct >= 0 ? "+" : "";
      ui.setStatus(
        `A/B Race: ${baseline.avgTokPerSec.toFixed(1)} -> ${optimized.avgTokPerSec.toFixed(1)} tok/s (${sign}${speedupPct.toFixed(0)}%, ${exactMatchCount}/${comparedCount} exact)`,
        "ok",
      );
      return result;
    } catch (err) {
      ui.setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally {
      ui.setGenerating(false);
    }
  };

  const ui = mountApp(root, {
    onOptChange(next) {
      optFlags = { ...next };
    },
    onFaultChange(next) {
      faultConfig = { enabled: next.enabled, target: next.target ?? faultConfig.target ?? "attention" };
      if (gpu) {
        void runFullParitySuite(gpu.device, faultConfig).then((results) => {
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
      ui.setAbRace(null);
      ui.setStatus(engine ? "Model ready" : "WebGPU parity PASS — click Generate or Race A/B", "ok");
    },
    async onRunParity(faultOrEnabled) {
      if (!gpu) return [];
      const fc: FaultConfig =
        typeof faultOrEnabled === "boolean"
          ? { enabled: faultOrEnabled, target: faultConfig.target ?? "attention" }
          : faultOrEnabled;
      ui.setStatus("Running 10-kernel WebGPU vs CPU parity…", "busy");
      const results = await runFullParitySuite(gpu.device, fc);
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
    async onRunAbRace(prompt) {
      await runAbRace(prompt, 24);
    },
    async onGenerate(prompt, opts, fault) {
      optFlags = { ...opts };
      faultConfig = { enabled: fault.enabled, target: fault.target ?? "attention" };
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

  window.__RUN_AB_RACE__ = (prompt, tokenBudget = 24) =>
    runAbRace(
      prompt ?? "In two short sentences, explain what a GPU compute shader does.",
      tokenBudget,
    );

  window.__SMOKE_GENERATE__ = async (prompt, maxTokens = 24, optsOverride, faultOn = false) => {
    optFlags = { ...DEFAULT_OPT_FLAGS, ...optsOverride };
    faultConfig =
      typeof faultOn === "boolean"
        ? { enabled: faultOn, target: "attention" }
        : { enabled: faultOn.enabled, target: faultOn.target ?? "attention" };
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
