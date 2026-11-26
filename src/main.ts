import {
  type AbRaceResult,
  type FaultConfig,
  type GpuAdapterInfo,
  type HillClimbReport,
  type LeverSpec,
  type OptFlags,
  type ParityResult,
  type TuningConfig,
  type TuningEvalResult,
} from "./kernels/types";
import { QWEN3_0_6B_4BIT } from "./model/config";
import { Qwen3WebGpuEngine, type ExecutionTuning } from "./model/qwen3";
import { Qwen3Tokenizer } from "./model/tokenizer";
import { loadQwen3Weights } from "./model/weights";
import {
  BASELINE_TUNING_CONFIG,
  DEFAULT_TUNING_CONFIG,
  REFERENCE_TUNING_CONFIG,
  configKey,
  encodeConfigToUrl,
  getLeverSpecs,
  normalizeTuningConfig,
  parseTuningUrl,
  scoreTuningRun,
} from "./tuning/config";
import { runCoordinateHillClimb } from "./tuning/hill_climb";
import { mountApp } from "./ui/app";
import { initWebGpu, WebGpuUnavailableError, type WebGpuContext } from "./webgpu/device";
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

const DEFAULT_PROMPT = "In two short sentences, explain what a GPU compute shader does.";

export interface OmarchyLabApi {
  version: string;
  getState: () => {
    ready: boolean;
    adapter: GpuAdapterInfo | null;
    model: string;
    config: TuningConfig;
    levers: LeverSpec[];
    last: TuningEvalResult | null;
    best: TuningEvalResult | null;
    history: TuningEvalResult[];
    shareUrl: string;
  };
  getLevers: () => LeverSpec[];
  set: (patch: Partial<TuningConfig>) => TuningConfig;
  run: (options?: {
    config?: Partial<TuningConfig>;
    prompt?: string;
    tokens?: number;
    label?: string;
  }) => Promise<TuningEvalResult>;
  hillClimb: (options?: {
    start?: "baseline" | "current";
    prompt?: string;
    tokens?: number;
    maxPasses?: number;
  }) => Promise<HillClimbReport>;
  read: () => {
    adapter: GpuAdapterInfo | null;
    config: TuningConfig;
    last: TuningEvalResult | null;
    best: TuningEvalResult | null;
    history: TuningEvalResult[];
    shareUrl: string;
  };
  shareUrl: (config?: Partial<TuningConfig>, prompt?: string) => string;
  stop: () => void;
}

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
    omarchyLab?: OmarchyLabApi;
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

function toOptFlags(c: TuningConfig): OptFlags {
  return {
    oneDispatchAttention: c.oneDispatchAttention,
    fusedRopeNorm: c.fusedRopeNorm,
    fusedSwigluQmm: c.fusedSwigluQmm,
  };
}

function toFaultConfig(c: TuningConfig): FaultConfig {
  return {
    enabled: c.faultEnabled,
    target: c.faultTarget,
  };
}

function toExecTuning(c: TuningConfig): ExecutionTuning {
  return {
    qmmWorkgroupSize: c.qmmWorkgroupSize,
    qmmPrecision: c.qmmPrecision,
    useSubgroups: c.useSubgroups,
    dispatchBatching: c.dispatchBatching,
    decodePacing: c.decodePacing,
  };
}

function syncDomLabJson(payload: unknown): void {
  let el = document.getElementById("omarchy-lab-state");
  if (!el) {
    el = document.createElement("script");
    el.id = "omarchy-lab-state";
    el.setAttribute("type", "application/json");
    document.head.appendChild(el);
  }
  el.textContent = JSON.stringify(payload);
}

async function boot() {
  const root = document.getElementById("app");
  if (!root) return;

  const urlParsed = parseTuningUrl(window.location.search);
  const pageOriginPath = window.location.origin + window.location.pathname;

  let gpu: WebGpuContext | null = null;
  let engine: Qwen3WebGpuEngine | null = null;
  let loadingPromise: Promise<Qwen3WebGpuEngine> | null = null;
  let tuningConfig: TuningConfig = normalizeTuningConfig(urlParsed.patch, null, DEFAULT_TUNING_CONFIG);
  let stopRequested = false;

  // Reference token cache keyed by `${prompt}::${tokenBudget}` so parity scoring is fast & exact
  const refTokenCache = new Map<string, number[]>();
  let lastEval: TuningEvalResult | null = null;
  let bestEval: TuningEvalResult | null = null;
  let evalHistory: TuningEvalResult[] = [];
  let manualStepCounter = 0;

  const getCustomPrompt = (override?: string): string | undefined => {
    const raw = (override ?? ui.getPrompt()).trim();
    return raw.length > 0 && raw !== DEFAULT_PROMPT ? raw : undefined;
  };

  const buildShareUrl = (cfg?: Partial<TuningConfig>, promptOverride?: string): string => {
    const target = cfg
      ? normalizeTuningConfig(cfg, gpu?.info ?? null, tuningConfig)
      : bestEval && bestEval.score > 0
        ? bestEval.config
        : tuningConfig;
    return encodeConfigToUrl(target, pageOriginPath, true, getCustomPrompt(promptOverride));
  };

  const publishLabState = (climbing = false) => {
    const shareUrl = buildShareUrl();
    ui.setTuningState({
      config: tuningConfig,
      climbing,
      lastEval,
      bestEval,
      history: evalHistory,
      shareUrl,
    });
    syncDomLabJson({
      version: "1.0",
      ready: Boolean(gpu),
      adapter: gpu?.info ?? null,
      model: QWEN3_0_6B_4BIT.repoId,
      config: tuningConfig,
      last: lastEval,
      best: bestEval,
      history: evalHistory,
      shareUrl,
    });
  };

  const applyTuningPatch = (patch: Partial<TuningConfig>, syncUrl = true): TuningConfig => {
    const prevFault = `${tuningConfig.faultEnabled}:${tuningConfig.faultTarget}`;
    tuningConfig = normalizeTuningConfig(patch, gpu?.info ?? null, tuningConfig);
    ui.setOptFlags(toOptFlags(tuningConfig));
    ui.setFaultConfig(toFaultConfig(tuningConfig));
    const nextFault = `${tuningConfig.faultEnabled}:${tuningConfig.faultTarget}`;
    if (gpu && prevFault !== nextFault) {
      void runFullParitySuite(gpu.device, toFaultConfig(tuningConfig)).then((results) => {
        ui.setParityResults(results);
      });
    }
    if (syncUrl && typeof history !== "undefined" && history.replaceState) {
      const nextUrl = encodeConfigToUrl(
        tuningConfig,
        pageOriginPath,
        ui.getState().zoom === 4,
        getCustomPrompt(),
      );
      history.replaceState(null, "", nextUrl);
    }
    publishLabState(false);
    return { ...tuningConfig };
  };

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

  const ensureReferenceTokens = async (prompt: string, tokenBudget: number): Promise<number[]> => {
    const key = `${prompt}::${tokenBudget}`;
    const cached = refTokenCache.get(key);
    if (cached) return cached;
    const eng = await ensureEngine();
    ui.setStatus("Computing canonical IEEE f32 reference tokens…", "busy");
    const refCfg = normalizeTuningConfig(REFERENCE_TUNING_CONFIG, gpu?.info ?? null);
    const out = await eng.generate(
      prompt,
      tokenBudget,
      () => toOptFlags(refCfg),
      () => ({ enabled: false, target: "attention" }),
      () => {},
      () => stopRequested,
      () => toExecTuning(refCfg),
    );
    if (out.tokens.length > 0) {
      refTokenCache.set(key, out.tokens);
    }
    return out.tokens;
  };

  const evaluateTuningConfig = async (
    cfg: TuningConfig,
    prompt: string,
    tokenBudget: number,
    label: string,
    step: number,
  ): Promise<TuningEvalResult> => {
    const eng = await ensureEngine();
    const refTokens = await ensureReferenceTokens(prompt, tokenBudget);
    const normalized = normalizeTuningConfig(cfg, gpu?.info ?? null);

    // 2-token silent warmup on the candidate pipelines so compile/clock warmup never biases speed
    await eng.generate(
      prompt,
      2,
      () => toOptFlags(normalized),
      () => toFaultConfig(normalized),
      () => {},
      () => stopRequested,
      () => toExecTuning(normalized),
    );

    const out = await eng.generate(
      prompt,
      tokenBudget,
      () => toOptFlags(normalized),
      () => toFaultConfig(normalized),
      () => {},
      () => stopRequested,
      () => toExecTuning(normalized),
    );

    const scored = scoreTuningRun({
      tokens: out.tokens,
      refTokens,
      avgTokPerSec: out.avgTokPerSec,
    });

    return {
      step,
      label,
      config: normalized,
      configKey: configKey(normalized),
      tokPerSec: Number(out.avgTokPerSec.toFixed(2)),
      msPerTok: out.avgTokPerSec > 0 ? Number((1000 / out.avgTokPerSec).toFixed(2)) : 0,
      dispatchesPerTok: out.dispatchesPerTok,
      tokens: out.tokens,
      text: out.text,
      parityMatched: scored.parityMatched,
      parityTotal: scored.parityTotal,
      parityPass: scored.parityPass,
      score: scored.score,
      status: scored.parityPass ? "manual" : "parity_fail",
      shareUrl: encodeConfigToUrl(normalized, pageOriginPath, true, getCustomPrompt(prompt)),
      timestampMs: Date.now(),
    };
  };

  const runSingleTuningEval = async (options?: {
    config?: Partial<TuningConfig>;
    prompt?: string;
    tokens?: number;
    label?: string;
  }): Promise<TuningEvalResult> => {
    stopRequested = false;
    if (options?.prompt !== undefined) {
      ui.setPrompt(options.prompt);
    }
    const prompt = options?.prompt ?? ui.getPrompt();
    const tokenBudget = options?.tokens ?? 24;
    if (options?.config) {
      applyTuningPatch(options.config, true);
    }
    const targetCfg = { ...tuningConfig };
    ui.setError(null);
    ui.setGenerating(true);
    publishLabState(false);

    try {
      ui.setStatus(`Benchmarking config (${targetCfg.qmmPrecision}, wg=${targetCfg.qmmWorkgroupSize}, ${targetCfg.dispatchBatching})…`, "busy");
      const res = await evaluateTuningConfig(
        targetCfg,
        prompt,
        tokenBudget,
        options?.label ?? `manual (${targetCfg.qmmPrecision}, wg=${targetCfg.qmmWorkgroupSize})`,
        manualStepCounter++,
      );
      lastEval = res;
      evalHistory.push(res);
      if (res.parityPass && (!bestEval || res.score > bestEval.score)) {
        bestEval = res;
      }
      publishLabState(false);
      ui.setStatus(
        res.parityPass
          ? `Score ${res.score.toFixed(1)} (${res.tokPerSec.toFixed(1)} tok/s, ${res.dispatchesPerTok} disp/tok, ${res.parityMatched}/${res.parityTotal} exact)`
          : `PARITY FAIL -> Score 0 (${res.parityMatched}/${res.parityTotal} matched reference)`,
        res.parityPass ? "ok" : "error",
      );
      return res;
    } catch (err) {
      ui.setError(err instanceof Error ? err.message : String(err));
      throw err;
    } finally {
      ui.setGenerating(false);
      publishLabState(false);
    }
  };

  const runTuningHillClimb = async (options?: {
    start?: "baseline" | "current";
    prompt?: string;
    tokens?: number;
    maxPasses?: number;
  }): Promise<HillClimbReport> => {
    stopRequested = false;
    if (options?.prompt !== undefined) {
      ui.setPrompt(options.prompt);
    }
    const prompt = options?.prompt ?? ui.getPrompt();
    const tokenBudget = options?.tokens ?? 24;
    const startMode = options?.start ?? "baseline";
    const startCfg =
      startMode === "current"
        ? normalizeTuningConfig({ ...tuningConfig, faultEnabled: false }, gpu?.info ?? null)
        : normalizeTuningConfig(BASELINE_TUNING_CONFIG, gpu?.info ?? null);

    ui.setError(null);
    ui.setGenerating(true);
    evalHistory = [];
    lastEval = null;
    bestEval = null;
    publishLabState(true);

    try {
      const refTokens = await ensureReferenceTokens(prompt, tokenBudget);
      const report = await runCoordinateHillClimb({
        prompt,
        tokenBudget,
        startConfig: startCfg,
        adapter: gpu?.info ?? null,
        referenceTokens: refTokens,
        maxPasses: options?.maxPasses,
        evaluate: async (cfg, label, step) => {
          applyTuningPatch(cfg, false);
          ui.setStatus(`Hill Climb step ${step}: testing ${label}…`, "busy");
          return evaluateTuningConfig(cfg, prompt, tokenBudget, label, step);
        },
        onStep: (stepRes, best, allSteps) => {
          lastEval = stepRes;
          bestEval = best.score > 0 ? best : bestEval;
          evalHistory = allSteps;
          publishLabState(true);
        },
        shouldStop: () => stopRequested,
      });

      // Adopt the winning config and update shareable URL
      manualStepCounter = report.steps.length;
      if (report.best.score > 0) {
        bestEval = report.best;
        lastEval = report.best;
        applyTuningPatch(report.best.config, true);
      }
      publishLabState(false);
      const passTag = report.passesCompleted > 1 ? `, ${report.passesCompleted} passes` : "";
      ui.setStatus(
        `Hill Climb complete: ${report.baseline.score.toFixed(1)} -> ${report.best.score.toFixed(1)} tok/s (+${report.gainPct.toFixed(1)}%, ${report.best.dispatchesPerTok} disp/tok${passTag}, ${report.best.parityMatched}/${report.best.parityTotal} exact)`,
        "ok",
      );
      return report;
    } catch (err) {
      ui.setError(err instanceof Error ? err.message : String(err));
      throw err;
    } finally {
      ui.setGenerating(false);
      publishLabState(false);
    }
  };

  const runAbRace = async (prompt: string, tokenBudget = 24): Promise<AbRaceResult | null> => {
    stopRequested = false;
    applyTuningPatch({ faultEnabled: false }, false);
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
      const execNow = toExecTuning(tuningConfig);

      // Short silent GPU warmup (3 tokens) so shader JIT and GPU clocks are identical for both legs
      await eng.generate(prompt, 3, () => baseOpts, () => ({ enabled: false }), () => {}, () => stopRequested, () => execNow);
      if (stopRequested) return null;

      // Leg A: Unfused baseline (537 dispatches/tok)
      applyTuningPatch(baseOpts, false);
      ui.setAbRace({ prompt, phase: "baseline", tokenBudget });
      ui.setStatus("A/B Race 1/2: Unfused baseline (537 dispatches/tok)…", "busy");
      const legA = await eng.generate(
        prompt,
        tokenBudget,
        () => baseOpts,
        () => ({ enabled: false }),
        (trace) => ui.pushTokenTrace(trace),
        () => stopRequested,
        () => execNow,
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
        dispatchesPerTok: legA.dispatchesPerTok || 537,
      };

      // Leg B: Optimized (PR #997 + fusions, 397 dispatches/tok)
      applyTuningPatch(optAll, false);
      ui.clearTokens();
      ui.setAbRace({ prompt, phase: "optimized", tokenBudget, baseline });
      ui.setStatus("A/B Race 2/2: PR #997 + fused kernels (397 dispatches/tok)…", "busy");
      const legB = await eng.generate(
        prompt,
        tokenBudget,
        () => optAll,
        () => ({ enabled: false }),
        (trace) => ui.pushTokenTrace(trace),
        () => stopRequested,
        () => execNow,
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
        dispatchesPerTok: legB.dispatchesPerTok || 397,
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
      applyTuningPatch(next, ui.getState().zoom === 4);
    },
    onFaultChange(next) {
      applyTuningPatch(
        { faultEnabled: next.enabled, faultTarget: next.target ?? tuningConfig.faultTarget },
        ui.getState().zoom === 4,
      );
    },
    onTuningConfigChange(patch) {
      applyTuningPatch(patch, true);
    },
    async onRunTuningBenchmark(prompt) {
      await runSingleTuningEval({ prompt, tokens: 24 });
    },
    async onRunHillClimb(prompt) {
      await runTuningHillClimb({ start: "baseline", prompt, tokens: 24 });
    },
    onResetTuningBaseline() {
      applyTuningPatch(BASELINE_TUNING_CONFIG, true);
    },
    onStop() {
      stopRequested = true;
    },
    onReset() {
      stopRequested = true;
      ui.clearTokens();
      ui.setAbRace(null);
      ui.setStatus(engine ? "Model ready" : "WebGPU parity PASS — click Generate, Race A/B, or Tuning Lab", "ok");
    },
    async onRunParity(faultOrEnabled) {
      if (!gpu) return [];
      const fc: FaultConfig =
        typeof faultOrEnabled === "boolean"
          ? { enabled: faultOrEnabled, target: tuningConfig.faultTarget }
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
      applyTuningPatch(
        { ...opts, faultEnabled: fault.enabled, faultTarget: fault.target ?? tuningConfig.faultTarget },
        false,
      );
      stopRequested = false;
      ui.setError(null);
      ui.clearTokens();
      ui.setGenerating(true);

      try {
        const eng = await ensureEngine();
        await eng.generate(
          prompt,
          96,
          () => toOptFlags(tuningConfig),
          () => toFaultConfig(tuningConfig),
          (trace) => ui.pushTokenTrace(trace),
          () => stopRequested,
          () => toExecTuning(tuningConfig),
        );
      } catch (err) {
        ui.setError(err instanceof Error ? err.message : String(err));
      } finally {
        ui.setGenerating(false);
      }
    },
  });

  if (urlParsed.prompt) {
    ui.setPrompt(urlParsed.prompt);
  }
  ui.setOptFlags(toOptFlags(tuningConfig));
  ui.setFaultConfig(toFaultConfig(tuningConfig));
  if (urlParsed.openTuningTab) {
    ui.setZoom(4);
  }

  window.__RUN_AB_RACE__ = (prompt, tokenBudget = 24) =>
    runAbRace(prompt ?? DEFAULT_PROMPT, tokenBudget);

  window.__SMOKE_GENERATE__ = async (prompt, maxTokens = 24, optsOverride, faultOn = false) => {
    const faultPatch: Partial<TuningConfig> =
      typeof faultOn === "boolean"
        ? { faultEnabled: faultOn, faultTarget: "attention" }
        : { faultEnabled: faultOn.enabled, faultTarget: faultOn.target ?? "attention" };
    applyTuningPatch({ ...DEFAULT_TUNING_CONFIG, ...optsOverride, ...faultPatch }, false);
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
        () => toOptFlags(tuningConfig),
        () => toFaultConfig(tuningConfig),
        (trace) => {
          lastDispatches = trace.totalDispatches;
          lastAttn = trace.attentionDispatches;
          ui.pushTokenTrace(trace);
        },
        () => stopRequested,
        () => toExecTuning(tuningConfig),
      );
      return {
        ...out,
        totalDispatches: lastDispatches || out.dispatchesPerTok,
        attentionDispatches: lastAttn,
      };
    } finally {
      ui.setGenerating(false);
    }
  };

  window.omarchyLab = {
    version: "1.0",
    getState: () => ({
      ready: Boolean(gpu),
      adapter: gpu?.info ?? null,
      model: QWEN3_0_6B_4BIT.repoId,
      config: { ...tuningConfig },
      levers: getLeverSpecs(gpu?.info ?? null),
      last: lastEval,
      best: bestEval,
      history: [...evalHistory],
      shareUrl: buildShareUrl(),
    }),
    getLevers: () => getLeverSpecs(gpu?.info ?? null),
    set: (patch) => applyTuningPatch(patch, true),
    run: (options) => runSingleTuningEval(options),
    hillClimb: (options) => runTuningHillClimb(options),
    read: () => ({
      adapter: gpu?.info ?? null,
      config: { ...tuningConfig },
      last: lastEval,
      best: bestEval,
      history: [...evalHistory],
      shareUrl: buildShareUrl(),
    }),
    shareUrl: (cfg, prompt) => buildShareUrl(cfg, prompt),
    stop: () => {
      stopRequested = true;
    },
  };

  try {
    ui.setStatus("Initializing WebGPU…", "busy");
    gpu = await initWebGpu();
    ui.setAdapterInfo(gpu.info);
    // Re-normalize initial URL config against real adapter capabilities (f16, subgroups, maxWG)
    tuningConfig = normalizeTuningConfig(tuningConfig, gpu.info);
    ui.setOptFlags(toOptFlags(tuningConfig));
    ui.setFaultConfig(toFaultConfig(tuningConfig));
    publishLabState(false);

    const cleanResults = await runFullParitySuite(gpu.device, false);
    const faultResults = [await verifySdpaOnlineParity(gpu.device, 16, 8, 192, true)];
    ui.setParityResults(
      tuningConfig.faultEnabled
        ? await runFullParitySuite(gpu.device, toFaultConfig(tuningConfig))
        : cleanResults,
    );

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
        ? `${cleanResults.length}/${cleanResults.length} kernels PASS CPU parity — click Generate or open Tuning Lab`
        : "WebGPU parity check failed",
      allCleanPassed ? "ok" : "error",
    );
    document.title =
      allCleanPassed && allFaultsDetected
        ? "PARITY_PASS — omarchy-mlx WebGPU"
        : "PARITY_FAIL — omarchy-mlx WebGPU";

    if (urlParsed.autorun && allCleanPassed) {
      ui.setZoom(4);
      void runSingleTuningEval({ prompt: ui.getPrompt(), tokens: 16, label: "URL autorun" });
    }
  } catch (err) {
    ui.setError(err instanceof Error ? err.message : String(err));
    document.title =
      err instanceof WebGpuUnavailableError
        ? "WebGPU unavailable — omarchy-mlx WebGPU"
        : "PARITY_ERROR — omarchy-mlx WebGPU";
  }
}

boot();
