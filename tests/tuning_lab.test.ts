import { describe, expect, it } from "bun:test";
import type { GpuAdapterInfo, TuningConfig } from "../src/kernels/types";
import {
  buildQmmVecSwigluWgsl,
  buildQmmVecWgsl,
  buildSdpaOnlineWgsl,
  qmmDispatchGridForWg,
  qmmRowsPerWg,
} from "../src/kernels/wgsl_variants";
import {
  BASELINE_TUNING_CONFIG,
  DEFAULT_TUNING_CONFIG,
  configKey,
  encodeConfigToUrl,
  getLeverSpecs,
  medianNumber,
  normalizeTuningConfig,
  parseTuningUrl,
  scoreTuningRun,
} from "../src/tuning/config";
import { runCoordinateHillClimb } from "../src/tuning/hill_climb";
import { summarizeTrajectoryInsight } from "../src/ui/tuning_lab";
import { initWebGpu, WebGpuUnavailableError } from "../src/webgpu/device";

const METAL_ADAPTER: GpuAdapterInfo = {
  vendor: "apple",
  architecture: "metal-3",
  device: "Apple GPU",
  description: "apple metal-3",
  maxBufferSize: 512 * 1024 * 1024,
  maxStorageBufferBindingSize: 256 * 1024 * 1024,
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeInvocationsPerWorkgroup: 1024,
  hasTimestampQuery: true,
  hasShaderF16: true,
  hasSubgroups: true,
};

const BASIC_ADAPTER: GpuAdapterInfo = {
  ...METAL_ADAPTER,
  vendor: "google",
  architecture: "swiftshader",
  maxComputeInvocationsPerWorkgroup: 256,
  hasShaderF16: false,
  hasSubgroups: false,
};

describe("Tuning Lab WGSL variants & grid math", () => {
  it("computes rows/WG and 2D-safe dispatch grids for 64, 128, and 256 workgroups", () => {
    expect(qmmRowsPerWg(64)).toBe(4);
    expect(qmmRowsPerWg(128)).toBe(8);
    expect(qmmRowsPerWg(256)).toBe(16);

    // Qwen3 vocab N=151936 stays 1D (<= 65535) for all 3 workgroup sizes
    expect(qmmDispatchGridForWg(151936, 64)).toEqual([37984, 1, 1]);
    expect(qmmDispatchGridForWg(151936, 128)).toEqual([18992, 1, 1]);
    expect(qmmDispatchGridForWg(151936, 256)).toEqual([9496, 1, 1]);
  });

  it("emits valid WGSL extensions and workgroup sizes for QMM and SDPA variants", () => {
    const f16Sg = buildQmmVecWgsl({ wgSize: 256, precision: "f16", useSubgroups: true });
    expect(f16Sg).toContain("enable f16;");
    expect(f16Sg).toContain("enable subgroups;");
    expect(f16Sg).toContain("@workgroup_size(256, 1, 1)");
    expect(f16Sg).toContain("ROWS_PER_WG: u32 = 16u");
    expect(f16Sg).toContain("subgroupAdd(vec2<f32>(acc0, acc1))");

    const f16AccumSwiglu = buildQmmVecSwigluWgsl({
      wgSize: 64,
      precision: "f16_accum",
      useSubgroups: false,
    });
    expect(f16AccumSwiglu).toContain("enable f16;");
    expect(f16AccumSwiglu).not.toContain("enable subgroups;");
    expect(f16AccumSwiglu).toContain("@workgroup_size(64, 1, 1)");
    expect(f16AccumSwiglu).toContain("ROWS_PER_WG: u32 = 4u");
    expect(f16AccumSwiglu).toContain("acc_g0_h: f16");

    const sdpaSg = buildSdpaOnlineWgsl(true);
    expect(sdpaSg).toContain("enable subgroups;");
    expect(sdpaSg).toContain("score = subgroupAdd(dot);");
  });
});

describe("Tuning Lab config, URL share links, and parity-gated scoring", () => {
  it("clamps unsupported GPU features cleanly on basic adapters", () => {
    const requested: Partial<TuningConfig> = {
      qmmWorkgroupSize: 256,
      qmmPrecision: "f16",
      useSubgroups: true,
      dispatchBatching: "per_layer_pass",
    };
    const onMetal = normalizeTuningConfig(requested, METAL_ADAPTER);
    expect(onMetal.qmmPrecision).toBe("f16");
    expect(onMetal.useSubgroups).toBe(true);
    expect(onMetal.qmmWorkgroupSize).toBe(256);

    const onBasic = normalizeTuningConfig(requested, BASIC_ADAPTER);
    expect(onBasic.qmmPrecision).toBe("f32");
    expect(onBasic.useSubgroups).toBe(false);
    expect(onBasic.qmmWorkgroupSize).toBe(256);

    const basicLevers = getLeverSpecs(BASIC_ADAPTER);
    expect(basicLevers.find((l) => l.key === "qmmPrecision")?.supportedValues).toEqual(["f32"]);
    expect(basicLevers.find((l) => l.key === "useSubgroups")?.supportedValues).toEqual([false]);
  });

  it("round-trips share URL encoding and parsing (including optional custom prompt)", () => {
    const cfg: TuningConfig = {
      ...DEFAULT_TUNING_CONFIG,
      oneDispatchAttention: true,
      fusedRopeNorm: false,
      fusedSwigluQmm: true,
      qmmWorkgroupSize: 256,
      qmmPrecision: "f16",
      useSubgroups: true,
      dispatchBatching: "single_pass",
      decodePacing: "unthrottled",
      gpuTokenChain: 4,
    };
    const url = encodeConfigToUrl(cfg, "https://joshuaswarren.github.io/omarchy-mlx-webgpu/");
    expect(url).toContain("chain=4");
    const parsed = parseTuningUrl(new URL(url).search);
    expect(parsed.openTuningTab).toBe(true);
    expect(parsed.hasTuningParams).toBe(true);
    expect(parsed.prompt).toBeUndefined();
    const roundTripped = normalizeTuningConfig(parsed.patch, METAL_ADAPTER);
    expect(roundTripped.gpuTokenChain).toBe(4);
    expect(configKey(roundTripped)).toBe(configKey(cfg));

    const customPrompt = "Explain RMSNorm in one sentence.";
    const urlWithPrompt = encodeConfigToUrl(
      cfg,
      "https://joshuaswarren.github.io/omarchy-mlx-webgpu/",
      true,
      customPrompt,
    );
    const parsedWithPrompt = parseTuningUrl(new URL(urlWithPrompt).search);
    expect(parsedWithPrompt.prompt).toBe(customPrompt);
    expect(configKey(normalizeTuningConfig(parsedWithPrompt.patch, METAL_ADAPTER))).toBe(configKey(cfg));
  });

  it("scores exact reference match as tok/s and scores ANY output divergence as 0", () => {
    const ref = [10, 20, 30, 40];
    const ok = scoreTuningRun({ tokens: [10, 20, 30, 40], refTokens: ref, avgTokPerSec: 112.456 });
    expect(ok.parityPass).toBe(true);
    expect(ok.parityMatched).toBe(4);
    expect(ok.score).toBe(112.46);

    const flippedOne = scoreTuningRun({ tokens: [10, 20, 99, 40], refTokens: ref, avgTokPerSec: 250.0 });
    expect(flippedOne.parityPass).toBe(false);
    expect(flippedOne.parityMatched).toBe(3);
    expect(flippedOne.score).toBe(0);

    const truncated = scoreTuningRun({ tokens: [10, 20, 30], refTokens: ref, avgTokPerSec: 180.0 });
    expect(truncated.parityPass).toBe(false);
    expect(truncated.score).toBe(0);
  });
});

describe("Coordinate-ascent hill climber", () => {
  it("climbs to the fastest parity-passing config and rejects a faster parity-failing config", async () => {
    const refTokens = [1, 2, 3, 4];
    const report = await runCoordinateHillClimb({
      prompt: "test",
      tokenBudget: 4,
      startConfig: BASELINE_TUNING_CONFIG,
      adapter: METAL_ADAPTER,
      referenceTokens: refTokens,
      evaluate: async (cfg, label, step) => {
        // Synthetic speed model: f16_accum is fastest (200 tok/s) but breaks parity -> MUST score 0!
        let tps = 50;
        if (cfg.dispatchBatching === "single_pass") tps += 15;
        else if (cfg.dispatchBatching === "per_layer_pass") tps += 6;
        if (cfg.oneDispatchAttention) tps += 12;
        if (cfg.fusedRopeNorm) tps += 8;
        if (cfg.fusedSwigluQmm) tps += 10;
        if (cfg.qmmWorkgroupSize === 256) tps += 9;
        else if (cfg.qmmWorkgroupSize === 128) tps += 4;
        if (cfg.useSubgroups) tps += 11;
        if (cfg.qmmPrecision === "f16") tps += 7;
        if (cfg.qmmPrecision === "f16_accum") tps += 80; // trap: faster but breaks parity!
        if (cfg.decodePacing === "unthrottled") tps += 14;
        else if (cfg.decodePacing === "every_4") tps += 6;

        const tokens = cfg.qmmPrecision === "f16_accum" ? [1, 2, 999, 4] : [...refTokens];
        const scored = scoreTuningRun({ tokens, refTokens, avgTokPerSec: tps });
        return {
          step,
          label,
          config: cfg,
          configKey: configKey(cfg),
          tokPerSec: tps,
          msPerTok: 1000 / tps,
          dispatchesPerTok: 397,
          tokens,
          text: "ok",
          ...scored,
          status: scored.parityPass ? "manual" : "parity_fail",
          shareUrl: encodeConfigToUrl(cfg, "https://example.test/"),
          timestampMs: Date.now(),
        };
      },
    });

    expect(report.baseline.score).toBe(50);
    // Expected winner: single_pass + all 3 fusions + wg=256 + subgroups + f16 (NOT f16_accum!) + unthrottled = 136
    expect(report.best.parityPass).toBe(true);
    expect(report.best.config.qmmPrecision).toBe("f16");
    expect(report.best.config.qmmWorkgroupSize).toBe(256);
    expect(report.best.config.useSubgroups).toBe(true);
    expect(report.best.config.dispatchBatching).toBe("single_pass");
    expect(report.best.config.decodePacing).toBe("unthrottled");
    expect(report.best.score).toBe(136);

    const failStep = report.steps.find((s) => s.config.qmmPrecision === "f16_accum");
    expect(failStep).toBeDefined();
    expect(failStep!.parityPass).toBe(false);
    expect(failStep!.score).toBe(0);
    expect(failStep!.status).toBe("parity_fail");
    expect(report.passesCompleted).toBeGreaterThanOrEqual(1);
  });

  it("uses multi-pass coordinate ascent to unlock non-separable lever synergies", async () => {
    const refTokens = [7, 8, 9];
    const report = await runCoordinateHillClimb({
      prompt: "synergy",
      tokenBudget: 3,
      startConfig: BASELINE_TUNING_CONFIG,
      adapter: BASIC_ADAPTER,
      referenceTokens: refTokens,
      maxPasses: 2,
      evaluate: async (cfg, label, step) => {
        // Synthetic interaction: fusedRopeNorm is slightly slower at wg=64 (-2),
        // but once wg=256 wins later in pass 1, wg=256 + fusedRopeNorm unlocks +30!
        let tps = 40;
        if (cfg.qmmWorkgroupSize === 256) tps += 10;
        if (cfg.fusedRopeNorm) {
          tps += cfg.qmmWorkgroupSize === 256 ? 30 : -2;
        }
        const scored = scoreTuningRun({ tokens: refTokens, refTokens, avgTokPerSec: tps });
        return {
          step,
          label,
          config: cfg,
          configKey: configKey(cfg),
          tokPerSec: tps,
          msPerTok: 1000 / tps,
          dispatchesPerTok: 481,
          tokens: [...refTokens],
          text: "ok",
          ...scored,
          status: "manual",
          shareUrl: encodeConfigToUrl(cfg, "https://example.test/", true, "synergy"),
          timestampMs: Date.now(),
        };
      },
    });

    // Pass 1 takes wg=256 (50), Pass 2 revisits rope=true on top of wg=256 -> 80!
    expect(report.passesCompleted).toBe(2);
    expect(report.best.config.qmmWorkgroupSize).toBe(256);
    expect(report.best.config.fusedRopeNorm).toBe(true);
    expect(report.best.score).toBe(80);
    expect(report.steps.some((s) => s.label === "p2: rope=true" && s.status === "accepted")).toBe(true);
  });

  it("explains clearly when fewer dispatches was not faster on the visitor's GPU", () => {
    const mkEval = (
      step: number,
      label: string,
      dispatchesPerTok: number,
      tokPerSec: number,
      parityPass = true,
    ) => ({
      step,
      label,
      config: BASELINE_TUNING_CONFIG,
      configKey: `k${step}`,
      tokPerSec,
      msPerTok: 1000 / tokPerSec,
      dispatchesPerTok,
      tokens: [1, 2],
      text: "ok",
      parityMatched: parityPass ? 2 : 1,
      parityTotal: 2,
      parityPass,
      score: parityPass ? tokPerSec : 0,
      status: parityPass ? ("accepted" as const) : ("parity_fail" as const),
      shareUrl: "https://example.test/",
      timestampMs: 0,
    });

    const best509 = mkEval(9, "wg=256", 509, 104.3, true);
    const slower453 = mkEval(6, "rope=true", 453, 70.9, true);
    const msg = summarizeTrajectoryInsight([slower453, best509], best509);
    expect(msg).toContain("On this GPU, fewer dispatches (453/tok at 70.9 tok/s) was not faster than 509/tok (104.3 tok/s)");

    const best397 = mkEval(5, "swiglu=true", 397, 130.0, true);
    const base537 = mkEval(0, "Start baseline", 537, 80.0, true);
    const msgFast = summarizeTrajectoryInsight([base537, best397], best397);
    expect(msgFast).toContain("cutting dispatches to 397/tok was also fastest (130.0 tok/s");
  });

  it("computes medians cleanly and enforces minGainPct noise floor in hill climbing", async () => {
    expect(medianNumber([133.7, 106.7, 120.7])).toBe(120.7);
    expect(medianNumber([10, 40, 20, 30])).toBe(25);
    expect(medianNumber([])).toBe(0);

    const refTokens = [4, 5, 6];
    const report = await runCoordinateHillClimb({
      prompt: "noise",
      tokenBudget: 3,
      startConfig: BASELINE_TUNING_CONFIG,
      adapter: BASIC_ADAPTER,
      referenceTokens: refTokens,
      minGainPct: 2.0,
      evaluate: async (cfg, label, step) => {
        // Tiny +0.5% jitter on oneDispatchAttention must be rejected under 2% noise floor,
        // while real +40% gain on gpuTokenChain=4 must be accepted!
        let tps = 100.0;
        if (cfg.oneDispatchAttention) tps += 0.5;
        if (cfg.gpuTokenChain === 4) tps += 40.0;
        const scored = scoreTuningRun({ tokens: refTokens, refTokens, avgTokPerSec: tps });
        return {
          step,
          label,
          config: cfg,
          configKey: configKey(cfg),
          tokPerSec: tps,
          msPerTok: 1000 / tps,
          dispatchesPerTok: 537,
          tokens: [...refTokens],
          text: "ok",
          ...scored,
          status: "manual",
          shareUrl: encodeConfigToUrl(cfg, "https://example.test/"),
          timestampMs: Date.now(),
        };
      },
    });

    expect(report.best.config.gpuTokenChain).toBe(4);
    expect(report.best.config.oneDispatchAttention).toBe(false);
    expect(report.best.score).toBe(140);
  });

  it("throws WebGpuUnavailableError when navigator.gpu is missing", async () => {
    await expect(initWebGpu()).rejects.toBeInstanceOf(WebGpuUnavailableError);
  });
});
