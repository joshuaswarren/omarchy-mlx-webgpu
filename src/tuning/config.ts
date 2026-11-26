import type {
  DecodePacing,
  DispatchBatching,
  FaultTarget,
  GpuAdapterInfo,
  LeverSpec,
  QmmPrecision,
  QmmWorkgroupSize,
  TuningConfig,
} from "../kernels/types";

export const DEFAULT_TUNING_CONFIG: TuningConfig = {
  oneDispatchAttention: true,
  fusedRopeNorm: true,
  fusedSwigluQmm: true,
  qmmWorkgroupSize: 128,
  qmmPrecision: "f32",
  useSubgroups: false,
  dispatchBatching: "single_pass",
  decodePacing: "every_token",
  faultEnabled: false,
  faultTarget: "attention",
};

/** Slow, unoptimized starting point for visible hill-climbing trajectories. */
export const BASELINE_TUNING_CONFIG: TuningConfig = {
  oneDispatchAttention: false,
  fusedRopeNorm: false,
  fusedSwigluQmm: false,
  qmmWorkgroupSize: 64,
  qmmPrecision: "f32",
  useSubgroups: false,
  dispatchBatching: "chunked_submit",
  decodePacing: "every_token",
  faultEnabled: false,
  faultTarget: "attention",
};

/** Canonical parity reference configuration (guaranteed portable IEEE f32). */
export const REFERENCE_TUNING_CONFIG: TuningConfig = {
  oneDispatchAttention: false,
  fusedRopeNorm: false,
  fusedSwigluQmm: false,
  qmmWorkgroupSize: 128,
  qmmPrecision: "f32",
  useSubgroups: false,
  dispatchBatching: "single_pass",
  decodePacing: "unthrottled",
  faultEnabled: false,
  faultTarget: "attention",
};

const VALID_WG: readonly QmmWorkgroupSize[] = [64, 128, 256];
const VALID_PREC: readonly QmmPrecision[] = ["f32", "f16", "f16_accum"];
const VALID_BATCH: readonly DispatchBatching[] = ["single_pass", "per_layer_pass", "chunked_submit"];
const VALID_PACE: readonly DecodePacing[] = ["every_token", "every_4", "unthrottled"];
const VALID_FAULT_TARGET: readonly FaultTarget[] = ["attention", "qmm", "rope", "norm"];

export function getLeverSpecs(adapter: GpuAdapterInfo | null): LeverSpec[] {
  const f16Ok = Boolean(adapter?.hasShaderF16);
  const sgOk = Boolean(adapter?.hasSubgroups);
  const maxWg = adapter?.maxComputeInvocationsPerWorkgroup ?? 256;
  const wgSupported = VALID_WG.filter((w) => w <= maxWg);

  return [
    {
      key: "oneDispatchAttention",
      label: "PR #997 1-dispatch attention",
      description: "Online softmax replaces 2-dispatch split-K + scratch (-28 dispatches/tok)",
      values: [false, true],
      supportedValues: [false, true],
      urlParam: "attn",
    },
    {
      key: "fusedRopeNorm",
      label: "Fused QK-Norm + RoPE",
      description: "Single shared-memory pass per Q/K head (-56 dispatches/tok)",
      values: [false, true],
      supportedValues: [false, true],
      urlParam: "rope",
    },
    {
      key: "fusedSwigluQmm",
      label: "Fused SwiGLU QMM",
      description: "Gate + Up Q4 GEMV + SiLU epilogue in one dispatch (-56 dispatches/tok)",
      values: [false, true],
      supportedValues: [false, true],
      urlParam: "swiglu",
    },
    {
      key: "qmmWorkgroupSize",
      label: "QMM workgroup size",
      description: "Threads per workgroup (64=4 rows, 128=8 rows, 256=16 rows/WG)",
      values: [...VALID_WG],
      supportedValues: wgSupported.length > 0 ? [...wgSupported] : [128],
      urlParam: "wg",
    },
    {
      key: "qmmPrecision",
      label: "QMM precision",
      description: "f32 baseline, f16 inner dot + f32 accum, or full f16_accum (parity-gated)",
      values: [...VALID_PREC],
      supportedValues: f16Ok ? [...VALID_PREC] : ["f32"],
      urlParam: "prec",
    },
    {
      key: "useSubgroups",
      label: "Hardware subgroupAdd",
      description: "Replaces shared-memory reduction barriers with SIMD subgroupAdd",
      values: [false, true],
      supportedValues: sgOk ? [false, true] : [false],
      urlParam: "sg",
    },
    {
      key: "dispatchBatching",
      label: "Command batching",
      description: "1 compute pass/token vs per-layer passes vs 4 queue submits/token",
      values: [...VALID_BATCH],
      supportedValues: [...VALID_BATCH],
      urlParam: "batch",
    },
    {
      key: "decodePacing",
      label: "Decode pacing",
      description: "UI timer yield & trace readback cadence (every_token, every_4, unthrottled)",
      values: [...VALID_PACE],
      supportedValues: [...VALID_PACE],
      urlParam: "pace",
    },
  ];
}

export function normalizeTuningConfig(
  patch: Partial<TuningConfig> | undefined,
  adapter: GpuAdapterInfo | null,
  base: TuningConfig = DEFAULT_TUNING_CONFIG,
): TuningConfig {
  const merged: TuningConfig = {
    ...base,
    ...(patch ?? {}),
  };

  if (!VALID_WG.includes(merged.qmmWorkgroupSize)) {
    merged.qmmWorkgroupSize = 128;
  }
  const maxWg = adapter?.maxComputeInvocationsPerWorkgroup ?? 256;
  if (merged.qmmWorkgroupSize > maxWg) {
    merged.qmmWorkgroupSize = 128;
  }
  if (!VALID_PREC.includes(merged.qmmPrecision)) {
    merged.qmmPrecision = "f32";
  }
  if (adapter && !adapter.hasShaderF16 && merged.qmmPrecision !== "f32") {
    merged.qmmPrecision = "f32";
  }
  if (adapter && !adapter.hasSubgroups && merged.useSubgroups) {
    merged.useSubgroups = false;
  }
  if (!VALID_BATCH.includes(merged.dispatchBatching)) {
    merged.dispatchBatching = "single_pass";
  }
  if (!VALID_PACE.includes(merged.decodePacing)) {
    merged.decodePacing = "every_token";
  }
  if (!VALID_FAULT_TARGET.includes(merged.faultTarget)) {
    merged.faultTarget = "attention";
  }
  merged.oneDispatchAttention = Boolean(merged.oneDispatchAttention);
  merged.fusedRopeNorm = Boolean(merged.fusedRopeNorm);
  merged.fusedSwigluQmm = Boolean(merged.fusedSwigluQmm);
  merged.useSubgroups = Boolean(merged.useSubgroups);
  merged.faultEnabled = Boolean(merged.faultEnabled);

  return merged;
}

export function configKey(c: TuningConfig): string {
  return [
    `a${c.oneDispatchAttention ? 1 : 0}`,
    `r${c.fusedRopeNorm ? 1 : 0}`,
    `s${c.fusedSwigluQmm ? 1 : 0}`,
    `w${c.qmmWorkgroupSize}`,
    `p:${c.qmmPrecision}`,
    `g${c.useSubgroups ? 1 : 0}`,
    `b:${c.dispatchBatching}`,
    `y:${c.decodePacing}`,
    `f${c.faultEnabled ? c.faultTarget : 0}`,
  ].join("|");
}

function parseBoolParam(raw: string | null): boolean | undefined {
  if (raw === null) return undefined;
  const v = raw.trim().toLowerCase();
  if (v === "1" || v === "true" || v === "on" || v === "yes") return true;
  if (v === "0" || v === "false" || v === "off" || v === "no") return false;
  return undefined;
}

export interface ParsedUrlState {
  patch: Partial<TuningConfig>;
  hasTuningParams: boolean;
  openTuningTab: boolean;
  autorun: boolean;
  prompt?: string;
}

export function parseTuningUrl(search: string): ParsedUrlState {
  const params = new URLSearchParams(search);
  const patch: Partial<TuningConfig> = {};

  const attn = parseBoolParam(params.get("attn"));
  if (attn !== undefined) patch.oneDispatchAttention = attn;

  const rope = parseBoolParam(params.get("rope"));
  if (rope !== undefined) patch.fusedRopeNorm = rope;

  const swiglu = parseBoolParam(params.get("swiglu"));
  if (swiglu !== undefined) patch.fusedSwigluQmm = swiglu;

  const wgRaw = params.get("wg");
  if (wgRaw !== null) {
    const n = Number(wgRaw);
    if (n === 64 || n === 128 || n === 256) patch.qmmWorkgroupSize = n;
  }

  const precRaw = params.get("prec") as QmmPrecision | null;
  if (precRaw && VALID_PREC.includes(precRaw)) patch.qmmPrecision = precRaw;

  const sg = parseBoolParam(params.get("sg"));
  if (sg !== undefined) patch.useSubgroups = sg;

  const batchRaw = params.get("batch") as DispatchBatching | null;
  if (batchRaw && VALID_BATCH.includes(batchRaw)) patch.dispatchBatching = batchRaw;

  const paceRaw = params.get("pace") as DecodePacing | null;
  if (paceRaw && VALID_PACE.includes(paceRaw)) patch.decodePacing = paceRaw;

  const fault = parseBoolParam(params.get("fault"));
  if (fault !== undefined) patch.faultEnabled = fault;

  const ft = params.get("faultTarget") as FaultTarget | null;
  if (ft && VALID_FAULT_TARGET.includes(ft)) patch.faultTarget = ft;

  const hasTuningParams = Object.keys(patch).length > 0;
  const tab = params.get("tab");
  const openTuningTab = tab === "tuning" || tab === "4" || hasTuningParams;
  const autorun = parseBoolParam(params.get("autorun")) === true;
  const prompt = params.get("prompt") ?? undefined;

  return { patch, hasTuningParams, openTuningTab, autorun, prompt };
}

export function encodeConfigToUrl(
  config: TuningConfig,
  originAndPath: string,
  includeTab = true,
): string {
  const url = new URL(originAndPath);
  url.search = "";
  if (includeTab) url.searchParams.set("tab", "tuning");
  url.searchParams.set("attn", config.oneDispatchAttention ? "1" : "0");
  url.searchParams.set("rope", config.fusedRopeNorm ? "1" : "0");
  url.searchParams.set("swiglu", config.fusedSwigluQmm ? "1" : "0");
  url.searchParams.set("wg", String(config.qmmWorkgroupSize));
  url.searchParams.set("prec", config.qmmPrecision);
  url.searchParams.set("sg", config.useSubgroups ? "1" : "0");
  url.searchParams.set("batch", config.dispatchBatching);
  url.searchParams.set("pace", config.decodePacing);
  if (config.faultEnabled) {
    url.searchParams.set("fault", "1");
    url.searchParams.set("faultTarget", config.faultTarget);
  }
  return url.toString();
}

export function scoreTuningRun(args: {
  tokens: number[];
  refTokens: number[];
  avgTokPerSec: number;
}): { parityMatched: number; parityTotal: number; parityPass: boolean; score: number } {
  const parityTotal = Math.max(args.tokens.length, args.refTokens.length);
  let parityMatched = 0;
  const n = Math.min(args.tokens.length, args.refTokens.length);
  for (let i = 0; i < n; i++) {
    if (args.tokens[i] === args.refTokens[i]) parityMatched++;
  }
  const parityPass =
    parityTotal > 0 &&
    args.tokens.length === args.refTokens.length &&
    parityMatched === parityTotal;
  const safeTps = Number.isFinite(args.avgTokPerSec) && args.avgTokPerSec > 0 ? args.avgTokPerSec : 0;
  return {
    parityMatched,
    parityTotal,
    parityPass,
    score: parityPass ? Number(safeTps.toFixed(2)) : 0,
  };
}
