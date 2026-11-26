/**
 * Shared types for omarchy-mlx compute kernels carried to WebGPU.
 * All UI, kernel runner, model runtime, and parity modules bind to this contract.
 */

export type KernelId =
  | "fast_norm"
  | "fast_rope"
  | "fast_rope_norm"
  | "qmm_vec"
  | "qmm_vec_swiglu"
  | "swiglu"
  | "sdpa_decode_split_p1"
  | "sdpa_decode_split_p2"
  | "sdpa_decode_online"
  | "embed_q4"
  | "residual_add"
  | "kv_cache_write"
  | "argmax";

/** Real omarchy-mlx optimization toggles controllable live during generation. */
export interface OptFlags {
  /** PR #997: 1-dispatch online-softmax SDPA (sdpa_decode_online) vs 2-dispatch split-K (p1 + p2). */
  oneDispatchAttention: boolean;
  /** fast_rope_norm: fuse per-head QK RMSNorm + RoPE into one dispatch per Q/K tensor (-56 dispatches/tok). */
  fusedRopeNorm: boolean;
  /** qmm_vec QMM_VEC_SWIGLU: fuse Gate+Up Q4 projections + SiLU*Up into one dispatch (-56 dispatches/tok). */
  fusedSwigluQmm: boolean;
}

export const DEFAULT_OPT_FLAGS: OptFlags = {
  oneDispatchAttention: true,
  fusedRopeNorm: true,
  fusedSwigluQmm: true,
};

/** Which real WGSL kernel family the fault switch corrupts (all TDR-safe). */
export type FaultTarget = "attention" | "qmm" | "rope" | "norm";

/** Single fault-injection switch: corrupts live kernel math safely (no TDR/OOB). */
export interface FaultConfig {
  /** When true, activates fault_mode=1 on the selected kernel target. */
  enabled: boolean;
  /** Target kernel family to corrupt (defaults to "attention"). */
  target?: FaultTarget;
}

/** Result of an automated back-to-back A/B decode race on the visitor's GPU. */
export interface AbRaceResult {
  prompt: string;
  phase: "baseline" | "optimized" | "done";
  tokenBudget: number;
  baseline?: {
    tokens: number[];
    text: string;
    avgTokPerSec: number;
    msPerTok: number;
    dispatchesPerTok: number;
  };
  optimized?: {
    tokens: number[];
    text: string;
    avgTokPerSec: number;
    msPerTok: number;
    dispatchesPerTok: number;
  };
  speedupPct?: number;
  exactMatchCount?: number;
  comparedCount?: number;
}

/** One recorded GPU compute dispatch for Zoom Level 2 (Kernel Graph) & Level 3 (Dispatch Inspector). */
export interface DispatchEvent {
  index: number;
  layer: number; // -1 for embed / final norm / lm_head
  kernel: KernelId;
  label: string;
  workgroups: [number, number, number];
  workgroupSize: [number, number, number];
  bytesRead: number;
  bytesWritten: number;
  /** Optional synthetic or measured duration in microseconds. */
  durationUs?: number;
}

/** Per-token execution trace emitted by Qwen3WebGPU to drive the 3-level Zoom UI. */
export interface TokenTrace {
  step: number;
  tokenId: number;
  text: string;
  tokenMs: number;
  tokPerSec: number;
  totalDispatches: number;
  attentionDispatches: number;
  optFlags: OptFlags;
  faultEnabled: boolean;
  faultTarget?: FaultTarget;
  dispatches: DispatchEvent[];
  /** Optional sample of attention weights / hidden activations for Level-3 heatmap. */
  sampleSlice?: Float32Array;
}

/** Numerical parity verdict comparing a WebGPU kernel output against the in-page CPU reference. */
export interface ParityResult {
  kernel: KernelId;
  label: string;
  shape: string;
  elements: number;
  maxAbsDiff: number;
  meanAbsDiff: number;
  cosineSimilarity: number;
  tolerance: number;
  passed: boolean;
  gpuMs: number;
  cpuMs: number;
  faultInjected: boolean;
  /** First 64 elements of (gpu - cpu) for sparkline/heatmap visualization. */
  diffSample: number[];
  gpuSample: number[];
  cpuSample: number[];
}

export interface GpuAdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxComputeWorkgroupStorageSize: number;
  maxComputeInvocationsPerWorkgroup: number;
  hasTimestampQuery: boolean;
  hasShaderF16: boolean;
  hasSubgroups: boolean;
}

export type QmmWorkgroupSize = 64 | 128 | 256;
export type QmmPrecision = "f32" | "f16" | "f16_accum";
export type DispatchBatching = "single_pass" | "per_layer_pass" | "chunked_submit";
export type DecodePacing = "every_token" | "every_4" | "unthrottled";

/** Complete set of real GPU execution levers exposed in the Tuning Lab & Agent API. */
export interface TuningConfig {
  oneDispatchAttention: boolean;
  fusedRopeNorm: boolean;
  fusedSwigluQmm: boolean;
  qmmWorkgroupSize: QmmWorkgroupSize;
  qmmPrecision: QmmPrecision;
  useSubgroups: boolean;
  dispatchBatching: DispatchBatching;
  decodePacing: DecodePacing;
  faultEnabled: boolean;
  faultTarget: FaultTarget;
}

export interface LeverSpec {
  key: keyof TuningConfig;
  label: string;
  description: string;
  values: Array<string | number | boolean>;
  supportedValues: Array<string | number | boolean>;
  urlParam: string;
}

export interface TuningEvalResult {
  step: number;
  label: string;
  config: TuningConfig;
  configKey: string;
  tokPerSec: number;
  msPerTok: number;
  dispatchesPerTok: number;
  tokens: number[];
  text: string;
  parityMatched: number;
  parityTotal: number;
  parityPass: boolean;
  /** Live score: tokPerSec rounded to 2 decimals when parityPass is true, else 0. */
  score: number;
  status: "baseline" | "accepted" | "rejected" | "parity_fail" | "manual";
  shareUrl: string;
  timestampMs: number;
}

export interface HillClimbReport {
  prompt: string;
  tokenBudget: number;
  adapter: GpuAdapterInfo | null;
  referenceTokens: number[];
  baseline: TuningEvalResult;
  best: TuningEvalResult;
  steps: TuningEvalResult[];
  passesCompleted: number;
  gainPct: number;
  shareUrl: string;
  stoppedEarly: boolean;
}

export interface ModelLoadProgress {
  stage: "config" | "tokenizer" | "weights" | "upload" | "ready";
  loadedBytes: number;
  totalBytes: number;
  message: string;
  fromCache?: boolean;
}
