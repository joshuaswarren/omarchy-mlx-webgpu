import type {
  GpuAdapterInfo,
  HillClimbReport,
  TuningConfig,
  TuningEvalResult,
} from "../kernels/types";
import { configKey, getLeverSpecs, normalizeTuningConfig } from "./config";

export interface HillClimbOptions {
  prompt: string;
  tokenBudget: number;
  startConfig: TuningConfig;
  adapter: GpuAdapterInfo | null;
  referenceTokens: number[];
  evaluate: (
    config: TuningConfig,
    label: string,
    step: number,
  ) => Promise<TuningEvalResult>;
  onStep?: (stepResult: TuningEvalResult, best: TuningEvalResult, allSteps: TuningEvalResult[]) => void;
  shouldStop?: () => boolean;
}

/**
 * Deterministic coordinate-ascent hill climber over all supported GPU levers.
 *
 * Strictly enforces the parity gate: any candidate that alters output tokens
 * scores 0 and is rejected immediately (`status = "parity_fail"`).
 */
export async function runCoordinateHillClimb(
  opts: HillClimbOptions,
): Promise<HillClimbReport> {
  const { prompt, tokenBudget, adapter, referenceTokens, evaluate, onStep, shouldStop } = opts;
  const steps: TuningEvalResult[] = [];
  const visited = new Set<string>();

  // Start clean with fault disabled unless explicitly testing fault behavior
  const start = normalizeTuningConfig({ ...opts.startConfig, faultEnabled: false }, adapter);
  visited.add(configKey(start));

  const baselineRaw = await evaluate(start, "Start baseline", 0);
  const baseline: TuningEvalResult = {
    ...baselineRaw,
    status: baselineRaw.parityPass ? "baseline" : "parity_fail",
  };
  steps.push(baseline);
  let best = baseline;
  onStep?.(baseline, best, [...steps]);

  const specs = getLeverSpecs(adapter);
  // Order levers: lift browser timer-yield bottleneck first so every GPU kernel
  // lever is measured cleanly GPU-bound, then structural fusions -> micro-arch.
  const leverOrder: Array<keyof TuningConfig> = [
    "decodePacing",
    "dispatchBatching",
    "oneDispatchAttention",
    "fusedRopeNorm",
    "fusedSwigluQmm",
    "qmmWorkgroupSize",
    "useSubgroups",
    "qmmPrecision",
  ];

  let stepNum = 1;
  let stoppedEarly = false;

  for (const key of leverOrder) {
    if (shouldStop?.()) {
      stoppedEarly = true;
      break;
    }
    const spec = specs.find((s) => s.key === key);
    if (!spec) continue;

    // Test every supported alternative value for this lever relative to current best
    for (const candidateVal of spec.supportedValues) {
      if (shouldStop?.()) {
        stoppedEarly = true;
        break;
      }
      if (best.config[key] === candidateVal) continue;

      const candidateCfg = normalizeTuningConfig(
        { ...best.config, [key]: candidateVal },
        adapter,
        best.config,
      );
      const ck = configKey(candidateCfg);
      if (visited.has(ck)) continue;
      visited.add(ck);

      const label = `${spec.urlParam}=${String(candidateVal)}`;
      const raw = await evaluate(candidateCfg, label, stepNum++);

      let status: TuningEvalResult["status"];
      if (!raw.parityPass) {
        status = "parity_fail";
      } else if (raw.score > best.score) {
        status = "accepted";
      } else {
        status = "rejected";
      }

      const recorded: TuningEvalResult = { ...raw, status };
      steps.push(recorded);
      if (status === "accepted") {
        best = recorded;
      }
      onStep?.(recorded, best, [...steps]);
    }
  }

  const gainPct =
    baseline.score > 0 && best.score > 0
      ? ((best.score - baseline.score) / baseline.score) * 100
      : 0;

  return {
    prompt,
    tokenBudget,
    adapter,
    referenceTokens,
    baseline,
    best,
    steps,
    gainPct,
    shareUrl: best.shareUrl,
    stoppedEarly,
  };
}
