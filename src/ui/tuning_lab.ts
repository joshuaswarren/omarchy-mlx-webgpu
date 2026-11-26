import type {
  DecodePacing,
  DispatchBatching,
  GpuAdapterInfo,
  GpuTokenChain,
  QmmPrecision,
  QmmWorkgroupSize,
  TuningConfig,
  TuningEvalResult,
} from "../kernels/types";
import { getLeverSpecs } from "../tuning/config";

export interface TuningLabHandlers {
  onConfigChange: (patch: Partial<TuningConfig>) => void;
  onRunBenchmark: () => void | Promise<void>;
  onRunHillClimb: () => void | Promise<void>;
  onResetBaseline: () => void;
  onCopyShareUrl: () => void | Promise<void>;
  onSelectStep: (step: TuningEvalResult) => void;
}

export interface TuningLabState {
  adapter: GpuAdapterInfo | null;
  config: TuningConfig;
  busy: boolean;
  climbing: boolean;
  lastEval: TuningEvalResult | null;
  bestEval: TuningEvalResult | null;
  history: TuningEvalResult[];
  shareUrl: string;
  copied: boolean;
  agentRepoUrl: string;
}

export interface TuningLabController {
  update(state: TuningLabState): void;
  resize(): void;
  destroy(): void;
}

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

/**
 * Produces a concise, honest one-line takeaway underneath the hill-climb trajectory
 * so visitors never mistake a measured GPU trade-off (for example fewer dispatches
 * scoring slower on a given adapter) for a bug.
 */
export function summarizeTrajectoryInsight(
  history: TuningEvalResult[],
  bestEval: TuningEvalResult | null,
): string | null {
  if (!bestEval || bestEval.score <= 0 || history.length < 2) return null;

  const passing = history.filter((h) => h.parityPass && h.score > 0);
  const fewerDispPassing = passing
    .filter((h) => h.dispatchesPerTok < bestEval.dispatchesPerTok)
    .sort((a, b) => a.dispatchesPerTok - b.dispatchesPerTok || b.score - a.score);

  if (fewerDispPassing.length > 0) {
    const fewest = fewerDispPassing[0];
    return `On this GPU, fewer dispatches (${fewest.dispatchesPerTok}/tok at ${fewest.score.toFixed(1)} tok/s) was not faster than ${bestEval.dispatchesPerTok}/tok (${bestEval.score.toFixed(1)} tok/s) — kept the faster config.`;
  }

  const fasterFail = history
    .filter((h) => !h.parityPass && h.tokPerSec > bestEval.score)
    .sort((a, b) => b.tokPerSec - a.tokPerSec)[0];
  if (fasterFail) {
    return `On this GPU, ${fasterFail.label} (${fasterFail.tokPerSec.toFixed(1)} tok/s) changed output tokens and scored 0 — kept ${bestEval.score.toFixed(1)} tok/s (${bestEval.dispatchesPerTok} disp/tok).`;
  }

  if (bestEval.dispatchesPerTok < 537) {
    return `On this GPU, cutting dispatches to ${bestEval.dispatchesPerTok}/tok was also fastest (${bestEval.score.toFixed(1)} tok/s, ${bestEval.parityMatched}/${bestEval.parityTotal} exact).`;
  }

  return null;
}

export function createTuningLabView(
  container: HTMLElement,
  handlers: TuningLabHandlers,
): TuningLabController {
  container.innerHTML = "";
  container.classList.add("tl-root");

  container.innerHTML = `
    <div class="tl-scorebar">
      <div class="tl-score-main">
        <div class="tl-score-box" data-tl-score-box>
          <span class="tl-score-k">LIVE SCORE</span>
          <span class="tl-score-v tabular" data-tl-score>--</span>
          <span class="tl-score-sub" data-tl-parity-tag>Run Benchmark or Hill Climb</span>
        </div>
        <div class="tl-stats">
          <div class="tl-stat">
            <span class="tl-stat-k">Speed</span>
            <span class="tl-stat-v tabular" data-tl-tps>-- tok/s</span>
          </div>
          <div class="tl-stat">
            <span class="tl-stat-k">Dispatches / tok</span>
            <span class="tl-stat-v tabular" data-tl-disp>--</span>
          </div>
          <div class="tl-stat">
            <span class="tl-stat-k">Reference parity</span>
            <span class="tl-stat-v tabular" data-tl-parity>--</span>
          </div>
          <div class="tl-stat">
            <span class="tl-stat-k">Best score</span>
            <span class="tl-stat-v tabular tl-best" data-tl-best>--</span>
          </div>
        </div>
      </div>
      <div class="tl-actions">
        <button type="button" class="btn-primary tl-btn-climb" data-tl-climb>Hill Climb</button>
        <button type="button" class="btn-secondary" data-tl-bench>Benchmark Current</button>
        <button type="button" class="btn-ghost" data-tl-baseline title="Set slow unfused baseline (537 dispatches, chunked submit, wg=64)">Unfused Baseline</button>
      </div>
    </div>

    <div class="tl-grid">
      <section class="tl-panel">
        <div class="tl-panel-head">
          <span class="tl-panel-title">Real GPU execution levers</span>
          <span class="tl-panel-meta" data-tl-caps>probing GPU capabilities</span>
        </div>
        <div class="tl-levers" data-tl-levers></div>
      </section>

      <section class="tl-panel tl-path-panel">
        <div class="tl-panel-head">
          <span class="tl-panel-title">Hill-climb trajectory &amp; log</span>
          <span class="tl-panel-meta tabular" data-tl-step-meta>0 evals | parity fail scores 0</span>
        </div>
        <canvas class="tl-chart" data-tl-chart aria-label="Hill-climb score trajectory"></canvas>
        <div class="tl-insight" data-tl-insight hidden></div>
        <div class="tl-table-wrap">
          <table class="tl-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Config / Step</th>
                <th>Disp</th>
                <th>tok/s</th>
                <th>Parity</th>
                <th>Score</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody data-tl-tbody>
              <tr><td colspan="7" class="tl-empty">Click <b>Hill Climb</b> to search all levers on your GPU, or toggle levers and click <b>Benchmark Current</b>.</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>

    <div class="tl-sharebar">
      <div class="tl-share-left">
        <span class="tl-share-label">Best config share link</span>
        <code class="tl-share-url" data-tl-share-url></code>
      </div>
      <div class="tl-share-actions">
        <button type="button" class="btn-mini tl-btn-copy" data-tl-copy>Copy Share Link</button>
        <a class="btn-mini tl-agent-link" data-tl-agents-link target="_blank" rel="noopener noreferrer">Agent API (AGENTS.md)</a>
      </div>
    </div>
  `;

  const scoreBoxEl = container.querySelector<HTMLElement>("[data-tl-score-box]")!;
  const scoreEl = container.querySelector<HTMLElement>("[data-tl-score]")!;
  const parityTagEl = container.querySelector<HTMLElement>("[data-tl-parity-tag]")!;
  const tpsEl = container.querySelector<HTMLElement>("[data-tl-tps]")!;
  const dispEl = container.querySelector<HTMLElement>("[data-tl-disp]")!;
  const parityEl = container.querySelector<HTMLElement>("[data-tl-parity]")!;
  const bestEl = container.querySelector<HTMLElement>("[data-tl-best]")!;
  const capsEl = container.querySelector<HTMLElement>("[data-tl-caps]")!;
  const leversEl = container.querySelector<HTMLElement>("[data-tl-levers]")!;
  const stepMetaEl = container.querySelector<HTMLElement>("[data-tl-step-meta]")!;
  const chartCanvas = container.querySelector<HTMLCanvasElement>("[data-tl-chart]")!;
  const insightEl = container.querySelector<HTMLElement>("[data-tl-insight]")!;
  const tbodyEl = container.querySelector<HTMLElement>("[data-tl-tbody]")!;
  const shareUrlEl = container.querySelector<HTMLElement>("[data-tl-share-url]")!;
  const copyBtn = container.querySelector<HTMLButtonElement>("[data-tl-copy]")!;
  const agentsLinkEl = container.querySelector<HTMLAnchorElement>("[data-tl-agents-link]")!;
  const climbBtn = container.querySelector<HTMLButtonElement>("[data-tl-climb]")!;
  const benchBtn = container.querySelector<HTMLButtonElement>("[data-tl-bench]")!;
  const baselineBtn = container.querySelector<HTMLButtonElement>("[data-tl-baseline]")!;

  climbBtn.addEventListener("click", () => void handlers.onRunHillClimb());
  benchBtn.addEventListener("click", () => void handlers.onRunBenchmark());
  baselineBtn.addEventListener("click", () => handlers.onResetBaseline());
  copyBtn.addEventListener("click", () => void handlers.onCopyShareUrl());

  let currentState: TuningLabState | null = null;

  function drawChart(state: TuningLabState) {
    const rect = chartCanvas.parentElement?.getBoundingClientRect();
    const width = Math.max(280, Math.floor((rect?.width || 480) - 20));
    const height = 112;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (chartCanvas.width !== Math.floor(width * dpr) || chartCanvas.height !== Math.floor(height * dpr)) {
      chartCanvas.width = Math.floor(width * dpr);
      chartCanvas.height = Math.floor(height * dpr);
      chartCanvas.style.width = `${width}px`;
      chartCanvas.style.height = `${height}px`;
    }
    const ctx = chartCanvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0c0f13";
    ctx.fillRect(0, 0, width, height);

    const hist = state.history;
    if (hist.length === 0) {
      ctx.fillStyle = "#64748b";
      ctx.font = '500 11px "IBM Plex Mono", ui-monospace, monospace';
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Hill-climb score path appears here (parity failure = 0)", width / 2, height / 2);
      return;
    }

    const padL = 38;
    const padR = 14;
    const padT = 14;
    const padB = 20;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const maxScore = Math.max(10, ...hist.map((h) => Math.max(h.score, h.tokPerSec))) * 1.15;

    // Gridlines
    ctx.font = '500 9.5px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const frac of [0, 0.5, 1]) {
      const y = padT + plotH * (1 - frac);
      ctx.strokeStyle = "rgba(139, 149, 165, 0.14)";
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();
      ctx.fillStyle = "#64748b";
      ctx.fillText((maxScore * frac).toFixed(0), padL - 5, y);
    }

    const stepX = hist.length > 1 ? plotW / (hist.length - 1) : plotW / 2;

    // Best-so-far frontier curve
    ctx.beginPath();
    ctx.strokeStyle = "#14b8a6";
    ctx.lineWidth = 2;
    let runningBest = 0;
    for (let i = 0; i < hist.length; i++) {
      runningBest = Math.max(runningBest, hist[i].score);
      const x = hist.length > 1 ? padL + i * stepX : padL + plotW / 2;
      const y = padT + plotH - (runningBest / maxScore) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Candidate step points
    for (let i = 0; i < hist.length; i++) {
      const h = hist[i];
      const x = hist.length > 1 ? padL + i * stepX : padL + plotW / 2;
      const y = padT + plotH - (h.score / maxScore) * plotH;
      const color =
        !h.parityPass
          ? "#ef4444"
          : h.status === "accepted"
            ? "#2dd4bf"
            : h.status === "baseline"
              ? "#f59e0b"
              : "#64748b";
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, h.status === "accepted" || h.status === "baseline" || !h.parityPass ? 4 : 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function renderLevers(state: TuningLabState) {
    const specs = getLeverSpecs(state.adapter);
    const caps: string[] = [];
    if (state.adapter) {
      caps.push(state.adapter.hasShaderF16 ? "shader-f16: ON" : "shader-f16: n/a");
      caps.push(state.adapter.hasSubgroups ? "subgroups: ON" : "subgroups: n/a");
      caps.push(`maxWG: ${state.adapter.maxComputeInvocationsPerWorkgroup}`);
    }
    capsEl.textContent = caps.length > 0 ? caps.join(" | ") : "probing GPU";

    leversEl.innerHTML = specs
      .map((spec) => {
        const curVal = state.config[spec.key];
        const btns = spec.values
          .map((val) => {
            const supported = spec.supportedValues.includes(val);
            const active = curVal === val;
            const label =
              typeof val === "boolean"
                ? val
                  ? "ON"
                  : "OFF"
                : spec.key === "gpuTokenChain"
                  ? `${val} tok`
                  : String(val).replaceAll("_", " ");
            return `<button
              type="button"
              class="tl-pill${active ? " is-active" : ""}${!supported ? " is-unsupported" : ""}"
              data-lever-key="${spec.key}"
              data-lever-val="${String(val)}"
              ${!supported || state.busy ? "disabled" : ""}
              title="${supported ? spec.description : "Unsupported by this WebGPU adapter"}"
            >${escapeHtml(label)}</button>`;
          })
          .join("");
        return `
          <div class="tl-lever-row">
            <div class="tl-lever-info">
              <span class="tl-lever-name">${escapeHtml(spec.label)}</span>
              <span class="tl-lever-desc">${escapeHtml(spec.description)}</span>
            </div>
            <div class="tl-lever-pills">${btns}</div>
          </div>`;
      })
      .join("");

    leversEl.querySelectorAll<HTMLButtonElement>("[data-lever-key]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.leverKey as keyof TuningConfig | undefined;
        const raw = btn.dataset.leverVal;
        if (!key || raw === undefined) return;
        let parsed: boolean | number | string = raw;
        if (raw === "true") parsed = true;
        else if (raw === "false") parsed = false;
        else if (key === "qmmWorkgroupSize") parsed = Number(raw) as QmmWorkgroupSize;
        else if (key === "qmmPrecision") parsed = raw as QmmPrecision;
        else if (key === "dispatchBatching") parsed = raw as DispatchBatching;
        else if (key === "decodePacing") parsed = raw as DecodePacing;
        else if (key === "gpuTokenChain") parsed = Number(raw) as GpuTokenChain;
        handlers.onConfigChange({ [key]: parsed } as Partial<TuningConfig>);
      });
    });
  }

  function renderTable(state: TuningLabState) {
    const hist = state.history;
    stepMetaEl.textContent = `${hist.length} eval${hist.length === 1 ? "" : "s"} | parity fail scores 0`;
    if (hist.length === 0) {
      tbodyEl.innerHTML = `<tr><td colspan="7" class="tl-empty">Click <b>Hill Climb</b> to search all levers on your GPU, or toggle levers and click <b>Benchmark Current</b>.</td></tr>`;
      return;
    }

    tbodyEl.innerHTML = hist
      .map((h, idx) => {
        const isBest = state.bestEval && state.bestEval.configKey === h.configKey && h.score > 0;
        const statusLabel =
          h.status === "parity_fail"
            ? "0: FAIL"
            : h.status === "accepted"
              ? "KEEP"
              : h.status === "baseline"
                ? "START"
                : h.status === "manual"
                  ? "RUN"
                  : "DROP";
        const statusCls =
          h.status === "parity_fail"
            ? "is-fail"
            : h.status === "accepted"
              ? "is-keep"
              : h.status === "baseline"
                ? "is-base"
                : "is-drop";
        const trialNote =
          h.trialScores && h.trialScores.length > 1
            ? ` | median of ${h.trialScores.length}: [${h.trialScores.map((s) => s.toFixed(1)).join(", ")}] tok/s`
            : "";
        return `
          <tr class="tl-row${isBest ? " is-best" : ""}" data-tl-row="${idx}" title="Click to load this config${escapeHtml(trialNote)}">
            <td class="tabular">${h.step}</td>
            <td class="mono">${escapeHtml(h.label)}</td>
            <td class="tabular">${h.dispatchesPerTok}</td>
            <td class="tabular">${h.tokPerSec.toFixed(1)}</td>
            <td class="tabular ${h.parityPass ? "tl-ok" : "tl-err"}">${h.parityMatched}/${h.parityTotal}</td>
            <td class="tabular tl-score-cell ${h.parityPass ? "" : "tl-err"}">${h.score.toFixed(1)}</td>
            <td><span class="tl-tag ${statusCls}">${statusLabel}</span></td>
          </tr>`;
      })
      .join("");

    tbodyEl.querySelectorAll<HTMLTableRowElement>("[data-tl-row]").forEach((row) => {
      row.addEventListener("click", () => {
        const idx = Number(row.dataset.tlRow);
        const item = state.history[idx];
        if (item) handlers.onSelectStep(item);
      });
    });
  }

  function renderAll(state: TuningLabState) {
    const active = state.lastEval;
    scoreBoxEl.classList.toggle("is-fail", Boolean(active && !active.parityPass));
    scoreBoxEl.classList.toggle("is-pass", Boolean(active && active.parityPass));
    if (active) {
      scoreEl.textContent = active.score.toFixed(1);
      parityTagEl.textContent = active.parityPass
        ? `PARITY PASS (${active.parityMatched}/${active.parityTotal} exact)`
        : `PARITY FAIL -> SCORE 0 (${active.parityMatched}/${active.parityTotal})`;
      tpsEl.textContent = `${active.tokPerSec.toFixed(1)} tok/s`;
      dispEl.textContent = String(active.dispatchesPerTok);
      parityEl.textContent = active.parityPass
        ? `${active.parityMatched}/${active.parityTotal} PASS`
        : `${active.parityMatched}/${active.parityTotal} FAIL`;
      parityEl.classList.toggle("tl-err", !active.parityPass);
      parityEl.classList.toggle("tl-ok", active.parityPass);
    } else {
      scoreEl.textContent = "--";
      parityTagEl.textContent = "Run Benchmark or Hill Climb";
      tpsEl.textContent = "-- tok/s";
      dispEl.textContent = "--";
      parityEl.textContent = "--";
      parityEl.classList.remove("tl-err", "tl-ok");
    }

    bestEl.textContent =
      state.bestEval && state.bestEval.score > 0
        ? `${state.bestEval.score.toFixed(1)} (${state.bestEval.dispatchesPerTok} disp)`
        : "--";

    climbBtn.disabled = state.busy;
    climbBtn.textContent = state.climbing ? "Climbing..." : "Hill Climb";
    benchBtn.disabled = state.busy;
    baselineBtn.disabled = state.busy;

    shareUrlEl.textContent = state.shareUrl;
    copyBtn.textContent = state.copied ? "Copied!" : "Copy Share Link";
    agentsLinkEl.href = `${state.agentRepoUrl}/blob/main/AGENTS.md`;

    renderLevers(state);
    drawChart(state);
    const insight = summarizeTrajectoryInsight(state.history, state.bestEval);
    insightEl.hidden = !insight;
    insightEl.textContent = insight ?? "";
    renderTable(state);
  }

  return {
    update(state: TuningLabState) {
      currentState = state;
      renderAll(state);
    },
    resize() {
      if (currentState) drawChart(currentState);
    },
    destroy() {
      container.innerHTML = "";
    },
  };
}
