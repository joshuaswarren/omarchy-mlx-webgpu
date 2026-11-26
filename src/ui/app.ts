import {
  DEFAULT_OPT_FLAGS,
  type DispatchEvent,
  type FaultConfig,
  type GpuAdapterInfo,
  type KernelId,
  type ModelLoadProgress,
  type OptFlags,
  type ParityResult,
  type TokenTrace,
} from "../kernels/types";
import { PR997_BENCHMARK, PUBLIC_APPLE_SILICON_HIGHLIGHTS } from "../data/public_benchmarks";
import { QWEN3_0_6B_4BIT } from "../model/config";
import {
  buildSyntheticSchedule,
  countDispatchesForOpts,
} from "../kernels/provenance";
import { createKernelGraphView } from "./zoom_canvas";
import { createInspectorView } from "./inspector";

export type ZoomLevel = 1 | 2 | 3;

export interface AppCallbacks {
  onGenerate?: (prompt: string, opts: OptFlags, fault: FaultConfig) => void | Promise<void>;
  onStop?: () => void;
  onReset?: () => void;
  onOptChange?: (opts: OptFlags) => void;
  onFaultChange?: (fault: FaultConfig) => void;
  onRunParity?: (faultEnabled: boolean) => void | Promise<ParityResult[]>;
  onZoomChange?: (level: ZoomLevel) => void;
}

export interface AppController {
  setAdapterInfo(info: GpuAdapterInfo): void;
  setLoadProgress(progress: ModelLoadProgress): void;
  setGenerating(running: boolean): void;
  pushTokenTrace(trace: TokenTrace): void;
  clearTokens(): void;
  setParityResults(results: ParityResult[]): void;
  setOptFlags(opts: OptFlags): void;
  setFaultConfig(fault: FaultConfig): void;
  setZoom(level: ZoomLevel): void;
  setStatus(message: string, tone?: "idle" | "busy" | "ok" | "error"): void;
  setError(message: string | null): void;
  getState(): {
    zoom: ZoomLevel;
    optFlags: OptFlags;
    fault: FaultConfig;
    generating: boolean;
    selectedLayer: number;
    selectedKernel: KernelId;
  };
  destroy(): void;
}

const PROMPT_PRESETS = [
  {
    label: "Online softmax",
    prompt: "Explain in three crisp sentences why online softmax removes a GPU dispatch during LLM decoding.",
  },
  {
    label: "4-bit weights",
    prompt: "Why does 4-bit group quantization speed up single-token LLM generation on a GPU?",
  },
  {
    label: "GPU haiku",
    prompt: "Write a short haiku about compute shaders running inside a web browser.",
  },
] as const;

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatBytesMb(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

export function mountApp(root: HTMLElement, callbacks: AppCallbacks = {}): AppController {
  let zoom: ZoomLevel = 1;
  let optFlags: OptFlags = { ...DEFAULT_OPT_FLAGS };
  let fault: FaultConfig = { enabled: false };
  let generating = false;
  let adapterReady = false;
  let modelLoaded = false;
  let modelLoading = false;
  let selectedLayer = 14;
  let selectedKernel: KernelId = "sdpa_decode_online";
  let selectedDispatch: DispatchEvent | null = null;
  let tokens: TokenTrace[] = [];
  let parityResults: ParityResult[] = [];
  let latestDispatches: DispatchEvent[] = buildSyntheticSchedule(optFlags, 192);
  let latestSampleSlice: Float32Array | undefined;
  let showPr997Table = false;

  // Live A/B throughput tracker across unfused vs optimized non-fault tokens
  let abBaseSamples: number[] = [];
  let abBaseDispatches = 537;
  let abOptSamples: number[] = [];
  let abOptDispatches = 397;

  root.innerHTML = `
    <div class="shell" data-fault="false">
      <header class="topbar">
        <div class="brand">
          <h1 class="brand-title">omarchy-mlx WebGPU</h1>
          <span class="brand-desc">Qwen3-0.6B-4bit compute kernels running live on your GPU</span>
        </div>
        <div class="hw-strip">
          <span class="hw-chip" data-adapter>WebGPU adapter: probing</span>
          <span class="hw-chip">${QWEN3_0_6B_4BIT.repoId} (${QWEN3_0_6B_4BIT.numHiddenLayers}L, HD=${QWEN3_0_6B_4BIT.headDim}, Q4 g${QWEN3_0_6B_4BIT.groupSize})</span>
          <a class="hw-link" href="${PR997_BENCHMARK.repoUrl}" target="_blank" rel="noopener noreferrer">omarchy-mlx</a>
        </div>
      </header>

      <div class="banner-error" data-error-banner hidden></div>

      <div class="loadbar" data-loadbar hidden>
        <div class="loadbar-meta">
          <span data-load-msg>Loading model</span>
          <span class="tabular" data-load-pct>0%</span>
        </div>
        <div class="loadbar-track"><div class="loadbar-fill" data-load-fill></div></div>
      </div>

      <section class="workspace">
        <div class="stage">
          <div class="stage-bar">
            <nav class="zoom-tabs" role="tablist" aria-label="Zoom levels">
              <button type="button" role="tab" class="zoom-tab is-active" data-zoom="1" aria-selected="true">
                <kbd>1</kbd> Token stream
              </button>
              <button type="button" role="tab" class="zoom-tab" data-zoom="2" aria-selected="false">
                <kbd>2</kbd> Kernel graph
              </button>
              <button type="button" role="tab" class="zoom-tab" data-zoom="3" aria-selected="false">
                <kbd>3</kbd> Dispatch &amp; shader
              </button>
            </nav>

            <div class="telemetry-readout">
              <div class="metric-cell">
                <span class="metric-k">Speed</span>
                <span class="metric-v tabular" data-readout-tps>-- tok/s</span>
              </div>
              <div class="metric-cell">
                <span class="metric-k">Dispatches / tok</span>
                <span class="metric-v tabular" data-readout-disp>397</span>
              </div>
              <div class="metric-cell">
                <span class="metric-k" data-readout-ab-label>Live A/B</span>
                <span class="metric-v tabular" data-readout-ab>--</span>
              </div>
              <div class="metric-cell">
                <span class="metric-k">Attention</span>
                <span class="metric-v tabular" data-readout-attn>PR #997 1-disp</span>
              </div>
            </div>
          </div>

          <div class="viewport">
            <div class="view-pane" data-pane="1">
              <div class="stream-header">
                <span class="stream-status" data-status-pill data-tone="idle">Ready</span>
                <span class="stream-meta tabular" data-token-count>0 tokens generated</span>
              </div>
              <div class="stream-output" data-token-stream aria-live="polite">
                <div class="stream-empty">
                  Enter a prompt on the right and click <b>Generate</b> to stream tokens directly from your GPU.
                  Switch to <b>2 Kernel graph</b> or <b>3 Dispatch &amp; shader</b> anytime to inspect live dispatches.
                </div>
              </div>
              <div class="spark-panel">
                <div class="spark-head">
                  <span>Live token throughput &amp; dispatch count</span>
                  <span class="tabular" data-spark-caption>Toggle optimizations mid-stream to watch dispatches drop</span>
                </div>
                <canvas class="spark-canvas" data-spark-canvas aria-label="Live throughput waveform"></canvas>
              </div>
            </div>

            <div class="view-pane" data-pane="2" hidden></div>
            <div class="view-pane" data-pane="3" hidden></div>
          </div>
        </div>

        <aside class="bench">
          <div class="bench-section">
            <div class="bench-head">
              <h2 class="bench-title">Prompt</h2>
              <div class="preset-row">
                ${PROMPT_PRESETS.map(
                  (p, idx) =>
                    `<button type="button" class="preset-btn" data-preset="${idx}">${p.label}</button>`,
                ).join("")}
              </div>
            </div>
            <textarea
              class="prompt-input"
              data-prompt
              rows="3"
              aria-label="Generation prompt"
            >${PROMPT_PRESETS[0].prompt}</textarea>
            <div class="action-row">
              <button type="button" class="btn-primary" data-btn-generate>Load &amp; Generate (320 MB, cached)</button>
              <button type="button" class="btn-secondary" data-btn-stop disabled>Stop</button>
              <button type="button" class="btn-ghost" data-btn-reset>Clear</button>
            </div>
          </div>

          <div class="bench-section">
            <div class="bench-head">
              <h2 class="bench-title">Optimizations</h2>
              <span class="bench-badge tabular" data-opt-saved>-140/tok</span>
            </div>

            <label class="opt-row opt-row-lead">
              <input type="checkbox" data-opt="oneDispatchAttention" checked />
              <span class="opt-body">
                <span class="opt-name">
                  PR #997 1-dispatch attention
                  <a href="${PR997_BENCHMARK.prUrl}" target="_blank" rel="noopener noreferrer" class="opt-link">#997</a>
                </span>
                <span class="opt-desc">Online softmax replaces split-K p1+p2 and scratch buffer</span>
              </span>
              <span class="opt-delta tabular">-28/tok</span>
            </label>

            <label class="opt-row">
              <input type="checkbox" data-opt="fusedRopeNorm" checked />
              <span class="opt-body">
                <span class="opt-name">Fused QK-Norm + RoPE</span>
                <span class="opt-desc">Single shared-memory pass per Q and K head</span>
              </span>
              <span class="opt-delta tabular">-56/tok</span>
            </label>

            <label class="opt-row">
              <input type="checkbox" data-opt="fusedSwigluQmm" checked />
              <span class="opt-body">
                <span class="opt-name">Fused SwiGLU QMM</span>
                <span class="opt-desc">Gate + Up Q4 GEMV + SiLU in one dispatch</span>
              </span>
              <span class="opt-delta tabular">-56/tok</span>
            </label>
          </div>

          <div class="bench-section fault-section" data-fault-box>
            <div class="fault-row">
              <div class="fault-copy">
                <span class="fault-title">Fault switch</span>
                <span class="fault-desc" data-fault-desc>Corrupts online-softmax rescale mid-stream (TDR-safe)</span>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked="false"
                class="fault-switch"
                data-fault-switch
                title="Shortcut: F"
              >
                <span class="fault-thumb"></span>
                <span class="fault-label" data-fault-label>OFF</span>
              </button>
            </div>
          </div>

          <div class="bench-section parity-section">
            <div class="bench-head">
              <div class="parity-head-left">
                <h2 class="bench-title">Live CPU parity</h2>
                <span class="parity-summary tabular" data-parity-summary hidden></span>
              </div>
              <button type="button" class="btn-mini" data-btn-parity>Re-run parity</button>
            </div>
            <div class="parity-list" data-parity-list>
              <div class="parity-empty">Waiting for WebGPU parity check</div>
            </div>
          </div>
        </aside>
      </section>

      <footer class="bench-strip" aria-label="Public Apple Silicon Linux Vulkan benchmarks">
        <div class="strip-head">
          <div class="strip-title-group">
            <span class="strip-title">Public Apple Silicon Linux Vulkan benchmarks</span>
            <a class="strip-pr-link" href="${PR997_BENCHMARK.prUrl}" target="_blank" rel="noopener noreferrer">
              PR #997: +8.5% to +10.9% on M1 Max (5/5 pairs)
            </a>
          </div>
          <button type="button" class="btn-mini" data-btn-pr997>Show PR #997 table</button>
        </div>

        <div class="strip-cards">
          ${PUBLIC_APPLE_SILICON_HIGHLIGHTS.map(
            (row) => `
            <a class="mac-card" href="${row.sourceUrl}" target="_blank" rel="noopener noreferrer">
              <div class="mac-top">
                <span class="mac-name">${escapeHtml(row.machine.replace(/\s*[\u2014-]+\s*Linux Vulkan$/, ""))}</span>
                <span class="mac-tps tabular">${row.decodeTokPerSec.toFixed(row.decodeTokPerSec % 1 === 0 ? 1 : 2)} tok/s</span>
              </div>
              <div class="mac-model">${escapeHtml(row.model)}</div>
              <div class="mac-ref">${escapeHtml(row.reference)}</div>
            </a>`,
          ).join("")}
        </div>

        <div class="pr997-drawer" data-pr997-drawer hidden>
          <div class="pr997-caption">
            ${escapeHtml(PR997_BENCHMARK.hardware)} | ${escapeHtml(PR997_BENCHMARK.model)} | ${PR997_BENCHMARK.dispatchDeltaPerToken} dispatches/token
          </div>
          <table class="pr997-table">
            <thead>
              <tr>
                <th>Context</th>
                <th>Split-K baseline</th>
                <th>PR #997 online</th>
                <th>Gain</th>
                <th>Pairs</th>
                <th>macOS Metal ref</th>
                <th>Greedy SHA-256</th>
              </tr>
            </thead>
            <tbody>
              ${PR997_BENCHMARK.rows
                .map(
                  (r) => `<tr>
                    <td class="tabular">${r.context}</td>
                    <td class="tabular">${r.baseTokPerSec.toFixed(2)} tok/s (${r.baseMsPerTok.toFixed(2)} ms)</td>
                    <td class="tabular"><b>${r.onlineTokPerSec.toFixed(2)} tok/s</b> (${r.onlineMsPerTok.toFixed(2)} ms)</td>
                    <td class="tabular pr997-gain">+${r.gainPct.toFixed(2)}%</td>
                    <td class="tabular">${r.wins}</td>
                    <td class="tabular">${r.macOsRefTokPerSec ? `${r.macOsRefTokPerSec.toFixed(2)} tok/s` : "--"}</td>
                    <td class="mono">${r.greedyShaPrefix}</td>
                  </tr>`,
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </footer>
    </div>
  `;

  const shellEl = root.querySelector<HTMLElement>(".shell")!;
  const adapterEl = root.querySelector<HTMLElement>("[data-adapter]")!;
  const errorBannerEl = root.querySelector<HTMLElement>("[data-error-banner]")!;
  const loadbarEl = root.querySelector<HTMLElement>("[data-loadbar]")!;
  const loadMsgEl = root.querySelector<HTMLElement>("[data-load-msg]")!;
  const loadPctEl = root.querySelector<HTMLElement>("[data-load-pct]")!;
  const loadFillEl = root.querySelector<HTMLElement>("[data-load-fill]")!;

  const zoomTabs = Array.from(root.querySelectorAll<HTMLButtonElement>("[data-zoom]"));
  const pane1 = root.querySelector<HTMLElement>('[data-pane="1"]')!;
  const pane2 = root.querySelector<HTMLElement>('[data-pane="2"]')!;
  const pane3 = root.querySelector<HTMLElement>('[data-pane="3"]')!;

  const readoutTpsEl = root.querySelector<HTMLElement>("[data-readout-tps]")!;
  const readoutDispEl = root.querySelector<HTMLElement>("[data-readout-disp]")!;
  const readoutAbLabelEl = root.querySelector<HTMLElement>("[data-readout-ab-label]")!;
  const readoutAbEl = root.querySelector<HTMLElement>("[data-readout-ab]")!;
  const readoutAttnEl = root.querySelector<HTMLElement>("[data-readout-attn]")!;
  const paritySummaryEl = root.querySelector<HTMLElement>("[data-parity-summary]")!;

  const statusPillEl = root.querySelector<HTMLElement>("[data-status-pill]")!;
  const tokenCountEl = root.querySelector<HTMLElement>("[data-token-count]")!;
  const tokenStreamEl = root.querySelector<HTMLElement>("[data-token-stream]")!;
  const sparkCanvas = root.querySelector<HTMLCanvasElement>("[data-spark-canvas]")!;
  const sparkCaptionEl = root.querySelector<HTMLElement>("[data-spark-caption]")!;

  const promptEl = root.querySelector<HTMLTextAreaElement>("[data-prompt]")!;
  const btnGenerate = root.querySelector<HTMLButtonElement>("[data-btn-generate]")!;
  const btnStop = root.querySelector<HTMLButtonElement>("[data-btn-stop]")!;
  const btnReset = root.querySelector<HTMLButtonElement>("[data-btn-reset]")!;
  const optSavedEl = root.querySelector<HTMLElement>("[data-opt-saved]")!;
  const optInputs = Array.from(root.querySelectorAll<HTMLInputElement>("[data-opt]"));
  const faultSwitchBtn = root.querySelector<HTMLButtonElement>("[data-fault-switch]")!;
  const faultLabelEl = root.querySelector<HTMLElement>("[data-fault-label]")!;
  const faultDescEl = root.querySelector<HTMLElement>("[data-fault-desc]")!;
  const btnParity = root.querySelector<HTMLButtonElement>("[data-btn-parity]")!;
  const parityListEl = root.querySelector<HTMLElement>("[data-parity-list]")!;
  const btnPr997 = root.querySelector<HTMLButtonElement>("[data-btn-pr997]")!;
  const pr997DrawerEl = root.querySelector<HTMLElement>("[data-pr997-drawer]")!;

  const kernelGraph = createKernelGraphView(pane2, {
    onSelectLayer(layer) {
      selectedLayer = layer;
      syncSubViews();
    },
    onSelectDispatch(dispatch) {
      selectedDispatch = dispatch;
      selectedKernel = dispatch.kernel;
      setZoom(3);
    },
  });

  const inspector = createInspectorView(pane3, {
    onSelectKernel(kernel) {
      selectedKernel = kernel;
      selectedDispatch =
        latestDispatches.find((d) => d.layer === selectedLayer && d.kernel === kernel) ??
        latestDispatches.find((d) => d.kernel === kernel) ??
        null;
      syncSubViews();
    },
  });

  function medianOf(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }

  function syncGenerateButton() {
    if (modelLoading) {
      btnGenerate.textContent = "Loading model...";
      btnGenerate.disabled = true;
    } else if (generating) {
      btnGenerate.textContent = modelLoaded ? "Generating..." : "Loading & generating...";
      btnGenerate.disabled = true;
    } else {
      btnGenerate.textContent = modelLoaded ? "Generate" : "Load & Generate (320 MB, cached)";
      btnGenerate.disabled = false;
    }
    btnStop.disabled = !generating;
  }

  function recordAbSample(trace: TokenTrace) {
    if (trace.faultEnabled || !Number.isFinite(trace.tokPerSec) || trace.tokPerSec <= 0) return;
    const allOpt =
      trace.optFlags.oneDispatchAttention &&
      trace.optFlags.fusedRopeNorm &&
      trace.optFlags.fusedSwigluQmm;
    const allBase =
      !trace.optFlags.oneDispatchAttention &&
      !trace.optFlags.fusedRopeNorm &&
      !trace.optFlags.fusedSwigluQmm;
    if (allOpt) {
      abOptSamples.push(trace.tokPerSec);
      if (abOptSamples.length > 24) abOptSamples.shift();
      abOptDispatches = trace.totalDispatches;
    } else {
      // Prefer pure unfused baseline (537) when mixed with partial opt toggles
      if (allBase && abBaseDispatches !== trace.totalDispatches) {
        abBaseSamples = [];
      }
      if (allBase || abBaseDispatches !== 537 || abBaseSamples.length === 0) {
        abBaseSamples.push(trace.tokPerSec);
        if (abBaseSamples.length > 24) abBaseSamples.shift();
        abBaseDispatches = trace.totalDispatches;
      }
    }
  }

  function updateReadouts() {
    const counts = countDispatchesForOpts(optFlags);
    const lastToken = tokens[tokens.length - 1];
    const tokenMatchesOpts =
      lastToken &&
      lastToken.optFlags.oneDispatchAttention === optFlags.oneDispatchAttention &&
      lastToken.optFlags.fusedRopeNorm === optFlags.fusedRopeNorm &&
      lastToken.optFlags.fusedSwigluQmm === optFlags.fusedSwigluQmm;
    const activeDispatches = tokenMatchesOpts ? lastToken.totalDispatches : counts.totalDispatches;
    readoutDispEl.textContent = `${activeDispatches}`;
    readoutDispEl.title = `Baseline unfused: ${counts.baselineDispatches} dispatches/token`;
    readoutAttnEl.textContent = fault.enabled
      ? "FAULT ACTIVE"
      : optFlags.oneDispatchAttention
        ? "PR #997 1-disp"
        : "Split-K 2-disp";
    readoutAttnEl.classList.toggle("is-fault", fault.enabled);
    optSavedEl.textContent = counts.savedDispatches > 0 ? `-${counts.savedDispatches}/tok` : "baseline";

    if (lastToken && Number.isFinite(lastToken.tokPerSec) && lastToken.tokPerSec > 0) {
      readoutTpsEl.textContent = `${lastToken.tokPerSec.toFixed(1)} tok/s`;
    } else {
      readoutTpsEl.textContent = "-- tok/s";
    }

    if (abBaseSamples.length > 0 && abOptSamples.length > 0) {
      const baseMed = medianOf(abBaseSamples);
      const optMed = medianOf(abOptSamples);
      const pct = ((optMed - baseMed) / baseMed) * 100;
      const sign = pct >= 0 ? "+" : "";
      readoutAbLabelEl.textContent = `A/B (${abBaseDispatches} -> ${abOptDispatches})`;
      readoutAbEl.textContent = `${baseMed.toFixed(1)} -> ${optMed.toFixed(1)} (${sign}${pct.toFixed(0)}%)`;
      readoutAbEl.classList.toggle("is-gain", pct >= 0);
    } else if (abOptSamples.length > 0) {
      const optMed = medianOf(abOptSamples);
      readoutAbLabelEl.textContent = "Live A/B";
      readoutAbEl.textContent = `Opt ${optMed.toFixed(1)} | toggle for base`;
      readoutAbEl.classList.remove("is-gain");
    } else if (abBaseSamples.length > 0) {
      const baseMed = medianOf(abBaseSamples);
      readoutAbLabelEl.textContent = "Live A/B";
      readoutAbEl.textContent = `Base ${baseMed.toFixed(1)} | enable opts`;
      readoutAbEl.classList.remove("is-gain");
    } else {
      readoutAbLabelEl.textContent = "Live A/B";
      readoutAbEl.textContent = "--";
      readoutAbEl.classList.remove("is-gain");
    }
  }

  function syncOptControls() {
    for (const input of optInputs) {
      const key = input.dataset.opt as keyof OptFlags;
      input.checked = optFlags[key];
    }
    shellEl.dataset.fault = String(fault.enabled);
    faultSwitchBtn.setAttribute("aria-checked", String(fault.enabled));
    faultSwitchBtn.classList.toggle("is-on", fault.enabled);
    faultLabelEl.textContent = fault.enabled ? "ON" : "OFF";
    faultDescEl.textContent = fault.enabled
      ? "Online-softmax rescale inverted (-1.0): watch output and CPU parity fracture"
      : "Corrupts online-softmax rescale mid-stream (TDR-safe)";
  }

  function drawSparkline() {
    const rect = sparkCanvas.parentElement?.getBoundingClientRect();
    const width = Math.max(260, Math.floor((rect?.width || 640) - 24));
    const height = 74;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (sparkCanvas.width !== Math.floor(width * dpr) || sparkCanvas.height !== Math.floor(height * dpr)) {
      sparkCanvas.width = Math.floor(width * dpr);
      sparkCanvas.height = Math.floor(height * dpr);
      sparkCanvas.style.width = `${width}px`;
      sparkCanvas.style.height = `${height}px`;
    }
    const ctx = sparkCanvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0c0f13";
    ctx.fillRect(0, 0, width, height);

    if (tokens.length === 0) {
      ctx.fillStyle = "#64748b";
      ctx.font = '500 11px "IBM Plex Mono", ui-monospace, monospace';
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const counts = countDispatchesForOpts(optFlags);
      ctx.fillText(
        `Armed at ${counts.totalDispatches} dispatches/tok (${counts.savedDispatches} fused out vs ${counts.baselineDispatches} baseline)`,
        width / 2,
        height / 2,
      );
      return;
    }

    const recent = tokens.slice(-60);
    const maxTps = Math.max(10, ...recent.map((t) => t.tokPerSec)) * 1.15;
    const padX = 8;
    const padY = 10;
    const plotW = width - padX * 2;
    const plotH = height - padY * 2;
    const step = recent.length > 1 ? plotW / (recent.length - 1) : plotW;

    // Background bars for dispatch count (397..537)
    const barW = Math.max(3, Math.floor(plotW / Math.max(12, recent.length)) - 2);
    for (let i = 0; i < recent.length; i++) {
      const t = recent[i];
      const cx = padX + i * step;
      const dh = Math.min(1, t.totalDispatches / 537) * plotH * 0.55;
      ctx.fillStyle = t.faultEnabled
        ? "rgba(239, 68, 68, 0.35)"
        : t.optFlags.oneDispatchAttention
          ? "rgba(20, 184, 166, 0.22)"
          : "rgba(59, 130, 246, 0.22)";
      ctx.fillRect(cx - barW / 2, padY + plotH - dh, barW, dh);
    }

    // Throughput line
    ctx.beginPath();
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    for (let i = 0; i < recent.length; i++) {
      const x = padX + i * step;
      const y = padY + plotH - (recent[i].tokPerSec / maxTps) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fault markers
    for (let i = 0; i < recent.length; i++) {
      if (!recent[i].faultEnabled) continue;
      const x = padX + i * step;
      const y = padY + plotH - (recent[i].tokPerSec / maxTps) * plotH;
      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    const last = recent[recent.length - 1];
    sparkCaptionEl.textContent = `step ${last.step} | ${last.tokPerSec.toFixed(1)} tok/s | ${last.totalDispatches} dispatches/tok`;
  }

  function renderTokens() {
    tokenCountEl.textContent = `${tokens.length} token${tokens.length === 1 ? "" : "s"} generated`;
    if (tokens.length === 0) {
      tokenStreamEl.innerHTML = `<div class="stream-empty">
        Enter a prompt on the right and click <b>Generate</b> to stream tokens directly from your GPU.
        Switch to <b>2 Kernel graph</b> or <b>3 Dispatch &amp; shader</b> anytime to inspect live dispatches.
      </div>`;
      drawSparkline();
      return;
    }

    const html = tokens
      .map((t, idx) => {
        const cls = t.faultEnabled ? "tok-chip tok-chip-fault" : "tok-chip";
        const title = `Step ${t.step}: ${t.tokPerSec.toFixed(1)} tok/s, ${t.totalDispatches} dispatches${t.faultEnabled ? " [FAULT]" : ""}`;
        return `<span class="${cls}" data-token-idx="${idx}" title="${escapeHtml(title)}">${escapeHtml(t.text)}</span>`;
      })
      .join("");
    tokenStreamEl.innerHTML = `<div class="stream-text">${html}${generating ? '<span class="stream-cursor"></span>' : ""}</div>`;
    tokenStreamEl.scrollTop = tokenStreamEl.scrollHeight;
    drawSparkline();
  }

  function renderParity() {
    if (parityResults.length === 0) {
      paritySummaryEl.hidden = true;
      parityListEl.innerHTML = `<div class="parity-empty">Waiting for WebGPU parity check</div>`;
      return;
    }

    const failedCount = parityResults.filter((r) => !r.passed).length;
    const passedCount = parityResults.length - failedCount;
    paritySummaryEl.hidden = false;
    paritySummaryEl.className = `parity-summary tabular ${failedCount > 0 ? "is-fail" : "is-pass"}`;
    paritySummaryEl.textContent =
      failedCount > 0
        ? `${failedCount} FAULT${failedCount === 1 ? "" : "S"} CAUGHT`
        : `${passedCount}/${parityResults.length} PASS`;

    // Surface failed/faulted kernels at the top so Fault Switch impact is immediate
    const ordered = parityResults
      .map((r, origIdx) => ({ r, origIdx }))
      .sort((a, b) => Number(a.r.passed) - Number(b.r.passed));

    parityListEl.innerHTML = ordered
      .map(({ r, origIdx }) => {
        const stateCls = r.passed ? "is-pass" : "is-fail";
        const badge = r.passed ? "PASS" : r.faultInjected ? "FAULT CAUGHT" : "FAIL";
        const maxBar = Math.max(1e-7, ...r.diffSample.map((v) => Math.abs(v)));
        const bars = r.diffSample
          .slice(0, 20)
          .map((v) => {
            const pct = Math.max(8, Math.min(100, Math.round((Math.abs(v) / maxBar) * 100)));
            return `<i style="height:${pct}%"></i>`;
          })
          .join("");

        return `
          <button type="button" class="parity-row ${stateCls}" data-parity-idx="${origIdx}">
            <div class="parity-row-top">
              <span class="parity-name">${escapeHtml(r.label)}</span>
              <span class="parity-badge ${stateCls}">${badge}</span>
            </div>
            <div class="parity-row-metrics tabular">
              <span>max ${r.maxAbsDiff.toExponential(1)}</span>
              <span>cos ${r.cosineSimilarity.toFixed(5)}</span>
              <span>${r.gpuMs.toFixed(1)} ms</span>
              <span class="parity-mini-bars ${stateCls}" aria-hidden="true">${bars}</span>
            </div>
          </button>`;
      })
      .join("");

    parityListEl.querySelectorAll<HTMLButtonElement>("[data-parity-idx]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.dataset.parityIdx);
        const item = parityResults[idx];
        if (!item) return;
        selectedKernel = item.kernel;
        selectedDispatch =
          latestDispatches.find((d) => d.layer === selectedLayer && d.kernel === item.kernel) ??
          latestDispatches.find((d) => d.kernel === item.kernel) ??
          null;
        setZoom(3);
      });
    });
  }

  function syncSubViews() {
    updateReadouts();
    if (zoom === 2) {
      kernelGraph.update({
        dispatches: latestDispatches,
        optFlags,
        faultEnabled: fault.enabled,
        selectedLayer,
        selectedDispatchIndex: selectedDispatch?.index ?? null,
      });
    } else if (zoom === 3) {
      inspector.update({
        kernel: selectedKernel,
        dispatch: selectedDispatch,
        faultEnabled: fault.enabled,
        parityResults,
        sampleSlice: latestSampleSlice,
      });
    } else {
      drawSparkline();
    }
  }

  function setZoom(level: ZoomLevel) {
    zoom = level;
    zoomTabs.forEach((tab) => {
      const active = Number(tab.dataset.zoom) === level;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    pane1.hidden = level !== 1;
    pane2.hidden = level !== 2;
    pane3.hidden = level !== 3;
    syncSubViews();
    callbacks.onZoomChange?.(level);
  }

  async function triggerParityRun(faultEnabled: boolean) {
    if (!callbacks.onRunParity) return;
    btnParity.disabled = true;
    btnParity.textContent = "Checking...";
    try {
      const res = await callbacks.onRunParity(faultEnabled);
      if (Array.isArray(res)) {
        parityResults = res;
        renderParity();
        syncSubViews();
      }
    } finally {
      btnParity.disabled = false;
      btnParity.textContent = "Re-run parity";
    }
  }

  function toggleFault() {
    fault = { enabled: !fault.enabled };
    syncOptControls();
    callbacks.onFaultChange?.({ ...fault });
    syncSubViews();
    if (!callbacks.onFaultChange) {
      void triggerParityRun(fault.enabled);
    }
  }

  // Wire controls
  zoomTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const lvl = Number(tab.dataset.zoom) as ZoomLevel;
      if (lvl === 1 || lvl === 2 || lvl === 3) setZoom(lvl);
    });
  });

  root.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.preset);
      const preset = PROMPT_PRESETS[idx];
      if (preset) {
        promptEl.value = preset.prompt;
        promptEl.focus();
      }
    });
  });

  btnGenerate.addEventListener("click", () => {
    const prompt = promptEl.value.trim() || PROMPT_PRESETS[0].prompt;
    callbacks.onGenerate?.(prompt, { ...optFlags }, { ...fault });
  });

  btnStop.addEventListener("click", () => {
    callbacks.onStop?.();
  });

  btnReset.addEventListener("click", () => {
    tokens = [];
    latestSampleSlice = undefined;
    renderTokens();
    updateReadouts();
    callbacks.onReset?.();
  });

  optInputs.forEach((input) => {
    input.addEventListener("change", () => {
      const key = input.dataset.opt as keyof OptFlags;
      optFlags = { ...optFlags, [key]: input.checked };
      // Immediately update dispatch schedule so Level 2 shrinks/expands on the spot
      latestDispatches = buildSyntheticSchedule(optFlags, Math.max(192, tokens.length + 64));
      if (key === "oneDispatchAttention") {
        selectedKernel = optFlags.oneDispatchAttention ? "sdpa_decode_online" : "sdpa_decode_split_p1";
        selectedDispatch =
          latestDispatches.find((d) => d.layer === selectedLayer && d.kernel === selectedKernel) ?? null;
      }
      callbacks.onOptChange?.({ ...optFlags });
      syncSubViews();
    });
  });

  faultSwitchBtn.addEventListener("click", toggleFault);

  btnParity.addEventListener("click", () => {
    void triggerParityRun(fault.enabled);
  });

  btnPr997.addEventListener("click", () => {
    showPr997Table = !showPr997Table;
    pr997DrawerEl.hidden = !showPr997Table;
    btnPr997.textContent = showPr997Table ? "Hide PR #997 table" : "Show PR #997 table";
  });

  tokenStreamEl.addEventListener("click", (ev) => {
    const target = (ev.target as HTMLElement | null)?.closest<HTMLElement>("[data-token-idx]");
    if (!target) return;
    const idx = Number(target.dataset.tokenIdx);
    const tr = tokens[idx];
    if (tr && tr.dispatches.length > 0) {
      latestDispatches = tr.dispatches;
      latestSampleSlice = tr.sampleSlice;
      setZoom(2);
    }
  });

  const onKeyDown = (ev: KeyboardEvent) => {
    const tag = (ev.target as HTMLElement | null)?.tagName;
    if (tag === "TEXTAREA" || tag === "INPUT" || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (ev.key === "1") setZoom(1);
    else if (ev.key === "2") setZoom(2);
    else if (ev.key === "3") setZoom(3);
    else if (ev.key === "f" || ev.key === "F") toggleFault();
  };
  window.addEventListener("keydown", onKeyDown);

  const onResize = () => {
    if (zoom === 1) drawSparkline();
    else if (zoom === 2) kernelGraph.resize();
    else inspector.resize();
  };
  window.addEventListener("resize", onResize);

  // Initial sync
  selectedDispatch =
    latestDispatches.find((d) => d.layer === selectedLayer && d.kernel === selectedKernel) ?? null;
  syncOptControls();
  updateReadouts();
  drawSparkline();

  return {
    setAdapterInfo(info: GpuAdapterInfo) {
      adapterReady = true;
      adapterEl.classList.remove("is-error");
      const label = [info.vendor, info.architecture || info.device]
        .filter((s) => s && s !== "unknown")
        .join(" / ");
      adapterEl.textContent = `WebGPU: ${label || info.description || "active"}`;
    },
    setLoadProgress(progress: ModelLoadProgress) {
      if (progress.stage === "ready") {
        modelLoaded = true;
        modelLoading = false;
        loadbarEl.hidden = true;
        statusPillEl.textContent = progress.fromCache ? "Model ready (cached)" : "Model ready";
        statusPillEl.dataset.tone = "ok";
        syncGenerateButton();
        return;
      }
      modelLoading = true;
      loadbarEl.hidden = false;
      const pct =
        progress.totalBytes > 0
          ? Math.min(100, Math.round((progress.loadedBytes / progress.totalBytes) * 100))
          : 0;
      loadMsgEl.textContent = progress.message;
      loadPctEl.textContent =
        progress.totalBytes > 0
          ? `${pct}% (${formatBytesMb(progress.loadedBytes)} / ${formatBytesMb(progress.totalBytes)})`
          : `${pct}%`;
      loadFillEl.style.width = `${pct}%`;
      statusPillEl.textContent = progress.message;
      statusPillEl.dataset.tone = "busy";
      syncGenerateButton();
    },
    setGenerating(running: boolean) {
      generating = running;
      if (!running) modelLoading = false;
      syncGenerateButton();
      statusPillEl.textContent = running ? "Generating on GPU" : tokens.length > 0 ? "Complete" : "Ready";
      statusPillEl.dataset.tone = running ? "busy" : tokens.length > 0 ? "ok" : "idle";
      renderTokens();
    },
    pushTokenTrace(trace: TokenTrace) {
      if (!modelLoaded) {
        modelLoaded = true;
        modelLoading = false;
        syncGenerateButton();
      }
      tokens.push(trace);
      recordAbSample(trace);
      if (trace.dispatches.length > 0) {
        latestDispatches = trace.dispatches;
      }
      if (trace.sampleSlice) {
        latestSampleSlice = trace.sampleSlice;
      }
      renderTokens();
      syncSubViews();
    },
    clearTokens() {
      tokens = [];
      latestSampleSlice = undefined;
      renderTokens();
      updateReadouts();
    },
    setParityResults(results: ParityResult[]) {
      parityResults = results;
      renderParity();
      syncSubViews();
    },
    setOptFlags(next: OptFlags) {
      optFlags = { ...next };
      latestDispatches = buildSyntheticSchedule(optFlags, Math.max(192, tokens.length + 64));
      syncOptControls();
      syncSubViews();
    },
    setFaultConfig(next: FaultConfig) {
      fault = { ...next };
      syncOptControls();
      syncSubViews();
    },
    setZoom,
    setStatus(message: string, tone: "idle" | "busy" | "ok" | "error" = "idle") {
      statusPillEl.textContent = message;
      statusPillEl.dataset.tone = tone;
    },
    setError(message: string | null) {
      if (!message) {
        errorBannerEl.hidden = true;
        errorBannerEl.textContent = "";
      } else {
        errorBannerEl.hidden = false;
        errorBannerEl.textContent = message;
        statusPillEl.textContent = "Error";
        statusPillEl.dataset.tone = "error";
        if (!adapterReady) {
          adapterEl.textContent = "No WebGPU";
          adapterEl.classList.add("is-error");
          if (parityResults.length === 0) {
            parityListEl.innerHTML = `<div class="parity-empty">WebGPU unavailable in this browser</div>`;
          }
        }
      }
    },
    getState() {
      return {
        zoom,
        optFlags: { ...optFlags },
        fault: { ...fault },
        generating,
        selectedLayer,
        selectedKernel,
      };
    },
    destroy() {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", onResize);
      kernelGraph.destroy();
      inspector.destroy();
    },
  };
}
