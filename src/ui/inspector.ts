import type { DispatchEvent, KernelId, ParityResult } from "../kernels/types";
import { KERNEL_PROVENANCE } from "../kernels/provenance";

export interface InspectorHandlers {
  onSelectKernel: (kernel: KernelId) => void;
}

export interface InspectorState {
  kernel: KernelId;
  dispatch: DispatchEvent | null;
  faultEnabled: boolean;
  parityResults: ParityResult[];
  sampleSlice?: Float32Array;
}

export interface InspectorController {
  update(state: InspectorState): void;
  resize(): void;
  destroy(): void;
}

const INSPECTABLE_KERNELS: KernelId[] = [
  "sdpa_decode_online",
  "sdpa_decode_split_p1",
  "sdpa_decode_split_p2",
  "fast_norm",
  "fast_rope_norm",
  "qmm_vec",
  "qmm_vec_swiglu",
];

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function highlightShader(code: string, faultEnabled: boolean): string {
  const lines = code.split("\n");
  return lines
    .map((line) => {
      const escaped = escapeHtml(line);
      const isFaultLine =
        faultEnabled &&
        (line.includes("fault_mode") || line.includes("FAULT SWITCH") || line.includes("Fault mode"));
      let formatted = escaped;
      if (/^\s*\/\//.test(line)) {
        formatted = `<span class="tok-comment">${escaped}</span>`;
      } else {
        formatted = escaped
          .replace(
            /\b(fn|let|var|const|for|if|else|loop|break|return|struct|layout|uniform|readonly|writeonly|buffer|shared|void|precise|in)\b/g,
            '<span class="tok-kw">$1</span>',
          )
          .replace(
            /\b(f32|u32|i32|vec2|vec3|vec4|uvec2|uvec4|float|uint|uint16_t|bool|array)\b/g,
            '<span class="tok-ty">$1</span>',
          )
          .replace(
            /\b(workgroupBarrier|barrier|subgroupAdd|subgroupMax|subgroupShuffle|inverseSqrt|inversesqrt|exp2|fma|select|max|min)\b/g,
            '<span class="tok-fn">$1</span>',
          );
      }
      return isFaultLine
        ? `<span class="code-line code-line-fault">${formatted}</span>`
        : `<span class="code-line">${formatted}</span>`;
    })
    .join("");
}

function formatBytes(bytes: number): string {
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(2)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

/** Deterministic fallback sample when parity has not yet run on a specific kernel. */
function fallbackValues(kernel: KernelId, faultEnabled: boolean, mode: "diff" | "act"): number[] {
  const out: number[] = [];
  let seed = 0;
  for (let i = 0; i < kernel.length; i++) seed = (seed * 31 + kernel.charCodeAt(i)) >>> 0;
  for (let i = 0; i < 64; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const u = ((seed >>> 8) / 0xffffff) * 2 - 1;
    if (mode === "act") {
      out.push(faultEnabled ? u * 2.4 * (i % 2 === 0 ? 1 : -1.35) : u * 0.65);
    } else {
      out.push(faultEnabled ? (i % 2 === 0 ? 0.42 : -0.68) * (0.6 + Math.abs(u)) : u * 2.4e-6);
    }
  }
  return out;
}

export function createInspectorView(
  container: HTMLElement,
  handlers: InspectorHandlers,
): InspectorController {
  container.innerHTML = "";
  container.classList.add("ins-root");

  const pickerBar = document.createElement("div");
  pickerBar.className = "ins-picker";

  const topGrid = document.createElement("div");
  topGrid.className = "ins-top-grid";
  topGrid.innerHTML = `
    <section class="ins-panel">
      <div class="ins-panel-head">
        <span class="ins-panel-title">Workgroup grid</span>
        <span class="ins-panel-meta" data-ins-wg-meta></span>
      </div>
      <canvas class="ins-canvas" data-ins-wg-canvas aria-label="Workgroup grid layout"></canvas>
      <div class="ins-metrics" data-ins-metrics></div>
    </section>
    <section class="ins-panel">
      <div class="ins-panel-head">
        <span class="ins-panel-title" data-ins-heat-title>CPU diff heatmap (64 dims)</span>
        <div class="ins-heat-modes">
          <button type="button" class="ins-mode-btn is-active" data-heat-mode="diff">GPU - CPU diff</button>
          <button type="button" class="ins-mode-btn" data-heat-mode="act">Activations</button>
        </div>
      </div>
      <div class="ins-heat-wrap">
        <canvas class="ins-canvas" data-ins-heat-canvas aria-label="Activation and parity diff heatmap"></canvas>
        <div class="ins-heat-tip" data-ins-heat-tip hidden></div>
      </div>
      <div class="ins-heat-legend" data-ins-heat-legend></div>
    </section>
  `;

  const codeGrid = document.createElement("div");
  codeGrid.className = "ins-code-grid";
  codeGrid.innerHTML = `
    <section class="ins-code-panel">
      <div class="ins-code-head">
        <span class="ins-code-label">omarchy-mlx Vulkan GLSL</span>
        <a class="ins-source-link" data-ins-glsl-link target="_blank" rel="noopener noreferrer"></a>
      </div>
      <pre class="ins-code"><code data-ins-glsl></code></pre>
    </section>
    <section class="ins-code-panel">
      <div class="ins-code-head">
        <span class="ins-code-label">Browser WebGPU WGSL</span>
        <span class="ins-wgsl-path" data-ins-wgsl-path></span>
      </div>
      <pre class="ins-code"><code data-ins-wgsl></code></pre>
    </section>
  `;

  const noteBar = document.createElement("div");
  noteBar.className = "ins-note";

  container.append(pickerBar, topGrid, codeGrid, noteBar);

  const wgMetaEl = topGrid.querySelector<HTMLElement>("[data-ins-wg-meta]")!;
  const wgCanvas = topGrid.querySelector<HTMLCanvasElement>("[data-ins-wg-canvas]")!;
  const metricsEl = topGrid.querySelector<HTMLElement>("[data-ins-metrics]")!;
  const heatTitleEl = topGrid.querySelector<HTMLElement>("[data-ins-heat-title]")!;
  const heatCanvas = topGrid.querySelector<HTMLCanvasElement>("[data-ins-heat-canvas]")!;
  const heatTipEl = topGrid.querySelector<HTMLElement>("[data-ins-heat-tip]")!;
  const heatLegendEl = topGrid.querySelector<HTMLElement>("[data-ins-heat-legend]")!;
  const modeBtns = Array.from(topGrid.querySelectorAll<HTMLButtonElement>("[data-heat-mode]"));
  const glslLinkEl = codeGrid.querySelector<HTMLAnchorElement>("[data-ins-glsl-link]")!;
  const glslCodeEl = codeGrid.querySelector<HTMLElement>("[data-ins-glsl]")!;
  const wgslPathEl = codeGrid.querySelector<HTMLElement>("[data-ins-wgsl-path]")!;
  const wgslCodeEl = codeGrid.querySelector<HTMLElement>("[data-ins-wgsl]")!;

  let currentState: InspectorState | null = null;
  let heatMode: "diff" | "act" = "diff";
  let lastHeatValues: number[] = [];
  let lastHeatParity: ParityResult | null = null;

  function setupCanvas(canvas: HTMLCanvasElement, cssHeight: number) {
    const rect = canvas.parentElement?.getBoundingClientRect();
    const width = Math.max(220, Math.floor((rect?.width || 360) - 2));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(cssHeight * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(cssHeight * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${cssHeight}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height: cssHeight };
  }

  function findParity(state: InspectorState): ParityResult | null {
    const direct = state.parityResults.find((r) => r.kernel === state.kernel);
    if (direct) return direct;
    if (state.kernel === "sdpa_decode_split_p2") {
      return state.parityResults.find((r) => r.kernel === "sdpa_decode_split_p1") ?? null;
    }
    return state.parityResults[0] ?? null;
  }

  function drawWorkgroupGrid(state: InspectorState) {
    const setup = setupCanvas(wgCanvas, 124);
    if (!setup) return;
    const { ctx, width, height } = setup;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0c0f13";
    ctx.fillRect(0, 0, width, height);

    const prov = KERNEL_PROVENANCE[state.kernel];
    const wg: [number, number, number] = state.dispatch?.workgroups ??
      (state.kernel === "sdpa_decode_online"
        ? [16, 1, 1]
        : state.kernel === "sdpa_decode_split_p1"
          ? [16, 3, 1]
          : [16, 1, 1]);
    const wgSize: [number, number, number] = state.dispatch?.workgroupSize ?? prov.workgroupSize;

    wgMetaEl.textContent = `grid (${wg[0]}, ${wg[1]}, ${wg[2]}) | wg (${wgSize[0]}, ${wgSize[1]}, ${wgSize[2]})`;

    // Left half: workgroup dispatch grid (capped at 16x4 visual cells with +N label)
    const leftW = Math.floor(width * 0.52);
    const visCols = Math.min(16, Math.max(1, wg[0]));
    const visRows = Math.min(4, Math.max(1, wg[1]));
    const cellGap = 3;
    const gridPad = 12;
    const cellW = Math.max(6, Math.floor((leftW - gridPad * 2 - cellGap * (visCols - 1)) / visCols));
    const cellH = Math.max(10, Math.min(18, Math.floor((height - 46 - cellGap * (visRows - 1)) / visRows)));

    ctx.fillStyle = "#8b95a5";
    ctx.font = '500 10px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    const totalWg = wg[0] * wg[1] * wg[2];
    ctx.fillText(`Workgroups: ${totalWg}`, gridPad, 8);

    const isFaultTarget =
      state.faultEnabled &&
      (state.kernel === "sdpa_decode_online" ||
        state.kernel === "sdpa_decode_split_p1" ||
        state.kernel === "fast_norm");
    const activeColor = isFaultTarget
      ? "#ef4444"
      : prov.family === "attention"
        ? "#f59e0b"
        : prov.family === "qmm"
          ? "#3b82f6"
          : prov.family === "swiglu"
            ? "#a855f7"
            : "#14b8a6";

    for (let r = 0; r < visRows; r++) {
      for (let c = 0; c < visCols; c++) {
        const cx = gridPad + c * (cellW + cellGap);
        const cy = 26 + r * (cellH + cellGap);
        ctx.fillStyle = activeColor;
        ctx.globalAlpha = r === 0 && c === 0 ? 1.0 : 0.72;
        ctx.fillRect(cx, cy, cellW, cellH);
        if (r === 0 && c === 0) {
          ctx.globalAlpha = 1.0;
          ctx.strokeStyle = "#f8fafc";
          ctx.lineWidth = 1.25;
          ctx.strokeRect(cx + 0.5, cy + 0.5, cellW - 1, cellH - 1);
        }
      }
    }
    ctx.globalAlpha = 1.0;
    if (wg[0] > 16) {
      ctx.fillStyle = "#8b95a5";
      ctx.fillText(`+${wg[0] - 16} more along X`, gridPad, 26 + visRows * (cellH + cellGap) + 4);
    }

    // Divider
    ctx.strokeStyle = "rgba(139, 149, 165, 0.18)";
    ctx.beginPath();
    ctx.moveTo(leftW, 10);
    ctx.lineTo(leftW, height - 10);
    ctx.stroke();

    // Right half: inside WG[0,0,0] thread/subgroup topology
    const rx = leftW + 12;
    const rw = width - rx - 10;
    const threads = wgSize[0] * wgSize[1] * wgSize[2];
    const subgroups = Math.max(1, Math.min(8, Math.floor(threads / 32)));
    ctx.fillStyle = "#8b95a5";
    ctx.fillText(`Inside WG[0]: ${threads} threads (${subgroups}x32)`, rx, 8);

    const rowH = Math.max(6, Math.floor((height - 46) / subgroups) - 2);
    for (let sg = 0; sg < subgroups; sg++) {
      const sy = 26 + sg * (rowH + 2);
      ctx.fillStyle = isFaultTarget && sg % 2 === 1 ? "rgba(239, 68, 68, 0.55)" : "rgba(59, 130, 246, 0.32)";
      ctx.fillRect(rx, sy, rw, rowH);
      // 32 lane ticks inside each group
      ctx.fillStyle = isFaultTarget && sg % 2 === 1 ? "#ef4444" : activeColor;
      const laneW = Math.max(1, rw / 32);
      for (let l = 0; l < 32; l += 2) {
        ctx.fillRect(rx + l * laneW, sy, Math.max(1, laneW - 0.5), rowH);
      }
    }

    ctx.fillStyle = isFaultTarget ? "#fca5a5" : "#94a3b8";
    ctx.font = '500 9.5px "IBM Plex Mono", ui-monospace, monospace';
    ctx.fillText(
      isFaultTarget
        ? "Fault: rescale weight = -1.0"
        : state.kernel === "sdpa_decode_online"
          ? "Zero scratch, 1 barrier merge"
          : "Coalesced workgroup reduction",
      rx,
      height - 15,
    );

    const readStr = state.dispatch ? formatBytes(state.dispatch.bytesRead) : "model tensor";
    const writeStr = state.dispatch ? formatBytes(state.dispatch.bytesWritten) : "output tensor";
    metricsEl.innerHTML = `
      <span><b>Kernel</b> ${prov.title}</span>
      <span><b>Traffic</b> ${readStr} read / ${writeStr} write</span>
    `;
  }

  function drawHeatmap(state: InspectorState) {
    const setup = setupCanvas(heatCanvas, 124);
    if (!setup) return;
    const { ctx, width, height } = setup;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0c0f13";
    ctx.fillRect(0, 0, width, height);

    const parity = findParity(state);
    lastHeatParity = parity;

    let values: number[];
    let sourceTag: string;
    if (heatMode === "diff") {
      if (parity && parity.diffSample.length > 0) {
        values = parity.diffSample.slice(0, 64);
        sourceTag = `${parity.label} (max ${parity.maxAbsDiff.toExponential(2)})`;
      } else {
        values = fallbackValues(state.kernel, state.faultEnabled, "diff");
        sourceTag = state.faultEnabled ? "Fault mode active" : "Run CPU parity to populate live diff";
      }
    } else {
      if (state.sampleSlice && state.sampleSlice.length >= 16) {
        values = Array.from(state.sampleSlice.slice(0, 64));
        sourceTag = "Live token activation slice";
      } else if (parity && parity.gpuSample.length > 0) {
        values = parity.gpuSample.slice(0, 64);
        sourceTag = `${parity.label} GPU output`;
      } else {
        values = fallbackValues(state.kernel, state.faultEnabled, "act");
        sourceTag = "Kernel output preview";
      }
    }
    lastHeatValues = values;

    heatTitleEl.textContent =
      heatMode === "diff" ? "WebGPU vs CPU diff heatmap (64 dims)" : "Output activation heatmap (64 dims)";

    const cols = 16;
    const rows = 4;
    const padX = 10;
    const padY = 10;
    const gap = 3;
    const cellW = (width - padX * 2 - gap * (cols - 1)) / cols;
    const cellH = (height - padY * 2 - gap * (rows - 1)) / rows;

    const maxMag = Math.max(1e-7, ...values.map((v) => Math.abs(v)));
    const isFaultScale = heatMode === "diff" && (maxMag > 1e-3 || state.faultEnabled);

    for (let i = 0; i < Math.min(64, values.length); i++) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      const x = padX + c * (cellW + gap);
      const y = padY + r * (cellH + gap);
      const v = values[i];

      if (heatMode === "diff") {
        const mag = Math.abs(v);
        if (isFaultScale) {
          const norm = Math.min(1, mag / Math.max(0.25, maxMag));
          const red = Math.round(70 + norm * 180);
          const green = Math.round(28 + (1 - norm) * 65);
          const blue = Math.round(32 + (1 - norm) * 40);
          ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        } else {
          // Clean parity: tiny 1e-6 floating-point noise in calm slate-teal
          const norm = Math.min(1, mag / 1e-4);
          const rC = Math.round(18 + norm * 22);
          const gC = Math.round(48 + norm * 115);
          const bC = Math.round(58 + norm * 96);
          ctx.fillStyle = `rgb(${rC}, ${gC}, ${bC})`;
        }
      } else {
        const norm = Math.max(-1, Math.min(1, v / maxMag));
        if (norm >= 0) {
          const rC = Math.round(22 + norm * 225);
          const gC = Math.round(35 + norm * 125);
          const bC = Math.round(52 - norm * 35);
          ctx.fillStyle = `rgb(${rC}, ${gC}, ${bC})`;
        } else {
          const a = -norm;
          const rC = Math.round(22 + a * 30);
          const gC = Math.round(45 + a * 95);
          const bC = Math.round(68 + a * 175);
          ctx.fillStyle = `rgb(${rC}, ${gC}, ${bC})`;
        }
      }

      ctx.fillRect(x, y, cellW, cellH);
    }

    heatLegendEl.innerHTML = `
      <span class="${isFaultScale ? "ins-legend-fault" : ""}">${sourceTag}</span>
      <span>peak ${maxMag.toExponential(2)}</span>
    `;
  }

  function renderCodeAndPicker(state: InspectorState) {
    pickerBar.innerHTML = INSPECTABLE_KERNELS.map((id) => {
      const p = KERNEL_PROVENANCE[id];
      const active = id === state.kernel ? "is-active" : "";
      const badge = p.prNumber ? ` PR #${p.prNumber}` : "";
      return `<button type="button" class="ins-pill ${active}" data-ins-kernel="${id}">${p.shortLabel}${badge}</button>`;
    }).join("");

    pickerBar.querySelectorAll<HTMLButtonElement>("[data-ins-kernel]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const k = btn.dataset.insKernel as KernelId | undefined;
        if (k) handlers.onSelectKernel(k);
      });
    });

    const prov = KERNEL_PROVENANCE[state.kernel];
    glslLinkEl.href = prov.omarchyUrl;
    glslLinkEl.textContent = prov.omarchyPath.split("/").slice(-2).join("/");
    glslCodeEl.innerHTML = highlightShader(prov.glslSnippet, state.faultEnabled);
    wgslPathEl.textContent = prov.wgslPath;
    wgslCodeEl.innerHTML = highlightShader(prov.wgslSnippet, state.faultEnabled);
    noteBar.textContent = `${prov.summary} ${prov.portNotes}`;
  }

  function renderAll() {
    if (!currentState) return;
    renderCodeAndPicker(currentState);
    drawWorkgroupGrid(currentState);
    drawHeatmap(currentState);
  }

  modeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const m = btn.dataset.heatMode === "act" ? "act" : "diff";
      heatMode = m;
      modeBtns.forEach((b) => b.classList.toggle("is-active", b === btn));
      if (currentState) drawHeatmap(currentState);
    });
  });

  const onHeatMove = (ev: MouseEvent) => {
    if (lastHeatValues.length === 0) return;
    const rect = heatCanvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left - 10;
    const my = ev.clientY - rect.top - 10;
    const cw = (rect.width - 20) / 16;
    const ch = (rect.height - 20) / 4;
    const c = Math.floor(mx / cw);
    const r = Math.floor(my / ch);
    if (c < 0 || c >= 16 || r < 0 || r >= 4) {
      heatTipEl.hidden = true;
      return;
    }
    const idx = r * 16 + c;
    if (idx >= lastHeatValues.length) {
      heatTipEl.hidden = true;
      return;
    }
    const val = lastHeatValues[idx];
    const p = lastHeatParity;
    if (heatMode === "diff" && p && idx < p.gpuSample.length && idx < p.cpuSample.length) {
      heatTipEl.textContent = `[${idx}] gpu=${p.gpuSample[idx].toFixed(5)} cpu=${p.cpuSample[idx].toFixed(5)} diff=${val.toExponential(2)}`;
    } else {
      heatTipEl.textContent = `[${idx}] value=${val.toExponential(3)}`;
    }
    heatTipEl.style.left = `${Math.min(rect.width - 240, Math.max(8, ev.clientX - rect.left + 8))}px`;
    heatTipEl.style.top = `${Math.max(4, ev.clientY - rect.top - 26)}px`;
    heatTipEl.hidden = false;
  };

  const onHeatLeave = () => {
    heatTipEl.hidden = true;
  };

  heatCanvas.addEventListener("mousemove", onHeatMove);
  heatCanvas.addEventListener("mouseleave", onHeatLeave);

  const resizeObserver =
    typeof ResizeObserver !== "undefined"
      ? new ResizeObserver(() => {
          if (!currentState) return;
          drawWorkgroupGrid(currentState);
          drawHeatmap(currentState);
        })
      : null;
  resizeObserver?.observe(container);

  return {
    update(state: InspectorState) {
      currentState = state;
      renderAll();
    },
    resize() {
      if (!currentState) return;
      drawWorkgroupGrid(currentState);
      drawHeatmap(currentState);
    },
    destroy() {
      resizeObserver?.disconnect();
      heatCanvas.removeEventListener("mousemove", onHeatMove);
      heatCanvas.removeEventListener("mouseleave", onHeatLeave);
    },
  };
}
