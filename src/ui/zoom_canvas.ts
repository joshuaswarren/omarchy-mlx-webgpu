import type { DispatchEvent, KernelId, OptFlags } from "../kernels/types";
import {
  KERNEL_PROVENANCE,
  countDispatchesForOpts,
  type KernelFamily,
} from "../kernels/provenance";

export interface KernelGraphHandlers {
  onSelectLayer: (layer: number) => void;
  onSelectDispatch: (dispatch: DispatchEvent) => void;
}

export interface KernelGraphState {
  dispatches: DispatchEvent[];
  optFlags: OptFlags;
  faultEnabled: boolean;
  selectedLayer: number;
  selectedDispatchIndex: number | null;
}

export interface KernelGraphController {
  update(state: KernelGraphState): void;
  resize(): void;
  destroy(): void;
}

const FAMILY_COLORS: Record<KernelFamily, string> = {
  attention: "#f59e0b",
  qmm: "#3b82f6",
  norm_rope: "#14b8a6",
  swiglu: "#a855f7",
  memory: "#64748b",
};

const FAMILY_LABELS: Record<KernelFamily, string> = {
  attention: "Attention (SDPA)",
  qmm: "Q4 GEMV",
  norm_rope: "RMSNorm / RoPE",
  swiglu: "SwiGLU MLP",
  memory: "Embed / KV / Add",
};

interface LayerHitBox {
  layer: number; // -1 for embed, 0..27 for layers, -2 for head
  x: number;
  y: number;
  w: number;
  h: number;
  count: number;
}

interface BarHitBox {
  dispatch: DispatchEvent;
  x: number;
  y: number;
  w: number;
  h: number;
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

function formatBytes(bytes: number): string {
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(2)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

export function createKernelGraphView(
  container: HTMLElement,
  handlers: KernelGraphHandlers,
): KernelGraphController {
  container.innerHTML = "";
  container.classList.add("kg-root");

  const header = document.createElement("div");
  header.className = "kg-header";
  header.innerHTML = `
    <div class="kg-header-left">
      <span class="kg-title">28-layer decode graph</span>
      <span class="kg-subtitle" data-kg-summary></span>
    </div>
    <div class="kg-legend">
      ${(Object.keys(FAMILY_COLORS) as KernelFamily[])
        .map(
          (fam) =>
            `<span class="kg-legend-item"><i class="kg-dot" style="background:${FAMILY_COLORS[fam]}"></i>${FAMILY_LABELS[fam]}</span>`,
        )
        .join("")}
    </div>
  `;

  const overviewWrap = document.createElement("div");
  overviewWrap.className = "kg-canvas-wrap kg-overview-wrap";
  const overviewCanvas = document.createElement("canvas");
  overviewCanvas.className = "kg-canvas";
  overviewCanvas.setAttribute("aria-label", "28-layer GPU dispatch graph");
  const overviewTip = document.createElement("div");
  overviewTip.className = "kg-tooltip";
  overviewTip.hidden = true;
  overviewWrap.append(overviewCanvas, overviewTip);

  const detailHeader = document.createElement("div");
  detailHeader.className = "kg-detail-header";
  detailHeader.innerHTML = `
    <div class="kg-detail-left">
      <span class="kg-detail-title" data-kg-layer-title>Layer 14 dispatches</span>
      <span class="kg-detail-meta" data-kg-layer-meta></span>
    </div>
    <div class="kg-detail-actions">
      <button type="button" class="kg-layer-btn" data-kg-prev aria-label="Previous layer">Prev</button>
      <button type="button" class="kg-layer-btn" data-kg-next aria-label="Next layer">Next</button>
      <span class="kg-hint">Click any bar to inspect shader</span>
    </div>
  `;

  const detailWrap = document.createElement("div");
  detailWrap.className = "kg-canvas-wrap kg-detail-wrap";
  const detailCanvas = document.createElement("canvas");
  detailCanvas.className = "kg-canvas";
  detailCanvas.setAttribute("aria-label", "Single-layer dispatch waterfall");
  const detailTip = document.createElement("div");
  detailTip.className = "kg-tooltip";
  detailTip.hidden = true;
  detailWrap.append(detailCanvas, detailTip);

  const eliminatedStrip = document.createElement("div");
  eliminatedStrip.className = "kg-eliminated";

  container.append(header, overviewWrap, detailHeader, detailWrap, eliminatedStrip);

  const summaryEl = header.querySelector<HTMLElement>("[data-kg-summary]")!;
  const layerTitleEl = detailHeader.querySelector<HTMLElement>("[data-kg-layer-title]")!;
  const layerMetaEl = detailHeader.querySelector<HTMLElement>("[data-kg-layer-meta]")!;
  const prevBtn = detailHeader.querySelector<HTMLButtonElement>("[data-kg-prev]")!;
  const nextBtn = detailHeader.querySelector<HTMLButtonElement>("[data-kg-next]")!;

  let currentState: KernelGraphState | null = null;
  let layerHits: LayerHitBox[] = [];
  let barHits: BarHitBox[] = [];
  let hoveredLayer: number | null = null;
  let hoveredBarIndex: number | null = null;

  // Smooth per-layer bar-height animation when opt toggles flip
  let animFromPerLayer = 19;
  let animToPerLayer = 19;
  let animCurrentPerLayer = 19;
  let animStartMs = 0;
  const animDurationMs = 280;
  let rafId = 0;

  function colorForKernel(kernel: KernelId, faultEnabled: boolean): string {
    const fam = KERNEL_PROVENANCE[kernel]?.family ?? "memory";
    if (faultEnabled && fam === "attention") return "#ef4444";
    return FAMILY_COLORS[fam];
  }

  function setupCanvas(canvas: HTMLCanvasElement, cssHeight: number): {
    ctx: CanvasRenderingContext2D;
    width: number;
    height: number;
  } | null {
    const rect = canvas.parentElement?.getBoundingClientRect();
    const width = Math.max(320, Math.floor(rect?.width || 720));
    const height = cssHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function drawOverview(state: KernelGraphState) {
    const setup = setupCanvas(overviewCanvas, 178);
    if (!setup) return;
    const { ctx, width, height } = setup;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0c0f13";
    ctx.fillRect(0, 0, width, height);

    const padL = 38;
    const padR = 14;
    const padT = 18;
    const padB = 26;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    const baselineMax = 19;

    // Horizontal reference gridlines (0, 6, 12, 19 dispatches/layer)
    ctx.font = '500 10px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tick of [0, 6, 12, 19]) {
      const y = padT + plotH - (tick / baselineMax) * plotH;
      ctx.strokeStyle = tick === 19 ? "rgba(139, 149, 165, 0.28)" : "rgba(139, 149, 165, 0.12)";
      ctx.setLineDash(tick === 19 ? [3, 3] : []);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = tick === 19 ? "#94a3b8" : "#64748b";
      ctx.fillText(String(tick), padL - 7, y);
    }

    // Group dispatches by column: Emb (-1 before layer 0), L0..L27, Head (-1 after layer 27)
    const embedDispatches: DispatchEvent[] = [];
    const headDispatches: DispatchEvent[] = [];
    const byLayer: DispatchEvent[][] = Array.from({ length: 28 }, () => []);
    let sawLayer = false;
    for (const d of state.dispatches) {
      if (d.layer >= 0 && d.layer < 28) {
        sawLayer = true;
        byLayer[d.layer].push(d);
      } else if (!sawLayer) {
        embedDispatches.push(d);
      } else {
        headDispatches.push(d);
      }
    }

    const columns: Array<{ id: number; label: string; items: DispatchEvent[]; isLayer: boolean }> = [
      { id: -1, label: "E", items: embedDispatches, isLayer: false },
      ...byLayer.map((items, l) => ({ id: l, label: String(l), items, isLayer: true })),
      { id: -2, label: "H", items: headDispatches, isLayer: false },
    ];

    const colStep = plotW / columns.length;
    const barW = Math.max(6, Math.floor(colStep - 4));
    const slotH = plotH / baselineMax;
    layerHits = [];

    for (let c = 0; c < columns.length; c++) {
      const col = columns[c];
      const cx = padL + c * colStep + (colStep - barW) / 2;
      const isSelected = col.isLayer && col.id === state.selectedLayer;
      const isHovered = hoveredLayer === col.id;

      // Ghost baseline outline for transformer layers showing eliminated dispatches
      if (col.isLayer) {
        ctx.strokeStyle = isSelected
          ? "rgba(245, 158, 11, 0.55)"
          : "rgba(139, 149, 165, 0.18)";
        ctx.lineWidth = isSelected ? 1.5 : 1;
        ctx.setLineDash([2, 2]);
        ctx.strokeRect(cx + 0.5, padT + 0.5, barW - 1, plotH - 1);
        ctx.setLineDash([]);
      }

      if (isSelected) {
        ctx.fillStyle = "rgba(245, 158, 11, 0.10)";
        ctx.fillRect(cx - 2, padT - 3, barW + 4, plotH + 6);
      } else if (isHovered) {
        ctx.fillStyle = "rgba(232, 236, 241, 0.06)";
        ctx.fillRect(cx - 1, padT - 2, barW + 2, plotH + 4);
      }

      // Scale active stack height smoothly during opt-toggle transitions
      const targetCount = col.items.length;
      const visualScale =
        col.isLayer && targetCount > 0 ? animCurrentPerLayer / targetCount : 1;

      const segGap = 1.25;
      for (let i = 0; i < col.items.length; i++) {
        const d = col.items[i];
        const segH = Math.max(2, slotH * visualScale - segGap);
        const y = padT + plotH - (i + 1) * slotH * visualScale + segGap / 2;
        ctx.fillStyle = colorForKernel(d.kernel, state.faultEnabled);
        ctx.globalAlpha = isSelected || isHovered ? 1.0 : 0.88;
        ctx.fillRect(cx, y, barW, segH);
      }
      ctx.globalAlpha = 1.0;

      // X axis labels (every 4th layer + selected + bookends)
      const showLabel =
        !col.isLayer ||
        col.id === 0 ||
        col.id === 27 ||
        col.id % 4 === 0 ||
        col.id === state.selectedLayer;
      if (showLabel) {
        ctx.fillStyle = isSelected ? "#f59e0b" : "#8b95a5";
        ctx.font = isSelected
          ? '600 10px "IBM Plex Mono", ui-monospace, monospace'
          : '400 10px "IBM Plex Mono", ui-monospace, monospace';
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(col.label, cx + barW / 2, padT + plotH + 7);
      }

      layerHits.push({
        layer: col.id,
        x: padL + c * colStep,
        y: padT,
        w: colStep,
        h: plotH + 18,
        count: col.items.length,
      });
    }

    // Baseline callout badge in top-right of plot
    const counts = countDispatchesForOpts(state.optFlags);
    if (counts.savedDispatches > 0) {
      const badge = `-${counts.savedDispatches} dispatches/tok vs baseline`;
      ctx.font = '600 10px "IBM Plex Mono", ui-monospace, monospace';
      const tw = ctx.measureText(badge).width + 14;
      const bx = width - padR - tw;
      const by = 3;
      ctx.fillStyle = "rgba(20, 184, 166, 0.14)";
      ctx.strokeStyle = "rgba(20, 184, 166, 0.45)";
      ctx.lineWidth = 1;
      ctx.fillRect(bx, by, tw, 16);
      ctx.strokeRect(bx + 0.5, by + 0.5, tw - 1, 15);
      ctx.fillStyle = "#2dd4bf";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(badge, bx + 7, by + 8.5);
    }
  }

  function drawDetail(state: KernelGraphState) {
    const setup = setupCanvas(detailCanvas, 128);
    if (!setup) return;
    const { ctx, width, height } = setup;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0c0f13";
    ctx.fillRect(0, 0, width, height);

    const layerEvents = state.dispatches.filter((d) => d.layer === state.selectedLayer);
    layerTitleEl.textContent = `Layer ${state.selectedLayer} dispatch stream`;
    const totalUs = layerEvents.reduce((acc, d) => acc + (d.durationUs ?? 16), 0);
    const totalRead = layerEvents.reduce((acc, d) => acc + d.bytesRead, 0);
    const fusedOut = Math.max(0, 19 - layerEvents.length);
    layerMetaEl.textContent = `${layerEvents.length} dispatches (${fusedOut} fused out) | ${formatBytes(totalRead)} read`;

    const padX = 12;
    const barY = 22;
    const barH = 54;
    const usableW = width - padX * 2;
    const gap = 4;
    const n = layerEvents.length;
    barHits = [];

    if (n === 0) return;

    // Weight bars gently by estimated duration so heavy Q4/SDPA ops read wider while every bar stays clickable
    const weights = layerEvents.map((d) => Math.max(18, Math.sqrt(d.durationUs ?? 20) * 10));
    const sumW = weights.reduce((a, b) => a + b, 0);
    const totalGap = gap * (n - 1);
    const availW = Math.max(100, usableW - totalGap);

    let x = padX;
    for (let i = 0; i < n; i++) {
      const d = layerEvents[i];
      const w = Math.max(24, (weights[i] / sumW) * availW);
      const isSelected = state.selectedDispatchIndex === d.index;
      const isHovered = hoveredBarIndex === d.index;
      const color = colorForKernel(d.kernel, state.faultEnabled);

      // Bar body
      ctx.fillStyle = color;
      ctx.globalAlpha = isSelected ? 1.0 : isHovered ? 0.95 : 0.78;
      ctx.fillRect(x, barY, w, barH);
      ctx.globalAlpha = 1.0;

      // Top highlight edge
      ctx.fillStyle = "rgba(255,255,255,0.22)";
      ctx.fillRect(x, barY, w, 2);

      if (isSelected || isHovered) {
        ctx.strokeStyle = "#f8fafc";
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.strokeRect(x + 0.5, barY + 0.5, w - 1, barH - 1);
      }

      // Compact label inside bar so every dispatch reads cleanly
      const compactLabel = (ev: DispatchEvent, barWidth: number): string => {
        if (ev.kernel === "sdpa_decode_online") return barWidth >= 60 ? "SDPA #997" : "#997";
        if (ev.kernel === "sdpa_decode_split_p1") return barWidth >= 50 ? "SDPA p1" : "p1";
        if (ev.kernel === "sdpa_decode_split_p2") return barWidth >= 50 ? "SDPA p2" : "p2";
        if (ev.kernel === "fast_rope_norm") return ev.label.startsWith("K") ? "K+RoPE" : "Q+RoPE";
        if (ev.kernel === "fast_norm") return "Norm";
        if (ev.kernel === "fast_rope") return "RoPE";
        if (ev.kernel === "kv_cache_write") return "KV";
        if (ev.kernel === "residual_add") return "Add";
        if (ev.kernel === "qmm_vec_swiglu") return "SwiGLU Q4";
        if (ev.kernel === "swiglu") return barWidth >= 46 ? "SwiGLU" : "SiLU";
        if (ev.label.startsWith("q_proj")) return "Q proj";
        if (ev.label.startsWith("k_proj")) return "K proj";
        if (ev.label.startsWith("v_proj")) return "V proj";
        if (ev.label.startsWith("o_proj")) return "O proj";
        if (ev.label.startsWith("gate_proj")) return "Gate";
        if (ev.label.startsWith("up_proj")) return "Up";
        if (ev.label.startsWith("down_proj")) return "Down";
        return KERNEL_PROVENANCE[ev.kernel]?.shortLabel ?? ev.kernel;
      };
      const shortName = compactLabel(d, w);
      ctx.save();
      ctx.beginPath();
      ctx.rect(x + 3, barY + 3, Math.max(0, w - 6), barH - 6);
      ctx.clip();
      ctx.fillStyle = "#080b10";
      ctx.font = '600 10px "IBM Plex Mono", ui-monospace, monospace';
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(shortName, x + 4, barY + 8);
      if (w >= 48) {
        ctx.font = '500 9px "IBM Plex Mono", ui-monospace, monospace';
        ctx.fillStyle = "rgba(8, 11, 16, 0.85)";
        ctx.fillText(`${d.workgroups[0]}x${d.workgroups[1]}`, x + 4, barY + 24);
      }
      ctx.restore();

      // Index tick underneath
      ctx.fillStyle = isSelected ? "#f59e0b" : "#64748b";
      ctx.font = '500 9px "IBM Plex Mono", ui-monospace, monospace';
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(String(i + 1), x + w / 2, barY + barH + 8);

      barHits.push({ dispatch: d, x, y: barY, w, h: barH });
      x += w + gap;
    }

    // Subtle sequence arrow line at bottom
    ctx.fillStyle = "#64748b";
    ctx.font = '400 10px "IBM Plex Sans", sans-serif';
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(
      `Execution order within Layer ${state.selectedLayer} (${totalUs} us est. profile)`,
      padX,
      height - 6,
    );

    // Render eliminated fusion badges cleanly underneath
    const eliminated: string[] = [];
    if (state.optFlags.oneDispatchAttention) {
      eliminated.push("PR #997 removed sdpa_decode_split_p2 (-1/layer, -28/tok)");
    }
    if (state.optFlags.fusedRopeNorm) {
      eliminated.push("Fused QK-Norm+RoPE removed q_rope & k_rope (-2/layer, -56/tok)");
    }
    if (state.optFlags.fusedSwigluQmm) {
      eliminated.push("Fused SwiGLU QMM removed separate up_proj & swiglu (-2/layer, -56/tok)");
    }
    if (eliminated.length === 0) {
      eliminatedStrip.innerHTML = `<span class="kg-elim-none">Baseline unfused graph active: all 19 dispatches per layer running (537/token).</span>`;
    } else {
      eliminatedStrip.innerHTML = eliminated
        .map((text) => `<span class="kg-elim-chip">${text}</span>`)
        .join("");
    }
  }

  function renderAll() {
    if (!currentState) return;
    const counts = countDispatchesForOpts(currentState.optFlags);
    summaryEl.textContent = `${counts.totalDispatches} dispatches/tok (${counts.perLayerDispatches}/layer, baseline ${counts.baselineDispatches})`;
    drawOverview(currentState);
    drawDetail(currentState);
  }

  function stepAnimation(now: number) {
    const t = Math.min(1, (now - animStartMs) / animDurationMs);
    const eased = 1 - Math.pow(1 - t, 3);
    animCurrentPerLayer = animFromPerLayer + (animToPerLayer - animFromPerLayer) * eased;
    renderAll();
    if (t < 1) {
      rafId = requestAnimationFrame(stepAnimation);
    } else {
      rafId = 0;
    }
  }

  function handleOverviewMove(ev: MouseEvent) {
    if (!currentState) return;
    const rect = overviewCanvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    const hit = layerHits.find((b) => mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h);
    if (!hit) {
      if (hoveredLayer !== null) {
        hoveredLayer = null;
        overviewTip.hidden = true;
        renderAll();
      }
      return;
    }
    if (hoveredLayer !== hit.layer) {
      hoveredLayer = hit.layer;
      renderAll();
    }
    const label =
      hit.layer === -1 ? "Embed" : hit.layer === -2 ? "Final Norm + LM Head + Argmax" : `Layer ${hit.layer}`;
    overviewTip.textContent = `${label}: ${hit.count} dispatches`;
    overviewTip.style.left = `${Math.min(rect.width - 190, Math.max(8, mx + 10))}px`;
    overviewTip.style.top = `${Math.max(6, my - 28)}px`;
    overviewTip.hidden = false;
  }

  function handleOverviewClick(ev: MouseEvent) {
    const rect = overviewCanvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    const hit = layerHits.find((b) => mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h);
    if (hit && hit.layer >= 0) {
      handlers.onSelectLayer(hit.layer);
    }
  }

  function handleDetailMove(ev: MouseEvent) {
    const rect = detailCanvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    const hit = barHits.find((b) => mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h);
    if (!hit) {
      if (hoveredBarIndex !== null) {
        hoveredBarIndex = null;
        detailTip.hidden = true;
        detailCanvas.style.cursor = "default";
        renderAll();
      }
      return;
    }
    detailCanvas.style.cursor = "pointer";
    if (hoveredBarIndex !== hit.dispatch.index) {
      hoveredBarIndex = hit.dispatch.index;
      renderAll();
    }
    const d = hit.dispatch;
    detailTip.textContent = `${d.label} | WG (${d.workgroups.join(",")}) | read ${formatBytes(d.bytesRead)}`;
    detailTip.style.left = `${Math.min(rect.width - 280, Math.max(8, mx + 10))}px`;
    detailTip.style.top = `${Math.max(4, my - 30)}px`;
    detailTip.hidden = false;
  }

  function handleDetailClick(ev: MouseEvent) {
    const rect = detailCanvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    const hit = barHits.find((b) => mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h);
    if (hit) {
      handlers.onSelectDispatch(hit.dispatch);
    }
  }

  const onOverviewLeave = () => {
    hoveredLayer = null;
    overviewTip.hidden = true;
    renderAll();
  };
  const onDetailLeave = () => {
    hoveredBarIndex = null;
    detailTip.hidden = true;
    renderAll();
  };
  const onPrev = () => {
    if (!currentState) return;
    handlers.onSelectLayer((currentState.selectedLayer + 27) % 28);
  };
  const onNext = () => {
    if (!currentState) return;
    handlers.onSelectLayer((currentState.selectedLayer + 1) % 28);
  };

  overviewCanvas.addEventListener("mousemove", handleOverviewMove);
  overviewCanvas.addEventListener("mouseleave", onOverviewLeave);
  overviewCanvas.addEventListener("click", handleOverviewClick);
  detailCanvas.addEventListener("mousemove", handleDetailMove);
  detailCanvas.addEventListener("mouseleave", onDetailLeave);
  detailCanvas.addEventListener("click", handleDetailClick);
  prevBtn.addEventListener("click", onPrev);
  nextBtn.addEventListener("click", onNext);

  const resizeObserver =
    typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => renderAll()) : null;
  resizeObserver?.observe(container);

  return {
    update(state: KernelGraphState) {
      const counts = countDispatchesForOpts(state.optFlags);
      const changed = currentState !== null && counts.perLayerDispatches !== animToPerLayer;
      currentState = state;
      if (changed && !prefersReducedMotion()) {
        animFromPerLayer = animCurrentPerLayer;
        animToPerLayer = counts.perLayerDispatches;
        animStartMs = performance.now();
        if (rafId) cancelAnimationFrame(rafId);
        rafId = requestAnimationFrame(stepAnimation);
      } else {
        animFromPerLayer = counts.perLayerDispatches;
        animToPerLayer = counts.perLayerDispatches;
        animCurrentPerLayer = counts.perLayerDispatches;
        renderAll();
      }
    },
    resize() {
      renderAll();
    },
    destroy() {
      if (rafId) cancelAnimationFrame(rafId);
      resizeObserver?.disconnect();
      overviewCanvas.removeEventListener("mousemove", handleOverviewMove);
      overviewCanvas.removeEventListener("mouseleave", onOverviewLeave);
      overviewCanvas.removeEventListener("click", handleOverviewClick);
      detailCanvas.removeEventListener("mousemove", handleDetailMove);
      detailCanvas.removeEventListener("mouseleave", onDetailLeave);
      detailCanvas.removeEventListener("click", handleDetailClick);
      prevBtn.removeEventListener("click", onPrev);
      nextBtn.removeEventListener("click", onNext);
    },
  };
}
