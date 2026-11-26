/**
 * Pure TypeScript CPU reference implementations for every omarchy-mlx kernel.
 * Arithmetic runs in Float64 internally and rounds to Float32 on output so the
 * in-page parity checker has an independent ground truth.
 */

export function bf16ToF32(bits16: number): number {
  const u32 = (bits16 & 0xffff) << 16;
  const buf = new ArrayBuffer(4);
  new Uint32Array(buf)[0] = u32;
  return new Float32Array(buf)[0];
}

/** RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w (matching fast_norm.comp). */
export function rmsNormCpu(
  input: Float32Array,
  weight: Float32Array,
  rows: number,
  rowLength: number,
  eps: number,
): Float32Array {
  const out = new Float32Array(rows * rowLength);
  const broadcast = weight.length === 1;
  for (let r = 0; r < rows; r++) {
    const base = r * rowLength;
    let sumSq = 0;
    for (let i = 0; i < rowLength; i++) {
      const v = input[base + i];
      sumSq += v * v;
    }
    const norm = 1.0 / Math.sqrt(sumSq / rowLength + eps);
    for (let i = 0; i < rowLength; i++) {
      const w = broadcast ? weight[0] : weight[i];
      out[base + i] = input[base + i] * norm * w;
    }
  }
  return out;
}

/**
 * Single-query GQA Scaled Dot-Product Attention reference (matching
 * sdpa_decode_online.comp and sdpa_decode_split_p1/p2.comp).
 *
 * Layout:
 *   q: [heads, headDim]
 *   kCache, vCache: [kvHeads, maxSeq, headDim] (first kLen positions active)
 *   out: [heads, headDim]
 */
export function sdpaDecodeCpu(
  q: Float32Array,
  kCache: Float32Array,
  vCache: Float32Array,
  heads: number,
  kvHeads: number,
  kLen: number,
  maxSeq: number,
  headDim: number,
  scale: number,
): Float32Array {
  const out = new Float32Array(heads * headDim);
  const repeats = Math.floor(heads / kvHeads);
  const scores = new Float64Array(kLen);

  for (let h = 0; h < heads; h++) {
    const kvh = Math.floor(h / repeats);
    const qBase = h * headDim;
    const kvHeadBase = kvh * maxSeq * headDim;

    let maxScore = -Infinity;
    for (let k = 0; k < kLen; k++) {
      const kBase = kvHeadBase + k * headDim;
      let dot = 0;
      for (let d = 0; d < headDim; d++) {
        dot += q[qBase + d] * kCache[kBase + d];
      }
      const s = dot * scale;
      scores[k] = s;
      if (s > maxScore) maxScore = s;
    }

    let sumExp = 0;
    for (let k = 0; k < kLen; k++) {
      const w = Math.exp(scores[k] - maxScore);
      scores[k] = w;
      sumExp += w;
    }

    const outBase = h * headDim;
    for (let d = 0; d < headDim; d++) {
      let acc = 0;
      for (let k = 0; k < kLen; k++) {
        acc += scores[k] * vCache[kvHeadBase + k * headDim + d];
      }
      out[outBase + d] = acc / sumExp;
    }
  }
  return out;
}

/** Half-split RoPE reference (Qwen3 traditional=false, matching fast_rope.comp). */
export function ropeCpu(
  input: Float32Array,
  heads: number,
  headDim: number,
  position: number,
  ropeTheta: number,
): Float32Array {
  const out = new Float32Array(heads * headDim);
  const half = headDim >> 1;
  for (let h = 0; h < heads; h++) {
    const base = h * headDim;
    for (let i = 0; i < half; i++) {
      const freq = Math.pow(ropeTheta, -(2 * i) / headDim);
      const angle = position * freq;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      const x0 = input[base + i];
      const x1 = input[base + i + half];
      out[base + i] = x0 * c - x1 * s;
      out[base + i + half] = x1 * c + x0 * s;
    }
  }
  return out;
}

/**
 * MLX affine 4-bit quantized matrix-vector product (group_size = 64, bits = 4):
 *   out[n] = sum_{k=0..K-1} x[k] * (nibble(w[n, k]) * scales[n, k/64] + biases[n, k/64])
 * where each uint32 word holds 8 little-endian 4-bit unsigned nibbles along K.
 */
export function qmmQ4Cpu(
  x: Float32Array,
  weightsU32: Uint32Array,
  scalesF32: Float32Array,
  biasesF32: Float32Array,
  N: number,
  K: number,
  groupSize = 64,
): Float32Array {
  const out = new Float32Array(N);
  const wordsPerRow = K >>> 3;
  const groupsPerRow = Math.floor(K / groupSize);
  for (let n = 0; n < N; n++) {
    const wRow = n * wordsPerRow;
    const gRow = n * groupsPerRow;
    let acc = 0;
    for (let g = 0; g < groupsPerRow; g++) {
      const scale = scalesF32[gRow + g];
      const bias = biasesF32[gRow + g];
      const kStart = g * groupSize;
      const wStart = wRow + (kStart >>> 3);
      const wordsInGroup = groupSize >>> 3;
      for (let wi = 0; wi < wordsInGroup; wi++) {
        const word = weightsU32[wStart + wi];
        const kBase = kStart + (wi << 3);
        for (let j = 0; j < 8; j++) {
          const q = (word >>> (j << 2)) & 0xf;
          acc += x[kBase + j] * (q * scale + bias);
        }
      }
    }
    out[n] = acc;
  }
  return out;
}

/** SwiGLU: out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]. */
export function swigluCpu(gate: Float32Array, up: Float32Array): Float32Array {
  const out = new Float32Array(gate.length);
  for (let i = 0; i < gate.length; i++) {
    const g = gate[i];
    out[i] = (g / (1 + Math.exp(-g))) * up[i];
  }
  return out;
}

/** MLX 4-bit embedding lookup for one token id. */
export function embedQ4Cpu(
  tokenId: number,
  weightsU32: Uint32Array,
  scalesF32: Float32Array,
  biasesF32: Float32Array,
  hiddenSize: number,
  groupSize = 64,
): Float32Array {
  const out = new Float32Array(hiddenSize);
  const wordsPerRow = hiddenSize >>> 3;
  const groupsPerRow = Math.floor(hiddenSize / groupSize);
  const wRow = tokenId * wordsPerRow;
  const gRow = tokenId * groupsPerRow;
  for (let g = 0; g < groupsPerRow; g++) {
    const scale = scalesF32[gRow + g];
    const bias = biasesF32[gRow + g];
    const kStart = g * groupSize;
    const wStart = wRow + (kStart >>> 3);
    for (let wi = 0; wi < groupSize >>> 3; wi++) {
      const word = weightsU32[wStart + wi];
      const kBase = kStart + (wi << 3);
      for (let j = 0; j < 8; j++) {
        const q = (word >>> (j << 2)) & 0xf;
        out[kBase + j] = q * scale + bias;
      }
    }
  }
  return out;
}

/** Fused per-head RMSNorm + half-split RoPE reference (matching fast_rope_norm.comp). */
export function ropeNormCpu(
  input: Float32Array,
  weight: Float32Array,
  heads: number,
  headDim: number,
  position: number,
  eps: number,
  ropeTheta: number,
): Float32Array {
  const normed = rmsNormCpu(input, weight, heads, headDim, eps);
  return ropeCpu(normed, heads, headDim, position, ropeTheta);
}

/** Fused Gate + Up Q4 affine QMM + SwiGLU reference (matching qmm_vec.comp QMM_VEC_SWIGLU). */
export function qmmSwigluQ4Cpu(
  x: Float32Array,
  gateWeightsU32: Uint32Array,
  gateScalesF32: Float32Array,
  gateBiasesF32: Float32Array,
  upWeightsU32: Uint32Array,
  upScalesF32: Float32Array,
  upBiasesF32: Float32Array,
  N: number,
  K: number,
  groupSize = 64,
): Float32Array {
  const gate = qmmQ4Cpu(x, gateWeightsU32, gateScalesF32, gateBiasesF32, N, K, groupSize);
  const up = qmmQ4Cpu(x, upWeightsU32, upScalesF32, upBiasesF32, N, K, groupSize);
  return swigluCpu(gate, up);
}

/** Elementwise residual add: out[i] = a[i] + b[i]. */
export function residualAddCpu(a: Float32Array, b: Float32Array): Float32Array {
  const n = Math.min(a.length, b.length);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
  return out;
}

/**
 * Single-token KV cache scatter write into [kvHeads, maxSeq, headDim] at position `pos`.
 * Returns updated copies of `kCache` and `vCache` without mutating caller arrays.
 */
export function kvCacheWriteCpu(
  kIn: Float32Array,
  vIn: Float32Array,
  kCache: Float32Array,
  vCache: Float32Array,
  kvHeads: number,
  headDim: number,
  maxSeq: number,
  pos: number,
): { kCache: Float32Array; vCache: Float32Array } {
  const nextK = new Float32Array(kCache);
  const nextV = new Float32Array(vCache);
  for (let h = 0; h < kvHeads; h++) {
    const srcBase = h * headDim;
    const dstBase = (h * maxSeq + pos) * headDim;
    for (let d = 0; d < headDim; d++) {
      nextK[dstBase + d] = kIn[srcBase + d];
      nextV[dstBase + d] = vIn[srcBase + d];
    }
  }
  return { kCache: nextK, vCache: nextV };
}

/** Deterministic argmax with lowest-index tie-breaking (matching elem.wgsl argmax_p1/p2). */
export function argmaxCpu(logits: Float32Array): { index: number; value: number } {
  if (logits.length === 0) {
    return { index: -1, value: -Infinity };
  }
  let bestIdx = 0;
  let bestVal = logits[0];
  for (let i = 1; i < logits.length; i++) {
    const v = logits[i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return { index: bestIdx, value: bestVal };
}

export interface QuantizedQ4Matrix {
  weightsU32: Uint32Array;
  scalesF32: Float32Array;
  biasesF32: Float32Array;
}

/** Deterministic MLX affine 4-bit (g64) quantized weight generator for parity tests. */
export function seededQ4Weights(
  N: number,
  K: number,
  seed: number,
  groupSize = 64,
): QuantizedQ4Matrix {
  const wordsPerRow = K >>> 3;
  const groupsPerRow = Math.floor(K / groupSize);
  const weightsU32 = new Uint32Array(N * wordsPerRow);
  const scalesF32 = new Float32Array(N * groupsPerRow);
  const biasesF32 = new Float32Array(N * groupsPerRow);

  let state = seed >>> 0;
  const nextU32 = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  };

  for (let i = 0; i < weightsU32.length; i++) {
    weightsU32[i] = nextU32();
  }
  for (let i = 0; i < scalesF32.length; i++) {
    const uScale = nextU32() / 4294967296;
    const uBias = nextU32() / 4294967296;
    scalesF32[i] = 0.005 + uScale * 0.04;
    biasesF32[i] = (uBias * 2 - 1) * 0.15;
  }

  return { weightsU32, scalesF32, biasesF32 };
}

/** Deterministic PRNG (Mulberry32) for reproducible parity test vectors. */
export function seededFloat32(length: number, seed: number, scale = 1.0): Float32Array {
  let state = seed >>> 0;
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    const u = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    out[i] = (u * 2 - 1) * scale;
  }
  return out;
}

export interface ErrorMetrics {
  maxAbsDiff: number;
  meanAbsDiff: number;
  cosineSimilarity: number;
  diffSample: number[];
  gpuSample: number[];
  cpuSample: number[];
}

export function compareArrays(gpu: Float32Array, cpu: Float32Array): ErrorMetrics {
  const n = Math.min(gpu.length, cpu.length);
  let maxAbsDiff = 0;
  let sumAbsDiff = 0;
  let dot = 0;
  let normG = 0;
  let normC = 0;
  for (let i = 0; i < n; i++) {
    const g = gpu[i];
    const c = cpu[i];
    const diff = Math.abs(g - c);
    if (!Number.isFinite(diff)) {
      maxAbsDiff = Infinity;
    } else if (diff > maxAbsDiff) {
      maxAbsDiff = diff;
    }
    sumAbsDiff += Number.isFinite(diff) ? diff : 1e9;
    dot += g * c;
    normG += g * g;
    normC += c * c;
  }
  const denom = Math.sqrt(normG) * Math.sqrt(normC);
  const cosineSimilarity =
    denom > 0 ? dot / denom : normG === 0 && normC === 0 ? 1.0 : 0.0;
  const sampleLen = Math.min(64, n);
  const diffSample: number[] = [];
  const gpuSample: number[] = [];
  const cpuSample: number[] = [];
  for (let i = 0; i < sampleLen; i++) {
    diffSample.push(gpu[i] - cpu[i]);
    gpuSample.push(gpu[i]);
    cpuSample.push(cpu[i]);
  }
  return {
    maxAbsDiff,
    meanAbsDiff: n > 0 ? sumAbsDiff / n : 0,
    cosineSimilarity,
    diffSample,
    gpuSample,
    cpuSample,
  };
}
