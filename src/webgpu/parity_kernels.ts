import elemWgsl from "../kernels/wgsl/elem.wgsl?raw";
import fastRopeWgsl from "../kernels/wgsl/fast_rope.wgsl?raw";
import fastRopeNormWgsl from "../kernels/wgsl/fast_rope_norm.wgsl?raw";
import qmmVecWgsl from "../kernels/wgsl/qmm_vec.wgsl?raw";
import qmmVecSwigluWgsl from "../kernels/wgsl/qmm_vec_swiglu.wgsl?raw";
import swigluWgsl from "../kernels/wgsl/swiglu.wgsl?raw";
import type { ParityResult } from "../kernels/types";
import {
  argmaxCpu,
  compareArrays,
  embedQ4Cpu,
  kvCacheWriteCpu,
  qmmQ4Cpu,
  qmmSwigluQ4Cpu,
  residualAddCpu,
  ropeCpu,
  ropeNormCpu,
  seededFloat32,
  seededQ4Weights,
  swigluCpu,
} from "../reference/cpu";
import {
  createCheckedShaderModule,
  createStorageBuffer,
  createUniformBuffer,
  readFloat32Buffer,
} from "./device";

export const KERNEL_WGSL = {
  fast_rope: fastRopeWgsl,
  fast_rope_norm: fastRopeNormWgsl,
  qmm_vec: qmmVecWgsl,
  qmm_vec_swiglu: qmmVecSwigluWgsl,
  swiglu: swigluWgsl,
  elem: elemWgsl,
} as const;

const QMM_ROWS_PER_WG = 8;
const MAX_WG_DIM = 65535;

/**
 * Computes a 2D-safe dispatch grid `[gridX, gridY, 1]` for `qmm_vec.wgsl` and
 * `qmm_vec_swiglu.wgsl` (8 output rows per workgroup, `gridX <= 65535`).
 * For Qwen3 vocab `N = 151936`, returns `[18992, 1, 1]`.
 */
export function qmmDispatchGrid(n: number): [number, number, number] {
  const totalWg = Math.max(1, Math.ceil(n / QMM_ROWS_PER_WG));
  if (totalWg <= MAX_WG_DIM) {
    return [totalWg, 1, 1];
  }
  const gridX = MAX_WG_DIM;
  const gridY = Math.ceil(totalWg / gridX);
  return [gridX, gridY, 1];
}

/** Chooses stage-1 workgroup count for 2-stage `argmax_p1` / `argmax_p2`. */
export function argmaxNumBlocks(count: number): number {
  return Math.max(1, Math.min(256, Math.ceil(count / 256)));
}

async function readUint32Buffer(
  device: GPUDevice,
  src: GPUBuffer,
  elementCount: number,
): Promise<Uint32Array> {
  const byteLength = elementCount * 4;
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return copy;
}

/**
 * Verifies half-split RoPE (`fast_rope.wgsl`, pair `(i, i + headDim/2)`, `theta = 1e6`)
 * against `ropeCpu`.
 */
export async function verifyFastRopeParity(
  device: GPUDevice,
  heads = 16,
  headDim = 128,
  position = 42,
  ropeTheta = 1_000_000,
  faultMode = false,
): Promise<ParityResult> {
  const input = seededFloat32(heads * headDim, 201, 1.5);

  const cpuStart = performance.now();
  const cpuOut = ropeCpu(input, heads, headDim, position, ropeTheta);
  const cpuMs = performance.now() - cpuStart;

  const module = await createCheckedShaderModule(device, "fast_rope", fastRopeWgsl);
  const pipeline = device.createComputePipeline({
    label: "fast_rope_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const srcBuf = createStorageBuffer(device, input);
  const dstBuf = device.createBuffer({
    size: heads * headDim * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = new ArrayBuffer(32);
  const u32 = new Uint32Array(paramBuf);
  const f32 = new Float32Array(paramBuf);
  u32[0] = heads;
  u32[1] = headDim;
  u32[2] = position;
  u32[3] = faultMode ? 1 : 0;
  f32[4] = ropeTheta;
  f32[5] = 1.0;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: srcBuf } },
      { binding: 1, resource: { buffer: dstBuf } },
      { binding: 2, resource: { buffer: uniformBuf } },
    ],
  });

  const gpuStart = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(heads, 1, 1);
  pass.end();
  device.queue.submit([enc.finish()]);
  const gpuOut = await readFloat32Buffer(device, dstBuf, heads * headDim);
  const gpuMs = performance.now() - gpuStart;

  srcBuf.destroy();
  dstBuf.destroy();
  uniformBuf.destroy();

  const metrics = compareArrays(gpuOut, cpuOut);
  // WGSL f32 sin/cos guarantees <= 2^-11 (~4.88e-4) error on [-pi, pi] across all backends.
  const tolerance = 1e-3;
  return {
    kernel: "fast_rope",
    label: "Half-Split RoPE (fast_rope.comp)",
    shape: `[${heads}, ${headDim}] pos=${position}`,
    elements: heads * headDim,
    ...metrics,
    tolerance,
    passed: metrics.maxAbsDiff <= tolerance,
    gpuMs,
    cpuMs,
    faultInjected: faultMode,
  };
}

/**
 * Verifies fused per-head Q/K RMSNorm + half-split RoPE (`fast_rope_norm.wgsl`)
 * against `ropeNormCpu`.
 */
export async function verifyFastRopeNormParity(
  device: GPUDevice,
  heads = 16,
  headDim = 128,
  position = 42,
  eps = 1e-6,
  ropeTheta = 1_000_000,
  faultMode = false,
): Promise<ParityResult> {
  const input = seededFloat32(heads * headDim, 211, 1.5);
  const weight = seededFloat32(headDim, 212, 1.0);

  const cpuStart = performance.now();
  const cpuOut = ropeNormCpu(input, weight, heads, headDim, position, eps, ropeTheta);
  const cpuMs = performance.now() - cpuStart;

  const module = await createCheckedShaderModule(device, "fast_rope_norm", fastRopeNormWgsl);
  const pipeline = device.createComputePipeline({
    label: "fast_rope_norm_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const srcBuf = createStorageBuffer(device, input);
  const wBuf = createStorageBuffer(device, weight);
  const dstBuf = device.createBuffer({
    size: heads * headDim * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = new ArrayBuffer(32);
  const u32 = new Uint32Array(paramBuf);
  const f32 = new Float32Array(paramBuf);
  u32[0] = heads;
  u32[1] = headDim;
  u32[2] = position;
  u32[3] = 0; // weight_broadcast = false
  f32[4] = eps;
  f32[5] = ropeTheta;
  f32[6] = 1.0;
  u32[7] = faultMode ? 1 : 0;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: srcBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: dstBuf } },
      { binding: 3, resource: { buffer: uniformBuf } },
    ],
  });

  const gpuStart = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(heads, 1, 1);
  pass.end();
  device.queue.submit([enc.finish()]);
  const gpuOut = await readFloat32Buffer(device, dstBuf, heads * headDim);
  const gpuMs = performance.now() - gpuStart;

  srcBuf.destroy();
  wBuf.destroy();
  dstBuf.destroy();
  uniformBuf.destroy();

  const metrics = compareArrays(gpuOut, cpuOut);
  // WGSL f32 sin/cos guarantees <= 2^-11 (~4.88e-4) error on [-pi, pi] across all backends.
  const tolerance = 1e-3;
  return {
    kernel: "fast_rope_norm",
    label: "Fused RMSNorm + RoPE (fast_rope_norm.comp)",
    shape: `[${heads}, ${headDim}] pos=${position}`,
    elements: heads * headDim,
    ...metrics,
    tolerance,
    passed: metrics.maxAbsDiff <= tolerance,
    gpuMs,
    cpuMs,
    faultInjected: faultMode,
  };
}

/**
 * Verifies MLX affine 4-bit (g64) quantized matrix-vector multiply (`qmm_vec.wgsl`)
 * against `qmmQ4Cpu`. Supports N up to 151936+ via 8 rows/WG + 2D grid indexing.
 */
export async function verifyQmmVecParity(
  device: GPUDevice,
  N = 3072,
  K = 1024,
  groupSize = 64,
  faultMode = false,
): Promise<ParityResult> {
  const x = seededFloat32(K, 301, 1.0);
  const { weightsU32, scalesF32, biasesF32 } = seededQ4Weights(N, K, 302, groupSize);

  const cpuStart = performance.now();
  const cpuOut = qmmQ4Cpu(x, weightsU32, scalesF32, biasesF32, N, K, groupSize);
  const cpuMs = performance.now() - cpuStart;

  const module = await createCheckedShaderModule(device, "qmm_vec", qmmVecWgsl);
  const pipeline = device.createComputePipeline({
    label: "qmm_vec_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const xBuf = createStorageBuffer(device, x);
  const wBuf = createStorageBuffer(device, weightsU32);
  const sBuf = createStorageBuffer(device, scalesF32);
  const bBuf = createStorageBuffer(device, biasesF32);
  const dstBuf = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = new ArrayBuffer(32);
  const u32 = new Uint32Array(paramBuf);
  u32[0] = N;
  u32[1] = K;
  u32[2] = groupSize;
  u32[3] = faultMode ? 1 : 0;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: sBuf } },
      { binding: 3, resource: { buffer: bBuf } },
      { binding: 4, resource: { buffer: dstBuf } },
      { binding: 5, resource: { buffer: uniformBuf } },
    ],
  });

  const [gx, gy, gz] = qmmDispatchGrid(N);
  const gpuStart = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(gx, gy, gz);
  pass.end();
  device.queue.submit([enc.finish()]);
  const gpuOut = await readFloat32Buffer(device, dstBuf, N);
  const gpuMs = performance.now() - gpuStart;

  xBuf.destroy();
  wBuf.destroy();
  sBuf.destroy();
  bBuf.destroy();
  dstBuf.destroy();
  uniformBuf.destroy();

  const metrics = compareArrays(gpuOut, cpuOut);
  const tolerance = 1e-3;
  return {
    kernel: "qmm_vec",
    label: "Q4 Affine QMM (qmm_vec.comp)",
    shape: `[1, ${K}] × [${N}, ${K}] g${groupSize}`,
    elements: N,
    ...metrics,
    tolerance,
    passed: metrics.maxAbsDiff <= tolerance,
    gpuMs,
    cpuMs,
    faultInjected: faultMode,
  };
}

/**
 * Verifies `qmm_vec.wgsl` at full Qwen3-0.6B LM-head scale (`N = 151936`, `K = 1024`).
 */
export async function verifyQmmVecLmHeadParity(
  device: GPUDevice,
  N = 151936,
  K = 1024,
  groupSize = 64,
  faultMode = false,
): Promise<ParityResult> {
  const result = await verifyQmmVecParity(device, N, K, groupSize, faultMode);
  return {
    ...result,
    label: "Q4 Affine QMM LM-Head (qmm_vec.comp)",
  };
}

/**
 * Verifies fused Gate + Up Q4 QMM + SwiGLU (`qmm_vec_swiglu.wgsl`) against
 * `qmmSwigluQ4Cpu`.
 */
export async function verifyQmmVecSwigluParity(
  device: GPUDevice,
  N = 3072,
  K = 1024,
  groupSize = 64,
  faultMode = false,
): Promise<ParityResult> {
  const x = seededFloat32(K, 401, 0.5);
  const gate = seededQ4Weights(N, K, 402, groupSize);
  const up = seededQ4Weights(N, K, 403, groupSize);

  const cpuStart = performance.now();
  const cpuOut = qmmSwigluQ4Cpu(
    x,
    gate.weightsU32,
    gate.scalesF32,
    gate.biasesF32,
    up.weightsU32,
    up.scalesF32,
    up.biasesF32,
    N,
    K,
    groupSize,
  );
  const cpuMs = performance.now() - cpuStart;

  const module = await createCheckedShaderModule(device, "qmm_vec_swiglu", qmmVecSwigluWgsl);
  const pipeline = device.createComputePipeline({
    label: "qmm_vec_swiglu_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const xBuf = createStorageBuffer(device, x);
  const gwBuf = createStorageBuffer(device, gate.weightsU32);
  const gsBuf = createStorageBuffer(device, gate.scalesF32);
  const gbBuf = createStorageBuffer(device, gate.biasesF32);
  const uwBuf = createStorageBuffer(device, up.weightsU32);
  const usBuf = createStorageBuffer(device, up.scalesF32);
  const ubBuf = createStorageBuffer(device, up.biasesF32);
  const dstBuf = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = new ArrayBuffer(32);
  const u32 = new Uint32Array(paramBuf);
  u32[0] = N;
  u32[1] = K;
  u32[2] = groupSize;
  u32[3] = faultMode ? 1 : 0;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: gwBuf } },
      { binding: 2, resource: { buffer: gsBuf } },
      { binding: 3, resource: { buffer: gbBuf } },
      { binding: 4, resource: { buffer: uwBuf } },
      { binding: 5, resource: { buffer: usBuf } },
      { binding: 6, resource: { buffer: ubBuf } },
      { binding: 7, resource: { buffer: dstBuf } },
      { binding: 8, resource: { buffer: uniformBuf } },
    ],
  });

  const [gx, gy, gz] = qmmDispatchGrid(N);
  const gpuStart = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(gx, gy, gz);
  pass.end();
  device.queue.submit([enc.finish()]);
  const gpuOut = await readFloat32Buffer(device, dstBuf, N);
  const gpuMs = performance.now() - gpuStart;

  xBuf.destroy();
  gwBuf.destroy();
  gsBuf.destroy();
  gbBuf.destroy();
  uwBuf.destroy();
  usBuf.destroy();
  ubBuf.destroy();
  dstBuf.destroy();
  uniformBuf.destroy();

  const metrics = compareArrays(gpuOut, cpuOut);
  const tolerance = 1e-3;
  return {
    kernel: "qmm_vec_swiglu",
    label: "Fused Gate+Up Q4 QMM + SwiGLU (qmm_vec.comp)",
    shape: `[1, ${K}] × 2×[${N}, ${K}] g${groupSize}`,
    elements: N,
    ...metrics,
    tolerance,
    passed: metrics.maxAbsDiff <= tolerance,
    gpuMs,
    cpuMs,
    faultInjected: faultMode,
  };
}

/** Verifies elementwise SwiGLU (`swiglu.wgsl`) against `swigluCpu`. */
export async function verifySwigluParity(
  device: GPUDevice,
  count = 3072,
  faultMode = false,
): Promise<ParityResult> {
  const gate = seededFloat32(count, 501, 2.0);
  const up = seededFloat32(count, 502, 2.0);

  const cpuStart = performance.now();
  const cpuOut = swigluCpu(gate, up);
  const cpuMs = performance.now() - cpuStart;

  const module = await createCheckedShaderModule(device, "swiglu", swigluWgsl);
  const pipeline = device.createComputePipeline({
    label: "swiglu_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const gateBuf = createStorageBuffer(device, gate);
  const upBuf = createStorageBuffer(device, up);
  const dstBuf = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = new ArrayBuffer(16);
  const u32 = new Uint32Array(paramBuf);
  u32[0] = count;
  u32[1] = faultMode ? 1 : 0;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gateBuf } },
      { binding: 1, resource: { buffer: upBuf } },
      { binding: 2, resource: { buffer: dstBuf } },
      { binding: 3, resource: { buffer: uniformBuf } },
    ],
  });

  const gpuStart = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(count / 256), 1, 1);
  pass.end();
  device.queue.submit([enc.finish()]);
  const gpuOut = await readFloat32Buffer(device, dstBuf, count);
  const gpuMs = performance.now() - gpuStart;

  gateBuf.destroy();
  upBuf.destroy();
  dstBuf.destroy();
  uniformBuf.destroy();

  const metrics = compareArrays(gpuOut, cpuOut);
  const tolerance = 1e-4;
  return {
    kernel: "swiglu",
    label: "SwiGLU Activation (swiglu.comp)",
    shape: `[${count}]`,
    elements: count,
    ...metrics,
    tolerance,
    passed: metrics.maxAbsDiff <= tolerance,
    gpuMs,
    cpuMs,
    faultInjected: faultMode,
  };
}

/** Verifies MLX affine 4-bit embedding lookup (`elem.wgsl` `embed_q4`) against `embedQ4Cpu`. */
export async function verifyEmbedQ4Parity(
  device: GPUDevice,
  vocabSize = 4096,
  hiddenSize = 1024,
  tokenId = 1337,
  groupSize = 64,
  faultMode = false,
): Promise<ParityResult> {
  const { weightsU32, scalesF32, biasesF32 } = seededQ4Weights(
    vocabSize,
    hiddenSize,
    601,
    groupSize,
  );

  const cpuStart = performance.now();
  const cpuOut = embedQ4Cpu(tokenId, weightsU32, scalesF32, biasesF32, hiddenSize, groupSize);
  const cpuMs = performance.now() - cpuStart;

  const module = await createCheckedShaderModule(device, "elem_embed_q4", elemWgsl);
  const pipeline = device.createComputePipeline({
    label: "embed_q4_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "embed_q4" },
  });

  const wBuf = createStorageBuffer(device, weightsU32);
  const sBuf = createStorageBuffer(device, scalesF32);
  const bBuf = createStorageBuffer(device, biasesF32);
  const dstBuf = device.createBuffer({
    size: hiddenSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = new ArrayBuffer(16);
  const u32 = new Uint32Array(paramBuf);
  u32[0] = tokenId;
  u32[1] = hiddenSize;
  u32[2] = groupSize;
  u32[3] = faultMode ? 1 : 0;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: wBuf } },
      { binding: 1, resource: { buffer: sBuf } },
      { binding: 2, resource: { buffer: bBuf } },
      { binding: 3, resource: { buffer: dstBuf } },
      { binding: 4, resource: { buffer: uniformBuf } },
    ],
  });

  const gpuStart = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(hiddenSize / 256), 1, 1);
  pass.end();
  device.queue.submit([enc.finish()]);
  const gpuOut = await readFloat32Buffer(device, dstBuf, hiddenSize);
  const gpuMs = performance.now() - gpuStart;

  wBuf.destroy();
  sBuf.destroy();
  bBuf.destroy();
  dstBuf.destroy();
  uniformBuf.destroy();

  const metrics = compareArrays(gpuOut, cpuOut);
  const tolerance = 1e-5;
  return {
    kernel: "embed_q4",
    label: "Q4 Token Embedding Lookup (elem.wgsl)",
    shape: `vocab=${vocabSize} tok=${tokenId} → [${hiddenSize}]`,
    elements: hiddenSize,
    ...metrics,
    tolerance,
    passed: metrics.maxAbsDiff <= tolerance,
    gpuMs,
    cpuMs,
    faultInjected: faultMode,
  };
}

/** Verifies elementwise residual addition (`elem.wgsl` `residual_add`) against `residualAddCpu`. */
export async function verifyResidualAddParity(
  device: GPUDevice,
  count = 1024,
  faultMode = false,
): Promise<ParityResult> {
  const a = seededFloat32(count, 611, 2.0);
  const b = seededFloat32(count, 612, 2.0);

  const cpuStart = performance.now();
  const cpuOut = residualAddCpu(a, b);
  const cpuMs = performance.now() - cpuStart;

  const module = await createCheckedShaderModule(device, "elem_residual_add", elemWgsl);
  const pipeline = device.createComputePipeline({
    label: "residual_add_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "residual_add" },
  });

  const aBuf = createStorageBuffer(device, a);
  const bBuf = createStorageBuffer(device, b);
  const dstBuf = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = new ArrayBuffer(16);
  const u32 = new Uint32Array(paramBuf);
  u32[0] = count;
  u32[1] = faultMode ? 1 : 0;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 5, resource: { buffer: aBuf } },
      { binding: 6, resource: { buffer: bBuf } },
      { binding: 7, resource: { buffer: dstBuf } },
      { binding: 8, resource: { buffer: uniformBuf } },
    ],
  });

  const gpuStart = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(count / 256), 1, 1);
  pass.end();
  device.queue.submit([enc.finish()]);
  const gpuOut = await readFloat32Buffer(device, dstBuf, count);
  const gpuMs = performance.now() - gpuStart;

  aBuf.destroy();
  bBuf.destroy();
  dstBuf.destroy();
  uniformBuf.destroy();

  const metrics = compareArrays(gpuOut, cpuOut);
  const tolerance = 1e-6;
  return {
    kernel: "residual_add",
    label: "Residual Add (elem.wgsl)",
    shape: `[${count}]`,
    elements: count,
    ...metrics,
    tolerance,
    passed: metrics.maxAbsDiff <= tolerance,
    gpuMs,
    cpuMs,
    faultInjected: faultMode,
  };
}

/**
 * Verifies single-token KV-cache scatter write (`elem.wgsl` `kv_cache_write`)
 * into `[kvHeads, maxSeq, headDim]` against `kvCacheWriteCpu`.
 */
export async function verifyKvCacheWriteParity(
  device: GPUDevice,
  kvHeads = 8,
  headDim = 128,
  maxSeq = 64,
  pos = 17,
  faultMode = false,
): Promise<ParityResult> {
  const tokenElems = kvHeads * headDim;
  const cacheElems = kvHeads * maxSeq * headDim;
  const kIn = seededFloat32(tokenElems, 621, 1.0);
  const vIn = seededFloat32(tokenElems, 622, 1.0);
  const kCacheInit = seededFloat32(cacheElems, 623, 0.5);
  const vCacheInit = seededFloat32(cacheElems, 624, 0.5);

  const cpuStart = performance.now();
  const cpuRes = kvCacheWriteCpu(kIn, vIn, kCacheInit, vCacheInit, kvHeads, headDim, maxSeq, pos);
  const cpuMs = performance.now() - cpuStart;

  const module = await createCheckedShaderModule(device, "elem_kv_cache_write", elemWgsl);
  const pipeline = device.createComputePipeline({
    label: "kv_cache_write_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "kv_cache_write" },
  });

  const kInBuf = createStorageBuffer(device, kIn);
  const vInBuf = createStorageBuffer(device, vIn);
  const kCacheBuf = createStorageBuffer(
    device,
    kCacheInit,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );
  const vCacheBuf = createStorageBuffer(
    device,
    vCacheInit,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  );

  const paramBuf = new ArrayBuffer(32);
  const u32 = new Uint32Array(paramBuf);
  u32[0] = kvHeads;
  u32[1] = headDim;
  u32[2] = maxSeq;
  u32[3] = pos;
  u32[4] = faultMode ? 1 : 0;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 9, resource: { buffer: kInBuf } },
      { binding: 10, resource: { buffer: vInBuf } },
      { binding: 11, resource: { buffer: kCacheBuf } },
      { binding: 12, resource: { buffer: vCacheBuf } },
      { binding: 13, resource: { buffer: uniformBuf } },
    ],
  });

  const gpuStart = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(tokenElems / 256), 1, 1);
  pass.end();
  device.queue.submit([enc.finish()]);
  const gpuK = await readFloat32Buffer(device, kCacheBuf, cacheElems);
  const gpuV = await readFloat32Buffer(device, vCacheBuf, cacheElems);
  const gpuMs = performance.now() - gpuStart;

  kInBuf.destroy();
  vInBuf.destroy();
  kCacheBuf.destroy();
  vCacheBuf.destroy();
  uniformBuf.destroy();

  // Compare both full K and V caches so out-of-position corruption would also fail.
  const gpuCombined = new Float32Array(cacheElems * 2);
  gpuCombined.set(gpuK, 0);
  gpuCombined.set(gpuV, cacheElems);
  const cpuCombined = new Float32Array(cacheElems * 2);
  cpuCombined.set(cpuRes.kCache, 0);
  cpuCombined.set(cpuRes.vCache, cacheElems);

  const metrics = compareArrays(gpuCombined, cpuCombined);
  const tolerance = 1e-6;
  return {
    kernel: "kv_cache_write",
    label: "KV Cache Write (elem.wgsl)",
    shape: `[${kvHeads}, ${maxSeq}, ${headDim}] @ pos=${pos}`,
    elements: cacheElems * 2,
    ...metrics,
    tolerance,
    passed: metrics.maxAbsDiff <= tolerance,
    gpuMs,
    cpuMs,
    faultInjected: faultMode,
  };
}

/**
 * Verifies 2-stage deterministic argmax (`elem.wgsl` `argmax_p1` + `argmax_p2`)
 * at full Qwen3 vocab size (`151936`), including lowest-index tie-breaking.
 */
export async function verifyArgmaxParity(
  device: GPUDevice,
  count = 151936,
  faultMode = false,
): Promise<ParityResult> {
  const logits = seededFloat32(count, 631, 12.0);
  // Plant an exact tie at two distant indices to verify deterministic lowest-index tie-breaking.
  if (count > 1024) {
    const plantedPeak = 25.5;
    logits[777] = plantedPeak;
    logits[count - 13] = plantedPeak;
  }

  const cpuStart = performance.now();
  const cpuRes = argmaxCpu(logits);
  const cpuMs = performance.now() - cpuStart;

  const numBlocks = argmaxNumBlocks(count);
  const module = await createCheckedShaderModule(device, "elem_argmax", elemWgsl);
  const p1Pipe = device.createComputePipeline({
    label: "argmax_p1_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "argmax_p1" },
  });
  const p2Pipe = device.createComputePipeline({
    label: "argmax_p2_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "argmax_p2" },
  });

  const logitsBuf = createStorageBuffer(device, logits);
  const partialValsBuf = device.createBuffer({
    size: numBlocks * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const partialIdxsBuf = device.createBuffer({
    size: numBlocks * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const outIdxBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const outValBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = new ArrayBuffer(16);
  const u32 = new Uint32Array(paramBuf);
  u32[0] = count;
  u32[1] = numBlocks;
  u32[2] = faultMode ? 1 : 0;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg1 = device.createBindGroup({
    layout: p1Pipe.getBindGroupLayout(0),
    entries: [
      { binding: 14, resource: { buffer: logitsBuf } },
      { binding: 15, resource: { buffer: partialValsBuf } },
      { binding: 16, resource: { buffer: partialIdxsBuf } },
      { binding: 17, resource: { buffer: uniformBuf } },
    ],
  });
  const bg2 = device.createBindGroup({
    layout: p2Pipe.getBindGroupLayout(0),
    entries: [
      { binding: 17, resource: { buffer: uniformBuf } },
      { binding: 18, resource: { buffer: partialValsBuf } },
      { binding: 19, resource: { buffer: partialIdxsBuf } },
      { binding: 20, resource: { buffer: outIdxBuf } },
      { binding: 21, resource: { buffer: outValBuf } },
    ],
  });

  const gpuStart = performance.now();
  const enc = device.createCommandEncoder();
  const pass1 = enc.beginComputePass();
  pass1.setPipeline(p1Pipe);
  pass1.setBindGroup(0, bg1);
  pass1.dispatchWorkgroups(numBlocks, 1, 1);
  pass1.end();
  const pass2 = enc.beginComputePass();
  pass2.setPipeline(p2Pipe);
  pass2.setBindGroup(0, bg2);
  pass2.dispatchWorkgroups(1, 1, 1);
  pass2.end();
  device.queue.submit([enc.finish()]);
  const [gpuIdxArr, gpuValArr] = await Promise.all([
    readUint32Buffer(device, outIdxBuf, 1),
    readFloat32Buffer(device, outValBuf, 1),
  ]);
  const gpuMs = performance.now() - gpuStart;

  logitsBuf.destroy();
  partialValsBuf.destroy();
  partialIdxsBuf.destroy();
  outIdxBuf.destroy();
  outValBuf.destroy();
  uniformBuf.destroy();

  const gpuIdx = gpuIdxArr[0];
  const gpuVal = gpuValArr[0];
  const idxDiff = Math.abs(gpuIdx - cpuRes.index);
  const valDiff = Math.abs(gpuVal - cpuRes.value);
  const maxAbsDiff = Math.max(idxDiff, valDiff);

  return {
    kernel: "argmax",
    label: "2-Stage Argmax (elem.wgsl)",
    shape: `[${count}] (${numBlocks} blocks → 1)`,
    elements: count,
    maxAbsDiff,
    meanAbsDiff: (idxDiff + valDiff) / 2,
    cosineSimilarity: gpuIdx === cpuRes.index && valDiff <= 1e-6 ? 1.0 : 0.0,
    diffSample: [gpuIdx - cpuRes.index, gpuVal - cpuRes.value],
    gpuSample: [gpuIdx, gpuVal],
    cpuSample: [cpuRes.index, cpuRes.value],
    tolerance: 0,
    passed: gpuIdx === cpuRes.index && valDiff <= 1e-6,
    gpuMs,
    cpuMs,
    faultInjected: faultMode,
  };
}

export interface KernelParityOptions {
  faultMode?: boolean;
  /** Also test `qmm_vec` at full Qwen3 vocab size (`N = 151936`, `K = 1024`, ~78 MB). */
  includeLmHead?: boolean;
}

/** Runs every compute-kernel parity test in this module sequentially on `device`. */
export async function verifyAllKernelsParity(
  device: GPUDevice,
  options: KernelParityOptions = {},
): Promise<ParityResult[]> {
  const faultMode = options.faultMode ?? false;
  const includeLmHead = options.includeLmHead ?? true;
  const results: ParityResult[] = [];
  results.push(await verifyFastRopeParity(device, 16, 128, 42, 1_000_000, faultMode));
  results.push(await verifyFastRopeNormParity(device, 16, 128, 42, 1e-6, 1_000_000, faultMode));
  results.push(await verifyQmmVecParity(device, 3072, 1024, 64, faultMode));
  if (includeLmHead) {
    results.push(await verifyQmmVecLmHeadParity(device, 151936, 1024, 64, faultMode));
  }
  results.push(await verifyQmmVecSwigluParity(device, 3072, 1024, 64, faultMode));
  results.push(await verifySwigluParity(device, 3072, faultMode));
  results.push(await verifyEmbedQ4Parity(device, 4096, 1024, 1337, 64, faultMode));
  results.push(await verifyResidualAddParity(device, 1024, faultMode));
  results.push(await verifyKvCacheWriteParity(device, 8, 128, 64, 17, faultMode));
  results.push(await verifyArgmaxParity(device, 151936, faultMode));
  return results;
}
