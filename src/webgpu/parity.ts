import fastNormWgsl from "../kernels/wgsl/fast_norm.wgsl?raw";
import sdpaOnlineWgsl from "../kernels/wgsl/sdpa_decode_online.wgsl?raw";
import sdpaSplitP1Wgsl from "../kernels/wgsl/sdpa_decode_split_p1.wgsl?raw";
import sdpaSplitP2Wgsl from "../kernels/wgsl/sdpa_decode_split_p2.wgsl?raw";
import type { ParityResult } from "../kernels/types";
import { compareArrays, rmsNormCpu, sdpaDecodeCpu, seededFloat32 } from "../reference/cpu";
import {
  createCheckedShaderModule,
  createStorageBuffer,
  createUniformBuffer,
  readFloat32Buffer,
} from "./device";

/**
 * Runs real omarchy-mlx WGSL kernels on the visitor's WebGPU device and checks
 * numerical parity against the in-page CPU reference.
 */
export async function verifyFastNormParity(
  device: GPUDevice,
  rows = 16,
  rowLength = 1024,
  eps = 1e-6,
  faultMode = false,
): Promise<ParityResult> {
  const input = seededFloat32(rows * rowLength, 42, 1.5);
  const weight = seededFloat32(rowLength, 108, 1.0);

  const cpuStart = performance.now();
  const cpuOut = rmsNormCpu(input, weight, rows, rowLength, eps);
  const cpuMs = performance.now() - cpuStart;

  const module = await createCheckedShaderModule(device, "fast_norm", fastNormWgsl);
  const pipeline = device.createComputePipeline({
    label: "fast_norm_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const inBuf = createStorageBuffer(device, input, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const wBuf = createStorageBuffer(device, weight, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outBuf = device.createBuffer({
    size: rows * rowLength * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = new ArrayBuffer(32);
  const u32 = new Uint32Array(paramBuf);
  const f32 = new Float32Array(paramBuf);
  u32[0] = rows;
  u32[1] = rowLength;
  u32[2] = 0;
  u32[3] = faultMode ? 1 : 0;
  f32[4] = eps;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: uniformBuf } },
    ],
  });

  const gpuStart = performance.now();
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(rows, 1, 1);
  pass.end();
  device.queue.submit([enc.finish()]);
  const gpuOut = await readFloat32Buffer(device, outBuf, rows * rowLength);
  const gpuMs = performance.now() - gpuStart;

  inBuf.destroy();
  wBuf.destroy();
  outBuf.destroy();
  uniformBuf.destroy();

  const metrics = compareArrays(gpuOut, cpuOut);
  const tolerance = 1e-4;
  return {
    kernel: "fast_norm",
    label: "RMSNorm (fast_norm.comp)",
    shape: `[${rows}, ${rowLength}]`,
    elements: rows * rowLength,
    ...metrics,
    tolerance,
    passed: metrics.maxAbsDiff <= tolerance,
    gpuMs,
    cpuMs,
    faultInjected: faultMode,
  };
}

/**
 * Verifies PR #997 one-dispatch online-softmax decode SDPA (HD=128) against CPU reference.
 */
export async function verifySdpaOnlineParity(
  device: GPUDevice,
  heads = 16,
  kvHeads = 8,
  kLen = 192,
  faultMode = false,
): Promise<ParityResult> {
  const headDim = 128;
  const maxSeq = kLen;
  const scale = 1.0 / Math.sqrt(headDim);

  const q = seededFloat32(heads * headDim, 701, 0.5);
  const k = seededFloat32(kvHeads * maxSeq * headDim, 702, 0.5);
  const v = seededFloat32(kvHeads * maxSeq * headDim, 703, 0.5);

  const cpuStart = performance.now();
  const cpuOut = sdpaDecodeCpu(q, k, v, heads, kvHeads, kLen, maxSeq, headDim, scale);
  const cpuMs = performance.now() - cpuStart;

  const module = await createCheckedShaderModule(device, "sdpa_decode_online", sdpaOnlineWgsl);
  const pipeline = device.createComputePipeline({
    label: "sdpa_decode_online_pipeline",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const qBuf = createStorageBuffer(device, q);
  const kBuf = createStorageBuffer(device, k);
  const vBuf = createStorageBuffer(device, v);
  const outBuf = device.createBuffer({
    size: heads * headDim * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = new ArrayBuffer(32);
  const u32 = new Uint32Array(paramBuf);
  const f32 = new Float32Array(paramBuf);
  u32[0] = heads;
  u32[1] = kvHeads;
  u32[2] = kLen;
  u32[3] = maxSeq;
  f32[4] = scale;
  u32[5] = faultMode ? 1 : 0;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: qBuf } },
      { binding: 1, resource: { buffer: kBuf } },
      { binding: 2, resource: { buffer: vBuf } },
      { binding: 3, resource: { buffer: outBuf } },
      { binding: 4, resource: { buffer: uniformBuf } },
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
  const gpuOut = await readFloat32Buffer(device, outBuf, heads * headDim);
  const gpuMs = performance.now() - gpuStart;

  qBuf.destroy();
  kBuf.destroy();
  vBuf.destroy();
  outBuf.destroy();
  uniformBuf.destroy();

  const metrics = compareArrays(gpuOut, cpuOut);
  const tolerance = 1e-4;
  return {
    kernel: "sdpa_decode_online",
    label: "PR #997 1-Dispatch Online SDPA (sdpa_decode_online.comp)",
    shape: `Q[${heads},128] × KV[${kvHeads},${kLen},128]`,
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
 * Verifies the 2-dispatch split-K decode SDPA baseline (P1 + P2) against CPU reference.
 */
export async function verifySdpaSplitParity(
  device: GPUDevice,
  heads = 16,
  kvHeads = 8,
  kLen = 192,
  faultMode = false,
): Promise<ParityResult> {
  const headDim = 128;
  const maxSeq = kLen;
  const bk = 64;
  const nblocks = Math.ceil(kLen / bk);
  const scale = 1.0 / Math.sqrt(headDim);

  const q = seededFloat32(heads * headDim, 701, 0.5);
  const k = seededFloat32(kvHeads * maxSeq * headDim, 702, 0.5);
  const v = seededFloat32(kvHeads * maxSeq * headDim, 703, 0.5);

  const cpuStart = performance.now();
  const cpuOut = sdpaDecodeCpu(q, k, v, heads, kvHeads, kLen, maxSeq, headDim, scale);
  const cpuMs = performance.now() - cpuStart;

  const p1Mod = await createCheckedShaderModule(device, "sdpa_split_p1", sdpaSplitP1Wgsl);
  const p2Mod = await createCheckedShaderModule(device, "sdpa_split_p2", sdpaSplitP2Wgsl);
  const p1Pipe = device.createComputePipeline({
    layout: "auto",
    compute: { module: p1Mod, entryPoint: "main" },
  });
  const p2Pipe = device.createComputePipeline({
    layout: "auto",
    compute: { module: p2Mod, entryPoint: "main" },
  });

  const qBuf = createStorageBuffer(device, q);
  const kBuf = createStorageBuffer(device, k);
  const vBuf = createStorageBuffer(device, v);
  const scratchBuf = device.createBuffer({
    size: heads * nblocks * (headDim + 4) * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const outBuf = device.createBuffer({
    size: heads * headDim * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = new ArrayBuffer(32);
  const u32 = new Uint32Array(paramBuf);
  const f32 = new Float32Array(paramBuf);
  u32[0] = heads;
  u32[1] = kvHeads;
  u32[2] = kLen;
  u32[3] = maxSeq;
  f32[4] = scale;
  u32[5] = faultMode ? 1 : 0;
  u32[6] = nblocks;
  const uniformBuf = createUniformBuffer(device, new Uint8Array(paramBuf));

  const bg1 = device.createBindGroup({
    layout: p1Pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: qBuf } },
      { binding: 1, resource: { buffer: kBuf } },
      { binding: 2, resource: { buffer: vBuf } },
      { binding: 3, resource: { buffer: scratchBuf } },
      { binding: 4, resource: { buffer: uniformBuf } },
    ],
  });
  const bg2 = device.createBindGroup({
    layout: p2Pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: scratchBuf } },
      { binding: 1, resource: { buffer: outBuf } },
      { binding: 2, resource: { buffer: uniformBuf } },
    ],
  });

  const gpuStart = performance.now();
  const enc = device.createCommandEncoder();
  const pass1 = enc.beginComputePass();
  pass1.setPipeline(p1Pipe);
  pass1.setBindGroup(0, bg1);
  pass1.dispatchWorkgroups(heads, nblocks, 1);
  pass1.end();
  const pass2 = enc.beginComputePass();
  pass2.setPipeline(p2Pipe);
  pass2.setBindGroup(0, bg2);
  pass2.dispatchWorkgroups(heads, 1, 1);
  pass2.end();
  device.queue.submit([enc.finish()]);
  const gpuOut = await readFloat32Buffer(device, outBuf, heads * headDim);
  const gpuMs = performance.now() - gpuStart;

  qBuf.destroy();
  kBuf.destroy();
  vBuf.destroy();
  scratchBuf.destroy();
  outBuf.destroy();
  uniformBuf.destroy();

  const metrics = compareArrays(gpuOut, cpuOut);
  const tolerance = 1e-4;
  return {
    kernel: "sdpa_decode_split_p1",
    label: "Baseline 2-Dispatch Split-K SDPA (p1 + p2)",
    shape: `Q[${heads},128] × KV[${kvHeads},${kLen},128] (${nblocks} blocks)`,
    elements: heads * headDim,
    ...metrics,
    tolerance,
    passed: metrics.maxAbsDiff <= tolerance,
    gpuMs,
    cpuMs,
    faultInjected: faultMode,
  };
}
