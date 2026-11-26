import type { GpuAdapterInfo } from "../kernels/types";

export interface WebGpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  info: GpuAdapterInfo;
}

export class WebGpuUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WebGpuUnavailableError";
  }
}

export async function initWebGpu(): Promise<WebGpuContext> {
  if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
    throw new WebGpuUnavailableError(
      "WebGPU is not available in this browser. Open in Chrome/Edge 113+ or Safari 26+.",
    );
  }

  let adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    adapter = await navigator.gpu.requestAdapter({ forceFallbackAdapter: true });
  }
  if (!adapter) {
    throw new WebGpuUnavailableError("No WebGPU adapter found on this device.");
  }

  const requiredFeatures: GPUFeatureName[] = [];
  const hasTimestampQuery = adapter.features.has("timestamp-query");
  if (hasTimestampQuery) {
    requiredFeatures.push("timestamp-query");
  }
  const hasShaderF16 = adapter.features.has("shader-f16");
  if (hasShaderF16) {
    requiredFeatures.push("shader-f16");
  }
  const hasSubgroups = adapter.features.has("subgroups" as GPUFeatureName);
  if (hasSubgroups) {
    requiredFeatures.push("subgroups" as GPUFeatureName);
  }

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(
        adapter.limits.maxStorageBufferBindingSize,
        256 * 1024 * 1024,
      ),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, 512 * 1024 * 1024),
    },
  });

  const adapterInfo = adapter.info;
  const info: GpuAdapterInfo = {
    vendor: adapterInfo.vendor || "unknown",
    architecture: adapterInfo.architecture || "unknown",
    device: adapterInfo.device || "WebGPU Device",
    description: adapterInfo.description || `${adapterInfo.vendor || "GPU"} WebGPU`,
    maxBufferSize: device.limits.maxBufferSize,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
    hasTimestampQuery,
    hasShaderF16,
    hasSubgroups,
  };

  return { adapter, device, info };
}

export async function createCheckedShaderModule(
  device: GPUDevice,
  label: string,
  code: string,
): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length > 0) {
    const details = errors
      .map((e) => `${label}:${e.lineNum}:${e.linePos} ${e.message}`)
      .join("\n");
    throw new Error(`WGSL compilation failed:\n${details}`);
  }
  return module;
}

export function createStorageBuffer(
  device: GPUDevice,
  data: ArrayBufferView,
  usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  label?: string,
): GPUBuffer {
  const alignedSize = Math.max(4, Math.ceil(data.byteLength / 4) * 4);
  const buffer = device.createBuffer({
    label,
    size: alignedSize,
    usage,
  });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

export function createUniformBuffer(
  device: GPUDevice,
  data: ArrayBufferView,
  label?: string,
): GPUBuffer {
  const alignedSize = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
  const buffer = device.createBuffer({
    label,
    size: alignedSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

export async function readFloat32Buffer(
  device: GPUDevice,
  src: GPUBuffer,
  elementCount: number,
): Promise<Float32Array> {
  const byteLength = elementCount * 4;
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return copy;
}
