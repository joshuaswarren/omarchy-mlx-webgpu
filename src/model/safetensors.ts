/**
 * Zero-dependency Safetensors parser and WebGPU tensor uploader for MLX 4-bit models.
 * Handles U32 packed 4-bit weights (uploaded directly as GPUBuffer) and BF16/F16/F32
 * scales, biases, and norm weights (unpacked to Float32 cleanly on upload).
 */

export interface TensorMeta {
  dtype: "U32" | "BF16" | "F16" | "F32" | string;
  shape: number[];
  data_offsets: [number, number];
}

export interface ParsedSafetensors {
  header: Record<string, TensorMeta>;
  dataOffset: number;
  buffer: ArrayBuffer;
}

export function parseSafetensors(buffer: ArrayBuffer): ParsedSafetensors {
  if (buffer.byteLength < 8) {
    throw new Error("Safetensors buffer too small");
  }
  const view = new DataView(buffer);
  const headerLenLo = view.getUint32(0, true);
  const headerLenHi = view.getUint32(4, true);
  if (headerLenHi !== 0) {
    throw new Error("Safetensors header length exceeds 4 GiB");
  }
  const headerLen = headerLenLo;
  const headerBytes = new Uint8Array(buffer, 8, headerLen);
  const rawHeader = JSON.parse(new TextDecoder().decode(headerBytes)) as Record<string, unknown>;
  const header: Record<string, TensorMeta> = {};
  for (const [k, v] of Object.entries(rawHeader)) {
    if (k === "__metadata__") continue;
    header[k] = v as TensorMeta;
  }
  return { header, dataOffset: 8 + headerLen, buffer };
}

/** Fast bulk BF16 (uint16[]) -> Float32Array conversion (exact bit shift, alignment-safe). */
export function bf16BytesToFloat32(buffer: ArrayBuffer, byteOffset: number, byteLength: number): Float32Array {
  const count = byteLength >>> 1;
  const src =
    (byteOffset & 1) === 0
      ? new Uint16Array(buffer, byteOffset, count)
      : new Uint16Array(buffer.slice(byteOffset, byteOffset + byteLength));
  const u32 = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    u32[i] = src[i] << 16;
  }
  return new Float32Array(u32.buffer);
}

/** IEEE-754 float16 (uint16[]) -> Float32Array conversion (alignment-safe). */
export function f16BytesToFloat32(buffer: ArrayBuffer, byteOffset: number, byteLength: number): Float32Array {
  const count = byteLength >>> 1;
  const src =
    (byteOffset & 1) === 0
      ? new Uint16Array(buffer, byteOffset, count)
      : new Uint16Array(buffer.slice(byteOffset, byteOffset + byteLength));
  const out = new Float32Array(count);
  const scratch = new ArrayBuffer(4);
  const u32v = new Uint32Array(scratch);
  const f32v = new Float32Array(scratch);
  for (let i = 0; i < count; i++) {
    const h = src[i];
    const sign = (h & 0x8000) << 16;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x03ff;
    if (exp === 0) {
      if (frac === 0) {
        u32v[0] = sign;
      } else {
        let f = frac;
        let e = -14;
        while ((f & 0x0400) === 0) {
          f <<= 1;
          e--;
        }
        f &= 0x03ff;
        u32v[0] = sign | ((e + 127) << 23) | (f << 13);
      }
    } else if (exp === 31) {
      u32v[0] = sign | 0x7f800000 | (frac << 13);
    } else {
      u32v[0] = sign | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    out[i] = f32v[0];
  }
  return out;
}

export function getTensorAsFloat32(st: ParsedSafetensors, name: string): Float32Array {
  const meta = st.header[name];
  if (!meta) throw new Error(`Missing tensor: ${name}`);
  const start = st.dataOffset + meta.data_offsets[0];
  const len = meta.data_offsets[1] - meta.data_offsets[0];
  if (meta.dtype === "BF16") {
    return bf16BytesToFloat32(st.buffer, start, len);
  }
  if (meta.dtype === "F16") {
    return f16BytesToFloat32(st.buffer, start, len);
  }
  if (meta.dtype === "F32") {
    return new Float32Array(st.buffer.slice(start, start + len));
  }
  throw new Error(`Unsupported float dtype ${meta.dtype} for ${name}`);
}

export function getTensorU32Slice(st: ParsedSafetensors, name: string): Uint8Array {
  const meta = st.header[name];
  if (!meta) throw new Error(`Missing tensor: ${name}`);
  if (meta.dtype !== "U32") {
    throw new Error(`Expected U32 tensor for ${name}, got ${meta.dtype}`);
  }
  const start = st.dataOffset + meta.data_offsets[0];
  const len = meta.data_offsets[1] - meta.data_offsets[0];
  return new Uint8Array(st.buffer, start, len);
}

export function uploadRawBuffer(
  device: GPUDevice,
  bytes: Uint8Array,
  label: string,
): GPUBuffer {
  const alignedSize = Math.max(4, Math.ceil(bytes.byteLength / 4) * 4);
  const buf = device.createBuffer({
    label,
    size: alignedSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return buf;
}

export function uploadFloat32Tensor(
  device: GPUDevice,
  st: ParsedSafetensors,
  name: string,
): GPUBuffer {
  const f32 = getTensorAsFloat32(st, name);
  return uploadRawBuffer(device, new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength), name);
}

export function uploadU32Tensor(
  device: GPUDevice,
  st: ParsedSafetensors,
  name: string,
): GPUBuffer {
  const u8 = getTensorU32Slice(st, name);
  return uploadRawBuffer(device, u8, name);
}
