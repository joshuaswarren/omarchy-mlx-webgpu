import { describe, expect, it } from "bun:test";
import { parseSafetensors, getTensorAsFloat32, getTensorU32Slice } from "../src/model/safetensors";
import { Qwen3Tokenizer } from "../src/model/tokenizer";
import {
  bf16ToF32,
  compareArrays,
  embedQ4Cpu,
  qmmQ4Cpu,
  rmsNormCpu,
  ropeCpu,
  sdpaDecodeCpu,
  seededFloat32,
  swigluCpu,
} from "../src/reference/cpu";

describe("CPU reference kernels", () => {
  it("RMSNorm normalizes unit vector with weight 1", () => {
    const x = new Float32Array([3, 4]);
    const w = new Float32Array([1, 1]);
    const y = rmsNormCpu(x, w, 1, 2, 0);
    const rms = Math.sqrt(12.5);
    expect(Math.abs(y[0] - 3 / rms)).toBeLessThan(1e-6);
    expect(Math.abs(y[1] - 4 / rms)).toBeLessThan(1e-6);
  });

  it("SDPA matches single-key identity (returns V[0])", () => {
    const q = seededFloat32(128, 1);
    const k = seededFloat32(128, 2);
    const v = seededFloat32(128, 3);
    const out = sdpaDecodeCpu(q, k, v, 1, 1, 1, 1, 128, 1 / Math.sqrt(128));
    expect(compareArrays(out, v).maxAbsDiff).toBeLessThan(1e-6);
  });

  it("RoPE at position 0 is identity", () => {
    const x = seededFloat32(256, 7);
    const out = ropeCpu(x, 2, 128, 0, 1_000_000);
    expect(compareArrays(out, x).maxAbsDiff).toBeLessThan(1e-6);
  });

  it("Q4 affine QMM matches manual nibble decode", () => {
    const x = new Float32Array(64).fill(2.0);
    const w = new Uint32Array(8).fill(0x33333333);
    const s = new Float32Array([0.5]);
    const b = new Float32Array([-1.0]);
    const out = qmmQ4Cpu(x, w, s, b, 1, 64, 64);
    expect(Math.abs(out[0] - 64.0)).toBeLessThan(1e-5);
    const emb = embedQ4Cpu(0, w, s, b, 64, 64);
    expect(Math.abs(emb[0] - 0.5)).toBeLessThan(1e-6);
  });

  it("SwiGLU and BF16 decode are exact", () => {
    expect(Math.abs(bf16ToF32(0x3f80) - 1.0)).toBe(0);
    const out = swigluCpu(new Float32Array([0]), new Float32Array([4]));
    expect(Math.abs(out[0])).toBeLessThan(1e-6);
  });

  it("compareArrays returns cosine 0 when gpu output is all zeros", () => {
    const m = compareArrays(new Float32Array([0, 0, 0]), new Float32Array([1, 2, 3]));
    expect(m.cosineSimilarity).toBe(0);
  });
});

describe("Safetensors parser & Qwen3 BPE tokenizer", () => {
  it("parses synthetic safetensors with BF16 and U32 tensors", () => {
    const headerJson = JSON.stringify({
      "w.bf16": { dtype: "BF16", shape: [2], data_offsets: [0, 4] },
      "w.u32": { dtype: "U32", shape: [1], data_offsets: [4, 8] },
    });
    const headerBytes = new TextEncoder().encode(headerJson);
    const buf = new ArrayBuffer(8 + headerBytes.byteLength + 8);
    const dv = new DataView(buf);
    dv.setUint32(0, headerBytes.byteLength, true);
    dv.setUint32(4, 0, true);
    new Uint8Array(buf, 8, headerBytes.byteLength).set(headerBytes);
    const dataStart = 8 + headerBytes.byteLength;
    dv.setUint16(dataStart + 0, 0x3f80, true); // 1.0 in bf16
    dv.setUint16(dataStart + 2, 0xc000, true); // -2.0 in bf16
    dv.setUint32(dataStart + 4, 0x12345678, true);

    const parsed = parseSafetensors(buf);
    const f32 = getTensorAsFloat32(parsed, "w.bf16");
    expect(f32[0]).toBe(1.0);
    expect(f32[1]).toBe(-2.0);
    const u8 = getTensorU32Slice(parsed, "w.u32");
    expect(new DataView(u8.buffer, u8.byteOffset, 4).getUint32(0, true)).toBe(0x12345678);
  });

  it("round-trips BPE encoding and special tokens", () => {
    const vocab = { H: 0, i: 1, Hi: 2, "Ġ": 3, "!": 4, "Ċ": 5 };
    const merges = "#version: 0.2\nH i\n";
    const added = { "<|im_start|>": 100, "<|im_end|>": 101 };
    const tok = new Qwen3Tokenizer(vocab, merges, added);
    const ids = tok.encode("<|im_start|>Hi!<|im_end|>");
    expect(ids).toEqual([100, 2, 4, 101]);
    expect(tok.decode(ids, true)).toBe("Hi!");
    expect(tok.decode(ids, false)).toBe("<|im_start|>Hi!<|im_end|>");
  });
});
