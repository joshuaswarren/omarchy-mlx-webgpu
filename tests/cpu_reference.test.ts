import { describe, expect, it } from "bun:test";
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
    // rms = sqrt((9 + 16)/2) = sqrt(12.5)
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
    // K = 64, N = 1, all nibbles = 3, scale = 0.5, bias = -1.0 -> weight = 0.5
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
});
