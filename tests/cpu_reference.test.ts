import { describe, expect, it } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseSafetensors, getTensorAsFloat32, getTensorU32Slice } from "../src/model/safetensors";
import { Qwen3Tokenizer } from "../src/model/tokenizer";
import {
  argmaxCpu,
  bf16ToF32,
  compareArrays,
  embedQ4Cpu,
  kvCacheWriteCpu,
  qmmQ4Cpu,
  qmmSwigluQ4Cpu,
  residualAddCpu,
  rmsNormCpu,
  ropeCpu,
  ropeNormCpu,
  sdpaDecodeCpu,
  seededFloat32,
  seededQ4Weights,
  swigluCpu,
} from "../src/reference/cpu";
import { argmaxNumBlocks, qmmDispatchGrid } from "../src/webgpu/parity_kernels";

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

  it("Fused RMSNorm + RoPE matches sequential rmsNormCpu then ropeCpu", () => {
    const x = seededFloat32(16 * 128, 11, 1.5);
    const w = seededFloat32(128, 12, 1.0);
    const fused = ropeNormCpu(x, w, 16, 128, 42, 1e-6, 1_000_000);
    const seq = ropeCpu(rmsNormCpu(x, w, 16, 128, 1e-6), 16, 128, 42, 1_000_000);
    expect(compareArrays(fused, seq).maxAbsDiff).toBe(0);
    const atZero = ropeNormCpu(x, w, 16, 128, 0, 1e-6, 1_000_000);
    expect(compareArrays(atZero, rmsNormCpu(x, w, 16, 128, 1e-6)).maxAbsDiff).toBeLessThan(1e-6);
  });

  it("Fused Gate+Up Q4 QMM + SwiGLU matches separate projections + swigluCpu", () => {
    const x = seededFloat32(256, 21, 0.5);
    const gate = seededQ4Weights(64, 256, 22, 64);
    const up = seededQ4Weights(64, 256, 23, 64);
    const fused = qmmSwigluQ4Cpu(
      x,
      gate.weightsU32,
      gate.scalesF32,
      gate.biasesF32,
      up.weightsU32,
      up.scalesF32,
      up.biasesF32,
      64,
      256,
      64,
    );
    const gOut = qmmQ4Cpu(x, gate.weightsU32, gate.scalesF32, gate.biasesF32, 64, 256, 64);
    const uOut = qmmQ4Cpu(x, up.weightsU32, up.scalesF32, up.biasesF32, 64, 256, 64);
    expect(compareArrays(fused, swigluCpu(gOut, uOut)).maxAbsDiff).toBe(0);
  });

  it("Residual add, KV cache write, deterministic argmax, and dispatch helpers match spec", () => {
    const a = new Float32Array([1.5, -2.25, 0.5, 4.0]);
    const b = new Float32Array([-0.5, 2.25, 1.25, -1.0]);
    expect(Array.from(residualAddCpu(a, b))).toEqual([1.0, 0.0, 1.75, 3.0]);

    const kvHeads = 2;
    const headDim = 4;
    const maxSeq = 3;
    const pos = 1;
    const kIn = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const vIn = new Float32Array([-1, -2, -3, -4, -5, -6, -7, -8]);
    const zeros = new Float32Array(kvHeads * maxSeq * headDim);
    const { kCache, vCache } = kvCacheWriteCpu(kIn, vIn, zeros, zeros, kvHeads, headDim, maxSeq, pos);
    expect(Array.from(kCache.slice(4, 8))).toEqual([1, 2, 3, 4]);
    expect(Array.from(kCache.slice(16, 20))).toEqual([5, 6, 7, 8]);
    expect(Array.from(vCache.slice(16, 20))).toEqual([-5, -6, -7, -8]);
    expect(kCache[0]).toBe(0);
    expect(kCache[8]).toBe(0);

    const neg = new Float32Array([-9.5, -3.25, -4.0, -3.25]);
    expect(argmaxCpu(neg)).toEqual({ index: 1, value: -3.25 });

    expect(qmmDispatchGrid(151936)).toEqual([18992, 1, 1]);
    expect(qmmDispatchGrid(600000)).toEqual([65535, 2, 1]);
    expect(argmaxNumBlocks(151936)).toBe(256);
  });
});

describe("WGSL portability guardrails", () => {
  const wgslFiles = [
    "fast_rope.wgsl",
    "fast_rope_norm.wgsl",
    "qmm_vec.wgsl",
    "qmm_vec_swiglu.wgsl",
    "swiglu.wgsl",
    "elem.wgsl",
  ];
  const forbiddenReserved = [
    "active",
    "filter",
    "target",
    "shared",
    "static",
    "set",
    "get",
    "match",
    "self",
    "super",
    "mod",
    "use",
    "null",
    "void",
    "class",
    "enum",
    "union",
  ];

  for (const file of wgslFiles) {
    it(`${file} uses portable WGSL (f32, <=16KB workgroup mem, no subgroup ext, no reserved words)`, () => {
      const src = readFileSync(resolve(import.meta.dir, "../src/kernels/wgsl", file), "utf8");
      const codeOnly = src
        .replace(/\/\/[^\n]*/g, "")
        .replace(/\/\*[\s\S]*?\*\//g, "");

      expect(codeOnly).not.toMatch(/\benable\s+(subgroups|f16)\b/);

      let wgBytes = 0;
      for (const m of codeOnly.matchAll(
        /var<workgroup>\s+\w+\s*:\s*array<(?:vec([234])<)?(f32|u32|i32)>?\s*,\s*(\d+)>/g,
      )) {
        const lanes = m[1] ? Number(m[1]) : 1;
        wgBytes += lanes * 4 * Number(m[3]);
      }
      expect(wgBytes).toBeLessThanOrEqual(16384);

      const bindings = [...codeOnly.matchAll(/@binding\((\d+)\)/g)].map((m) => Number(m[1]));
      expect(new Set(bindings).size).toBe(bindings.length);

      for (const kw of forbiddenReserved) {
        expect(codeOnly).not.toMatch(new RegExp(`\\b${kw}\\b`));
      }
    });
  }
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
