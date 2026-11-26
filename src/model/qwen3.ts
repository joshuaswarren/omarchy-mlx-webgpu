import elemWgsl from "../kernels/wgsl/elem.wgsl?raw";
import fastNormWgsl from "../kernels/wgsl/fast_norm.wgsl?raw";
import fastRopeWgsl from "../kernels/wgsl/fast_rope.wgsl?raw";
import fastRopeNormWgsl from "../kernels/wgsl/fast_rope_norm.wgsl?raw";
import qmmVecWgsl from "../kernels/wgsl/qmm_vec.wgsl?raw";
import qmmVecSwigluWgsl from "../kernels/wgsl/qmm_vec_swiglu.wgsl?raw";
import sdpaOnlineWgsl from "../kernels/wgsl/sdpa_decode_online.wgsl?raw";
import sdpaSplitP1Wgsl from "../kernels/wgsl/sdpa_decode_split_p1.wgsl?raw";
import sdpaSplitP2Wgsl from "../kernels/wgsl/sdpa_decode_split_p2.wgsl?raw";
import swigluWgsl from "../kernels/wgsl/swiglu.wgsl?raw";
import type {
  DispatchEvent,
  FaultConfig,
  KernelId,
  OptFlags,
  TokenTrace,
} from "../kernels/types";
import { createCheckedShaderModule, createUniformBuffer } from "../webgpu/device";
import { argmaxNumBlocks, qmmDispatchGrid } from "../webgpu/parity_kernels";
import type { Qwen3Config } from "./config";
import type { Qwen3Tokenizer } from "./tokenizer";
import type { Q4LinearWeights, Qwen3GpuWeights } from "./weights";

const MAX_SEQ = 512;
const SDPA_SPLIT_BK = 64;
const SDPA_MAX_BLOCKS = Math.ceil(MAX_SEQ / SDPA_SPLIT_BK);

interface LayerBindGroups {
  inNorm: GPUBindGroup;
  qProj: GPUBindGroup;
  kProj: GPUBindGroup;
  vProj: GPUBindGroup;
  qNorm: GPUBindGroup;
  kNorm: GPUBindGroup;
  qRope: GPUBindGroup;
  kRope: GPUBindGroup;
  qRopeNorm: GPUBindGroup;
  kRopeNorm: GPUBindGroup;
  kvWrite: GPUBindGroup;
  sdpaOnline: GPUBindGroup;
  sdpaSplitP1: GPUBindGroup;
  sdpaSplitP2: GPUBindGroup;
  oProj: GPUBindGroup;
  res1: GPUBindGroup;
  postNorm: GPUBindGroup;
  gateProj: GPUBindGroup;
  upProj: GPUBindGroup;
  swiglu: GPUBindGroup;
  fusedSwiglu: GPUBindGroup;
  downProj: GPUBindGroup;
  res2: GPUBindGroup;
}

export class Qwen3WebGpuEngine {
  readonly config: Qwen3Config;
  readonly device: GPUDevice;
  readonly weights: Qwen3GpuWeights;
  readonly tokenizer: Qwen3Tokenizer;

  private readonly pipelines: {
    fastNorm: GPUComputePipeline;
    fastRope: GPUComputePipeline;
    fastRopeNorm: GPUComputePipeline;
    qmmVec: GPUComputePipeline;
    qmmVecSwiglu: GPUComputePipeline;
    swiglu: GPUComputePipeline;
    sdpaOnline: GPUComputePipeline;
    sdpaSplitP1: GPUComputePipeline;
    sdpaSplitP2: GPUComputePipeline;
    embedQ4: GPUComputePipeline;
    residualAdd: GPUComputePipeline;
    kvCacheWrite: GPUComputePipeline;
    argmaxP1: GPUComputePipeline;
    argmaxP2: GPUComputePipeline;
  };

  // Pre-allocated activation scratch buffers & KV caches
  private readonly xA: GPUBuffer;
  private readonly xB: GPUBuffer;
  private readonly normBuf: GPUBuffer;
  private readonly qRaw: GPUBuffer;
  private readonly qNormBuf: GPUBuffer;
  private readonly qRot: GPUBuffer;
  private readonly kRaw: GPUBuffer;
  private readonly kNormBuf: GPUBuffer;
  private readonly kRot: GPUBuffer;
  private readonly vRaw: GPUBuffer;
  private readonly attnOut: GPUBuffer;
  private readonly sdpaScratch: GPUBuffer;
  private readonly oOut: GPUBuffer;
  private readonly gateBuf: GPUBuffer;
  private readonly upBuf: GPUBuffer;
  private readonly mlpAct: GPUBuffer;
  private readonly downOut: GPUBuffer;
  private readonly logitsBuf: GPUBuffer;
  private readonly amaxVals: GPUBuffer;
  private readonly amaxIdxs: GPUBuffer;
  private readonly amaxOutIdx: GPUBuffer;
  private readonly amaxOutVal: GPUBuffer;
  private readonly readbackIdx: GPUBuffer;
  private readonly readbackSlice: GPUBuffer;

  private readonly kCaches: GPUBuffer[];
  private readonly vCaches: GPUBuffer[];

  // Reusable dynamic uniforms updated once per token step
  private readonly uEmbed: GPUBuffer;
  private readonly uRopeQ: GPUBuffer;
  private readonly uRopeK: GPUBuffer;
  private readonly uRopeNormQ: GPUBuffer;
  private readonly uRopeNormK: GPUBuffer;
  private readonly uKvWrite: GPUBuffer;
  private readonly uSdpaOnline: GPUBuffer;
  private readonly uSdpaSplit: GPUBuffer;
  private readonly staticUniforms: GPUBuffer[];

  // Static bind groups (zero per-token allocation!)
  private readonly bgEmbed: GPUBindGroup;
  private readonly bgFinalNorm: GPUBindGroup;
  private readonly bgLmHead: GPUBindGroup;
  private readonly bgArgmaxP1: GPUBindGroup;
  private readonly bgArgmaxP2: GPUBindGroup;
  private readonly layerBgs: LayerBindGroups[];
  private readonly amaxBlocks: number;

  private constructor(
    device: GPUDevice,
    weights: Qwen3GpuWeights,
    tokenizer: Qwen3Tokenizer,
    pipelines: Qwen3WebGpuEngine["pipelines"],
  ) {
    this.device = device;
    this.weights = weights;
    this.config = weights.config;
    this.tokenizer = tokenizer;
    this.pipelines = pipelines;

    const { hiddenSize: H, numAttentionHeads: Q, numKeyValueHeads: KV, headDim: D, intermediateSize: I, vocabSize: V, numHiddenLayers: L, rmsNormEps: eps, ropeTheta, groupSize } = this.config;
    const qDim = Q * D;   // 2048
    const kvDim = KV * D; // 1024
    const RW = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;

    const mkBuf = (bytes: number, label: string, extra = 0) =>
      device.createBuffer({ label, size: Math.max(4, bytes), usage: RW | extra });

    this.xA = mkBuf(H * 4, "xA");
    this.xB = mkBuf(H * 4, "xB");
    this.normBuf = mkBuf(H * 4, "normBuf");
    this.qRaw = mkBuf(qDim * 4, "qRaw");
    this.qNormBuf = mkBuf(qDim * 4, "qNormBuf");
    this.qRot = mkBuf(qDim * 4, "qRot");
    this.kRaw = mkBuf(kvDim * 4, "kRaw");
    this.kNormBuf = mkBuf(kvDim * 4, "kNormBuf");
    this.kRot = mkBuf(kvDim * 4, "kRot");
    this.vRaw = mkBuf(kvDim * 4, "vRaw");
    this.attnOut = mkBuf(qDim * 4, "attnOut");
    this.sdpaScratch = mkBuf(Q * SDPA_MAX_BLOCKS * (D + 4) * 4, "sdpaScratch");
    this.oOut = mkBuf(H * 4, "oOut");
    this.gateBuf = mkBuf(I * 4, "gateBuf");
    this.upBuf = mkBuf(I * 4, "upBuf");
    this.mlpAct = mkBuf(I * 4, "mlpAct");
    this.downOut = mkBuf(H * 4, "downOut");
    this.logitsBuf = mkBuf(V * 4, "logitsBuf");

    this.amaxBlocks = argmaxNumBlocks(V);
    this.amaxVals = mkBuf(this.amaxBlocks * 4, "amaxVals");
    this.amaxIdxs = mkBuf(this.amaxBlocks * 4, "amaxIdxs");
    this.amaxOutIdx = mkBuf(4, "amaxOutIdx");
    this.amaxOutVal = mkBuf(4, "amaxOutVal");
    this.readbackIdx = device.createBuffer({
      label: "readbackIdx",
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.readbackSlice = device.createBuffer({
      label: "readbackSlice",
      size: 64 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const kvBytes = KV * MAX_SEQ * D * 4;
    this.kCaches = Array.from({ length: L }, (_, l) => mkBuf(kvBytes, `kCache_${l}`));
    this.vCaches = Array.from({ length: L }, (_, l) => mkBuf(kvBytes, `vCache_${l}`));

    // Static uniforms
    const mkNormU = (rows: number, rowLen: number) => {
      const ab = new ArrayBuffer(32);
      const u = new Uint32Array(ab);
      const f = new Float32Array(ab);
      u[0] = rows; u[1] = rowLen; u[2] = 0; u[3] = 0; f[4] = eps;
      return createUniformBuffer(device, new Uint8Array(ab));
    };
    const mkQmmU = (n: number, k: number) => {
      const u = new Uint32Array([n, k, groupSize, 0, 0, 0, 0, 0]);
      return createUniformBuffer(device, u);
    };

    const uNormHidden = mkNormU(1, H);
    const uNormQ = mkNormU(Q, D);
    const uNormK = mkNormU(KV, D);
    const uQmmQ = mkQmmU(qDim, H);
    const uQmmKV = mkQmmU(kvDim, H);
    const uQmmO = mkQmmU(H, qDim);
    const uQmmGateUp = mkQmmU(I, H);
    const uQmmDown = mkQmmU(H, I);
    const uQmmLmHead = mkQmmU(V, H);
    const uSwiglu = createUniformBuffer(device, new Uint32Array([I, 0, 0, 0]));
    const uResAdd = createUniformBuffer(device, new Uint32Array([H, 0, 0, 0]));
    const uArgmax = createUniformBuffer(device, new Uint32Array([V, this.amaxBlocks, 0, 0]));

    this.staticUniforms = [
      uNormHidden, uNormQ, uNormK, uQmmQ, uQmmKV, uQmmO,
      uQmmGateUp, uQmmDown, uQmmLmHead, uSwiglu, uResAdd, uArgmax,
    ];

    // Dynamic per-step uniforms
    this.uEmbed = createUniformBuffer(device, new Uint32Array([0, H, groupSize, 0]), "uEmbed");
    const mkRopeU = (heads: number) => {
      const ab = new ArrayBuffer(32);
      const u = new Uint32Array(ab); const f = new Float32Array(ab);
      u[0] = heads; u[1] = D; u[2] = 0; u[3] = 0; f[4] = ropeTheta; f[5] = 1.0;
      return createUniformBuffer(device, new Uint8Array(ab));
    };
    const mkRopeNormU = (heads: number) => {
      const ab = new ArrayBuffer(32);
      const u = new Uint32Array(ab); const f = new Float32Array(ab);
      u[0] = heads; u[1] = D; u[2] = 0; u[3] = 0; f[4] = eps; f[5] = ropeTheta; f[6] = 1.0; u[7] = 0;
      return createUniformBuffer(device, new Uint8Array(ab));
    };
    this.uRopeQ = mkRopeU(Q);
    this.uRopeK = mkRopeU(KV);
    this.uRopeNormQ = mkRopeNormU(Q);
    this.uRopeNormK = mkRopeNormU(KV);
    this.uKvWrite = createUniformBuffer(device, new Uint32Array([KV, D, MAX_SEQ, 0]), "uKvWrite");
    const sdpaAb = new ArrayBuffer(32);
    const su = new Uint32Array(sdpaAb); const sf = new Float32Array(sdpaAb);
    su[0] = Q; su[1] = KV; su[2] = 1; su[3] = MAX_SEQ; sf[4] = 1 / Math.sqrt(D); su[5] = 0; su[6] = 1;
    this.uSdpaOnline = createUniformBuffer(device, new Uint8Array(sdpaAb), "uSdpaOnline");
    this.uSdpaSplit = createUniformBuffer(device, new Uint8Array(sdpaAb), "uSdpaSplit");

    // Helper for QMM bind group
    const bgQmm = (x: GPUBuffer, lin: Q4LinearWeights, dst: GPUBuffer, u: GPUBuffer) =>
      device.createBindGroup({
        layout: pipelines.qmmVec.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: x } },
          { binding: 1, resource: { buffer: lin.weight } },
          { binding: 2, resource: { buffer: lin.scales } },
          { binding: 3, resource: { buffer: lin.biases } },
          { binding: 4, resource: { buffer: dst } },
          { binding: 5, resource: { buffer: u } },
        ],
      });

    this.bgEmbed = device.createBindGroup({
      layout: pipelines.embedQ4.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: weights.embed.weight } },
        { binding: 1, resource: { buffer: weights.embed.scales } },
        { binding: 2, resource: { buffer: weights.embed.biases } },
        { binding: 3, resource: { buffer: this.xA } },
        { binding: 4, resource: { buffer: this.uEmbed } },
      ],
    });

    this.bgFinalNorm = device.createBindGroup({
      layout: pipelines.fastNorm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.xA } },
        { binding: 1, resource: { buffer: weights.finalNorm } },
        { binding: 2, resource: { buffer: this.normBuf } },
        { binding: 3, resource: { buffer: uNormHidden } },
      ],
    });

    this.bgLmHead = bgQmm(this.normBuf, weights.embed, this.logitsBuf, uQmmLmHead);

    this.bgArgmaxP1 = device.createBindGroup({
      layout: pipelines.argmaxP1.getBindGroupLayout(0),
      entries: [
        { binding: 14, resource: { buffer: this.logitsBuf } },
        { binding: 15, resource: { buffer: this.amaxVals } },
        { binding: 16, resource: { buffer: this.amaxIdxs } },
        { binding: 17, resource: { buffer: uArgmax } },
      ],
    });

    this.bgArgmaxP2 = device.createBindGroup({
      layout: pipelines.argmaxP2.getBindGroupLayout(0),
      entries: [
        { binding: 17, resource: { buffer: uArgmax } },
        { binding: 18, resource: { buffer: this.amaxVals } },
        { binding: 19, resource: { buffer: this.amaxIdxs } },
        { binding: 20, resource: { buffer: this.amaxOutIdx } },
        { binding: 21, resource: { buffer: this.amaxOutVal } },
      ],
    });

    this.layerBgs = weights.layers.map((lw, l) => ({
      inNorm: device.createBindGroup({
        layout: pipelines.fastNorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.xA } },
          { binding: 1, resource: { buffer: lw.inputLayernorm } },
          { binding: 2, resource: { buffer: this.normBuf } },
          { binding: 3, resource: { buffer: uNormHidden } },
        ],
      }),
      qProj: bgQmm(this.normBuf, lw.qProj, this.qRaw, uQmmQ),
      kProj: bgQmm(this.normBuf, lw.kProj, this.kRaw, uQmmKV),
      vProj: bgQmm(this.normBuf, lw.vProj, this.vRaw, uQmmKV),
      qNorm: device.createBindGroup({
        layout: pipelines.fastNorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.qRaw } },
          { binding: 1, resource: { buffer: lw.qNorm } },
          { binding: 2, resource: { buffer: this.qNormBuf } },
          { binding: 3, resource: { buffer: uNormQ } },
        ],
      }),
      kNorm: device.createBindGroup({
        layout: pipelines.fastNorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.kRaw } },
          { binding: 1, resource: { buffer: lw.kNorm } },
          { binding: 2, resource: { buffer: this.kNormBuf } },
          { binding: 3, resource: { buffer: uNormK } },
        ],
      }),
      qRope: device.createBindGroup({
        layout: pipelines.fastRope.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.qNormBuf } },
          { binding: 1, resource: { buffer: this.qRot } },
          { binding: 2, resource: { buffer: this.uRopeQ } },
        ],
      }),
      kRope: device.createBindGroup({
        layout: pipelines.fastRope.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.kNormBuf } },
          { binding: 1, resource: { buffer: this.kRot } },
          { binding: 2, resource: { buffer: this.uRopeK } },
        ],
      }),
      qRopeNorm: device.createBindGroup({
        layout: pipelines.fastRopeNorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.qRaw } },
          { binding: 1, resource: { buffer: lw.qNorm } },
          { binding: 2, resource: { buffer: this.qRot } },
          { binding: 3, resource: { buffer: this.uRopeNormQ } },
        ],
      }),
      kRopeNorm: device.createBindGroup({
        layout: pipelines.fastRopeNorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.kRaw } },
          { binding: 1, resource: { buffer: lw.kNorm } },
          { binding: 2, resource: { buffer: this.kRot } },
          { binding: 3, resource: { buffer: this.uRopeNormK } },
        ],
      }),
      kvWrite: device.createBindGroup({
        layout: pipelines.kvCacheWrite.getBindGroupLayout(0),
        entries: [
          { binding: 9, resource: { buffer: this.kRot } },
          { binding: 10, resource: { buffer: this.vRaw } },
          { binding: 11, resource: { buffer: this.kCaches[l] } },
          { binding: 12, resource: { buffer: this.vCaches[l] } },
          { binding: 13, resource: { buffer: this.uKvWrite } },
        ],
      }),
      sdpaOnline: device.createBindGroup({
        layout: pipelines.sdpaOnline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.qRot } },
          { binding: 1, resource: { buffer: this.kCaches[l] } },
          { binding: 2, resource: { buffer: this.vCaches[l] } },
          { binding: 3, resource: { buffer: this.attnOut } },
          { binding: 4, resource: { buffer: this.uSdpaOnline } },
        ],
      }),
      sdpaSplitP1: device.createBindGroup({
        layout: pipelines.sdpaSplitP1.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.qRot } },
          { binding: 1, resource: { buffer: this.kCaches[l] } },
          { binding: 2, resource: { buffer: this.vCaches[l] } },
          { binding: 3, resource: { buffer: this.sdpaScratch } },
          { binding: 4, resource: { buffer: this.uSdpaSplit } },
        ],
      }),
      sdpaSplitP2: device.createBindGroup({
        layout: pipelines.sdpaSplitP2.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.sdpaScratch } },
          { binding: 1, resource: { buffer: this.attnOut } },
          { binding: 2, resource: { buffer: this.uSdpaSplit } },
        ],
      }),
      oProj: bgQmm(this.attnOut, lw.oProj, this.oOut, uQmmO),
      res1: device.createBindGroup({
        layout: pipelines.residualAdd.getBindGroupLayout(0),
        entries: [
          { binding: 5, resource: { buffer: this.xA } },
          { binding: 6, resource: { buffer: this.oOut } },
          { binding: 7, resource: { buffer: this.xB } },
          { binding: 8, resource: { buffer: uResAdd } },
        ],
      }),
      postNorm: device.createBindGroup({
        layout: pipelines.fastNorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.xB } },
          { binding: 1, resource: { buffer: lw.postAttentionLayernorm } },
          { binding: 2, resource: { buffer: this.normBuf } },
          { binding: 3, resource: { buffer: uNormHidden } },
        ],
      }),
      gateProj: bgQmm(this.normBuf, lw.gateProj, this.gateBuf, uQmmGateUp),
      upProj: bgQmm(this.normBuf, lw.upProj, this.upBuf, uQmmGateUp),
      swiglu: device.createBindGroup({
        layout: pipelines.swiglu.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gateBuf } },
          { binding: 1, resource: { buffer: this.upBuf } },
          { binding: 2, resource: { buffer: this.mlpAct } },
          { binding: 3, resource: { buffer: uSwiglu } },
        ],
      }),
      fusedSwiglu: device.createBindGroup({
        layout: pipelines.qmmVecSwiglu.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.normBuf } },
          { binding: 1, resource: { buffer: lw.gateProj.weight } },
          { binding: 2, resource: { buffer: lw.gateProj.scales } },
          { binding: 3, resource: { buffer: lw.gateProj.biases } },
          { binding: 4, resource: { buffer: lw.upProj.weight } },
          { binding: 5, resource: { buffer: lw.upProj.scales } },
          { binding: 6, resource: { buffer: lw.upProj.biases } },
          { binding: 7, resource: { buffer: this.mlpAct } },
          { binding: 8, resource: { buffer: uQmmGateUp } },
        ],
      }),
      downProj: bgQmm(this.mlpAct, lw.downProj, this.downOut, uQmmDown),
      res2: device.createBindGroup({
        layout: pipelines.residualAdd.getBindGroupLayout(0),
        entries: [
          { binding: 5, resource: { buffer: this.xB } },
          { binding: 6, resource: { buffer: this.downOut } },
          { binding: 7, resource: { buffer: this.xA } },
          { binding: 8, resource: { buffer: uResAdd } },
        ],
      }),
    }));
  }

  static async create(
    device: GPUDevice,
    weights: Qwen3GpuWeights,
    tokenizer: Qwen3Tokenizer,
  ): Promise<Qwen3WebGpuEngine> {
    const [
      fastNormMod, fastRopeMod, fastRopeNormMod, qmmVecMod, qmmVecSwigluMod,
      swigluMod, sdpaOnlineMod, sdpaSplitP1Mod, sdpaSplitP2Mod, elemMod,
    ] = await Promise.all([
      createCheckedShaderModule(device, "fast_norm", fastNormWgsl),
      createCheckedShaderModule(device, "fast_rope", fastRopeWgsl),
      createCheckedShaderModule(device, "fast_rope_norm", fastRopeNormWgsl),
      createCheckedShaderModule(device, "qmm_vec", qmmVecWgsl),
      createCheckedShaderModule(device, "qmm_vec_swiglu", qmmVecSwigluWgsl),
      createCheckedShaderModule(device, "swiglu", swigluWgsl),
      createCheckedShaderModule(device, "sdpa_decode_online", sdpaOnlineWgsl),
      createCheckedShaderModule(device, "sdpa_decode_split_p1", sdpaSplitP1Wgsl),
      createCheckedShaderModule(device, "sdpa_decode_split_p2", sdpaSplitP2Wgsl),
      createCheckedShaderModule(device, "elem", elemWgsl),
    ]);

    const mkPipe = (module: GPUShaderModule, entryPoint = "main") =>
      device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });

    return new Qwen3WebGpuEngine(device, weights, tokenizer, {
      fastNorm: mkPipe(fastNormMod),
      fastRope: mkPipe(fastRopeMod),
      fastRopeNorm: mkPipe(fastRopeNormMod),
      qmmVec: mkPipe(qmmVecMod),
      qmmVecSwiglu: mkPipe(qmmVecSwigluMod),
      swiglu: mkPipe(swigluMod),
      sdpaOnline: mkPipe(sdpaOnlineMod),
      sdpaSplitP1: mkPipe(sdpaSplitP1Mod),
      sdpaSplitP2: mkPipe(sdpaSplitP2Mod),
      embedQ4: mkPipe(elemMod, "embed_q4"),
      residualAdd: mkPipe(elemMod, "residual_add"),
      kvCacheWrite: mkPipe(elemMod, "kv_cache_write"),
      argmaxP1: mkPipe(elemMod, "argmax_p1"),
      argmaxP2: mkPipe(elemMod, "argmax_p2"),
    });
  }

  private writeStepUniforms(tokenId: number, pos: number, faultEnabled: boolean): number {
    const { hiddenSize: H, numAttentionHeads: Q, numKeyValueHeads: KV, headDim: D, rmsNormEps: eps, ropeTheta, groupSize } = this.config;
    const kLen = pos + 1;
    const nblocks = Math.max(1, Math.ceil(kLen / SDPA_SPLIT_BK));
    const faultU32 = faultEnabled ? 1 : 0;

    this.device.queue.writeBuffer(this.uEmbed, 0, new Uint32Array([tokenId, H, groupSize, 0]));

    const mkRope = (heads: number) => {
      const ab = new ArrayBuffer(32);
      const u = new Uint32Array(ab); const f = new Float32Array(ab);
      u[0] = heads; u[1] = D; u[2] = pos; u[3] = 0; f[4] = ropeTheta; f[5] = 1.0;
      return new Uint8Array(ab);
    };
    this.device.queue.writeBuffer(this.uRopeQ, 0, mkRope(Q));
    this.device.queue.writeBuffer(this.uRopeK, 0, mkRope(KV));

    const mkRopeNorm = (heads: number) => {
      const ab = new ArrayBuffer(32);
      const u = new Uint32Array(ab); const f = new Float32Array(ab);
      u[0] = heads; u[1] = D; u[2] = pos; u[3] = 0; f[4] = eps; f[5] = ropeTheta; f[6] = 1.0; u[7] = 0;
      return new Uint8Array(ab);
    };
    this.device.queue.writeBuffer(this.uRopeNormQ, 0, mkRopeNorm(Q));
    this.device.queue.writeBuffer(this.uRopeNormK, 0, mkRopeNorm(KV));

    this.device.queue.writeBuffer(this.uKvWrite, 0, new Uint32Array([KV, D, MAX_SEQ, pos]));

    const sdpaAb = new ArrayBuffer(32);
    const su = new Uint32Array(sdpaAb); const sf = new Float32Array(sdpaAb);
    su[0] = Q; su[1] = KV; su[2] = kLen; su[3] = MAX_SEQ;
    sf[4] = 1 / Math.sqrt(D); su[5] = faultU32; su[6] = nblocks;
    this.device.queue.writeBuffer(this.uSdpaOnline, 0, new Uint8Array(sdpaAb));
    this.device.queue.writeBuffer(this.uSdpaSplit, 0, new Uint8Array(sdpaAb));

    return nblocks;
  }

  /**
   * Executes one full 28-layer token step in a single WebGPU command submission.
   * Returns the greedy next token id (if computeLogits=true) and exact dispatch trace.
   */
  async forwardStep(
    tokenId: number,
    pos: number,
    opts: OptFlags,
    fault: FaultConfig,
    computeLogits: boolean,
    recordTrace: boolean,
  ): Promise<{ nextTokenId: number; dispatches: DispatchEvent[]; attentionDispatches: number; sampleSlice?: Float32Array }> {
    const { hiddenSize: H, numAttentionHeads: Q, numKeyValueHeads: KV, headDim: D, intermediateSize: I, vocabSize: V, numHiddenLayers: L, groupSize } = this.config;
    const qDim = Q * D;
    const kvDim = KV * D;
    const nblocks = this.writeStepUniforms(tokenId, pos, fault.enabled);
    const kLen = pos + 1;

    const dispatches: DispatchEvent[] = [];
    let dIdx = 0;
    let attentionDispatches = 0;
    const q4Bytes = (n: number, k: number) => Math.round(n * k * 0.5 + n * (k / groupSize) * 8);

    const enc = this.device.createCommandEncoder({ label: `step_${pos}` });
    const pass = enc.beginComputePass({ label: `qwen3_step_${pos}` });

    const run = (
      layer: number,
      kernel: KernelId,
      label: string,
      pipe: GPUComputePipeline,
      bg: GPUBindGroup,
      wg: [number, number, number],
      wgSize: [number, number, number],
      bytesRead: number,
      bytesWritten: number,
    ) => {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wg[0], wg[1], wg[2]);
      if (recordTrace) {
        dispatches.push({
          index: dIdx++,
          layer,
          kernel,
          label,
          workgroups: wg,
          workgroupSize: wgSize,
          bytesRead,
          bytesWritten,
        });
      }
    };

    const gQ = qmmDispatchGrid(qDim);
    const gKV = qmmDispatchGrid(kvDim);
    const gH = qmmDispatchGrid(H);
    const gI = qmmDispatchGrid(I);
    const gV = qmmDispatchGrid(V);

    // Token embedding
    run(-1, "embed_q4", "embed_tokens (4-bit)", this.pipelines.embedQ4, this.bgEmbed, [Math.ceil(H / 256), 1, 1], [256, 1, 1], q4Bytes(1, H), H * 4);

    // 28 transformer layers
    for (let l = 0; l < L; l++) {
      const lb = this.layerBgs[l];

      run(l, "fast_norm", "input_layernorm", this.pipelines.fastNorm, lb.inNorm, [1, 1, 1], [256, 1, 1], H * 8, H * 4);
      run(l, "qmm_vec", "q_proj (Q4)", this.pipelines.qmmVec, lb.qProj, gQ, [128, 1, 1], q4Bytes(qDim, H) + H * 4, qDim * 4);
      run(l, "qmm_vec", "k_proj (Q4)", this.pipelines.qmmVec, lb.kProj, gKV, [128, 1, 1], q4Bytes(kvDim, H) + H * 4, kvDim * 4);
      run(l, "qmm_vec", "v_proj (Q4)", this.pipelines.qmmVec, lb.vProj, gKV, [128, 1, 1], q4Bytes(kvDim, H) + H * 4, kvDim * 4);

      if (opts.fusedRopeNorm) {
        run(l, "fast_rope_norm", "q_norm + RoPE (fused)", this.pipelines.fastRopeNorm, lb.qRopeNorm, [Q, 1, 1], [128, 1, 1], qDim * 4 + D * 4, qDim * 4);
        run(l, "fast_rope_norm", "k_norm + RoPE (fused)", this.pipelines.fastRopeNorm, lb.kRopeNorm, [KV, 1, 1], [128, 1, 1], kvDim * 4 + D * 4, kvDim * 4);
      } else {
        run(l, "fast_norm", "q_norm", this.pipelines.fastNorm, lb.qNorm, [Q, 1, 1], [256, 1, 1], qDim * 4 + D * 4, qDim * 4);
        run(l, "fast_norm", "k_norm", this.pipelines.fastNorm, lb.kNorm, [KV, 1, 1], [256, 1, 1], kvDim * 4 + D * 4, kvDim * 4);
        run(l, "fast_rope", "q_rope", this.pipelines.fastRope, lb.qRope, [Q, 1, 1], [64, 1, 1], qDim * 4, qDim * 4);
        run(l, "fast_rope", "k_rope", this.pipelines.fastRope, lb.kRope, [KV, 1, 1], [64, 1, 1], kvDim * 4, kvDim * 4);
      }

      run(l, "kv_cache_write", "kv_cache_write", this.pipelines.kvCacheWrite, lb.kvWrite, [Math.ceil(kvDim / 256), 1, 1], [256, 1, 1], kvDim * 8, kvDim * 8);

      const kvReadBytes = KV * kLen * D * 4 * 2;
      if (opts.oneDispatchAttention) {
        attentionDispatches += 1;
        run(l, "sdpa_decode_online", "SDPA online (PR #997)", this.pipelines.sdpaOnline, lb.sdpaOnline, [Q, 1, 1], [256, 1, 1], qDim * 4 + kvReadBytes, qDim * 4);
      } else {
        attentionDispatches += 2;
        const scratchBytes = Q * nblocks * (D + 4) * 4;
        run(l, "sdpa_decode_split_p1", "SDPA split-K P1", this.pipelines.sdpaSplitP1, lb.sdpaSplitP1, [Q, nblocks, 1], [128, 1, 1], qDim * 4 + kvReadBytes, scratchBytes);
        run(l, "sdpa_decode_split_p2", "SDPA split-K P2", this.pipelines.sdpaSplitP2, lb.sdpaSplitP2, [Q, 1, 1], [128, 1, 1], scratchBytes, qDim * 4);
      }

      run(l, "qmm_vec", "o_proj (Q4)", this.pipelines.qmmVec, lb.oProj, gH, [128, 1, 1], q4Bytes(H, qDim) + qDim * 4, H * 4);
      run(l, "residual_add", "attn residual", this.pipelines.residualAdd, lb.res1, [Math.ceil(H / 256), 1, 1], [256, 1, 1], H * 8, H * 4);

      run(l, "fast_norm", "post_attn_layernorm", this.pipelines.fastNorm, lb.postNorm, [1, 1, 1], [256, 1, 1], H * 8, H * 4);

      if (opts.fusedSwigluQmm) {
        run(l, "qmm_vec_swiglu", "gate+up+SwiGLU (fused)", this.pipelines.qmmVecSwiglu, lb.fusedSwiglu, gI, [128, 1, 1], q4Bytes(I, H) * 2 + H * 4, I * 4);
      } else {
        run(l, "qmm_vec", "gate_proj (Q4)", this.pipelines.qmmVec, lb.gateProj, gI, [128, 1, 1], q4Bytes(I, H) + H * 4, I * 4);
        run(l, "qmm_vec", "up_proj (Q4)", this.pipelines.qmmVec, lb.upProj, gI, [128, 1, 1], q4Bytes(I, H) + H * 4, I * 4);
        run(l, "swiglu", "swiglu", this.pipelines.swiglu, lb.swiglu, [Math.ceil(I / 256), 1, 1], [256, 1, 1], I * 8, I * 4);
      }

      run(l, "qmm_vec", "down_proj (Q4)", this.pipelines.qmmVec, lb.downProj, gH, [128, 1, 1], q4Bytes(H, I) + I * 4, H * 4);
      run(l, "residual_add", "mlp residual", this.pipelines.residualAdd, lb.res2, [Math.ceil(H / 256), 1, 1], [256, 1, 1], H * 8, H * 4);
    }

    if (computeLogits) {
      run(-1, "fast_norm", "final_norm", this.pipelines.fastNorm, this.bgFinalNorm, [1, 1, 1], [256, 1, 1], H * 8, H * 4);
      run(-1, "qmm_vec", "lm_head (tied Q4)", this.pipelines.qmmVec, this.bgLmHead, gV, [128, 1, 1], q4Bytes(V, H) + H * 4, V * 4);
      run(-1, "argmax", "argmax P1", this.pipelines.argmaxP1, this.bgArgmaxP1, [this.amaxBlocks, 1, 1], [256, 1, 1], V * 4, this.amaxBlocks * 8);
      run(-1, "argmax", "argmax P2", this.pipelines.argmaxP2, this.bgArgmaxP2, [1, 1, 1], [256, 1, 1], this.amaxBlocks * 8, 8);
    }

    pass.end();

    if (!computeLogits) {
      this.device.queue.submit([enc.finish()]);
      return { nextTokenId: -1, dispatches, attentionDispatches };
    }

    enc.copyBufferToBuffer(this.amaxOutIdx, 0, this.readbackIdx, 0, 4);
    if (recordTrace) {
      enc.copyBufferToBuffer(this.attnOut, 0, this.readbackSlice, 0, 64 * 4);
    }
    this.device.queue.submit([enc.finish()]);

    await this.readbackIdx.mapAsync(GPUMapMode.READ);
    const nextTokenId = new Uint32Array(this.readbackIdx.getMappedRange())[0];
    this.readbackIdx.unmap();

    let sampleSlice: Float32Array | undefined;
    if (recordTrace) {
      await this.readbackSlice.mapAsync(GPUMapMode.READ);
      sampleSlice = new Float32Array(this.readbackSlice.getMappedRange().slice(0));
      this.readbackSlice.unmap();
    }

    return { nextTokenId, dispatches, attentionDispatches, sampleSlice };
  }

  /**
   * Streams greedy tokens from Qwen3-0.6B-4bit on the visitor's WebGPU device.
   * Respects live mid-generation changes to OptFlags and FaultConfig on every token.
   */
  async generate(
    userPrompt: string,
    maxNewTokens: number,
    getOpts: () => OptFlags,
    getFault: () => FaultConfig,
    onToken: (trace: TokenTrace) => void,
    shouldStop: () => boolean,
  ): Promise<{ tokens: number[]; text: string; avgTokPerSec: number }> {
    const formatted = this.tokenizer.formatChatPrompt(userPrompt);
    const promptIds = this.tokenizer.encode(formatted);
    if (promptIds.length >= MAX_SEQ - 8) {
      throw new Error(`Prompt too long (${promptIds.length} tokens; max ${MAX_SEQ - 8})`);
    }

    // Prefill prompt tokens 0..N-2 without LM head readback (batched onto GPU queue)
    const prefillOpts = getOpts();
    const prefillFault = getFault();
    for (let i = 0; i < promptIds.length - 1; i++) {
      if (shouldStop()) return { tokens: [], text: "", avgTokPerSec: 0 };
      await this.forwardStep(promptIds[i], i, prefillOpts, prefillFault, false, false);
    }
    await this.device.queue.onSubmittedWorkDone();

    // Decode first token from last prompt position
    const generated: number[] = [];
    let pos = promptIds.length - 1;
    let curToken = promptIds[pos];
    let decodedSoFar = "";
    let totalDecodeMs = 0;
    const eosSet = new Set(this.config.eosTokenIds);
    const budget = Math.min(maxNewTokens, MAX_SEQ - promptIds.length - 1);

    for (let step = 0; step < budget; step++) {
      if (shouldStop()) break;
      const opts = { ...getOpts() };
      const fault = { ...getFault() };
      const t0 = performance.now();
      const res = await this.forwardStep(curToken, pos, opts, fault, true, true);
      const tokenMs = Math.max(0.01, performance.now() - t0);
      totalDecodeMs += tokenMs;

      const nextId = res.nextTokenId;
      if (eosSet.has(nextId)) break;

      generated.push(nextId);
      const fullDecoded = this.tokenizer.decode(generated, true);
      const deltaText = fullDecoded.slice(decodedSoFar.length);
      decodedSoFar = fullDecoded;

      onToken({
        step,
        tokenId: nextId,
        text: deltaText,
        tokenMs,
        tokPerSec: 1000 / tokenMs,
        totalDispatches: res.dispatches.length,
        attentionDispatches: res.attentionDispatches,
        optFlags: opts,
        faultEnabled: fault.enabled,
        dispatches: res.dispatches,
        sampleSlice: res.sampleSlice,
      });

      curToken = nextId;
      pos += 1;
      // Yield 1 frame tick so browser paints live tokens & toggles stay instant
      await new Promise((r) => setTimeout(r, 0));
    }

    return {
      tokens: generated,
      text: decodedSoFar,
      avgTokPerSec: generated.length > 0 && totalDecodeMs > 0 ? (generated.length * 1000) / totalDecodeMs : 0,
    };
  }

  destroy(): void {
    const bufs = [
      this.xA, this.xB, this.normBuf, this.qRaw, this.qNormBuf, this.qRot,
      this.kRaw, this.kNormBuf, this.kRot, this.vRaw, this.attnOut,
      this.sdpaScratch, this.oOut, this.gateBuf, this.upBuf, this.mlpAct,
      this.downOut, this.logitsBuf, this.amaxVals, this.amaxIdxs,
      this.amaxOutIdx, this.amaxOutVal, this.readbackIdx, this.readbackSlice,
      this.uEmbed, this.uRopeQ, this.uRopeK, this.uRopeNormQ, this.uRopeNormK,
      this.uKvWrite, this.uSdpaOnline, this.uSdpaSplit,
      ...this.staticUniforms,
      ...this.kCaches,
      ...this.vCaches,
    ];
    for (const b of bufs) b.destroy();
  }
}
