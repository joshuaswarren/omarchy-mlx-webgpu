import type { ModelLoadProgress } from "../kernels/types";
import { fetchCachedArrayBuffer } from "./cache";
import type { Qwen3Config } from "./config";
import {
  getTensorAsFloat32,
  parseSafetensors,
  uploadFloat32Tensor,
  uploadU32Tensor,
  type ParsedSafetensors,
} from "./safetensors";

export interface Q4LinearWeights {
  weight: GPUBuffer; // U32 [N, K/8]
  scales: GPUBuffer; // F32 [N, K/64]
  biases: GPUBuffer; // F32 [N, K/64]
  N: number;
  K: number;
}

export interface Qwen3LayerWeights {
  inputLayernorm: GPUBuffer; // F32 [1024]
  postAttentionLayernorm: GPUBuffer; // F32 [1024]
  qNorm: GPUBuffer; // F32 [128]
  kNorm: GPUBuffer; // F32 [128]
  qProj: Q4LinearWeights; // [2048, 1024]
  kProj: Q4LinearWeights; // [1024, 1024]
  vProj: Q4LinearWeights; // [1024, 1024]
  oProj: Q4LinearWeights; // [1024, 2048]
  gateProj: Q4LinearWeights; // [3072, 1024]
  upProj: Q4LinearWeights; // [3072, 1024]
  downProj: Q4LinearWeights; // [1024, 3072]
}

export interface Qwen3GpuWeights {
  config: Qwen3Config;
  embed: Q4LinearWeights; // tied with lm_head: [151936, 1024]
  layers: Qwen3LayerWeights[];
  finalNorm: GPUBuffer; // F32 [1024]
  /** Layer 0 norm weights kept on CPU for live in-page CPU parity verification on real activations. */
  cpuLayer0InputNorm: Float32Array;
  cpuLayer0QNorm: Float32Array;
  totalGpuBytes: number;
  destroy(): void;
}

function uploadQ4Linear(
  device: GPUDevice,
  st: ParsedSafetensors,
  prefix: string,
  N: number,
  K: number,
): { weights: Q4LinearWeights; bytes: number } {
  const weight = uploadU32Tensor(device, st, `${prefix}.weight`);
  const scales = uploadFloat32Tensor(device, st, `${prefix}.scales`);
  const biases = uploadFloat32Tensor(device, st, `${prefix}.biases`);
  const bytes = weight.size + scales.size + biases.size;
  return { weights: { weight, scales, biases, N, K }, bytes };
}

export async function loadQwen3Weights(
  device: GPUDevice,
  config: Qwen3Config,
  onProgress?: (p: ModelLoadProgress) => void,
): Promise<Qwen3GpuWeights> {
  const rawBuf = await fetchCachedArrayBuffer(
    `${config.baseUrl}/${config.weightsFile}`,
    "weights",
    `${config.repoId} (${config.weightsFile})`,
    config.weightsBytes,
    onProgress,
  );

  onProgress?.({
    stage: "upload",
    loadedBytes: 0,
    totalBytes: config.numHiddenLayers + 2,
    message: "Parsing safetensors header & uploading embeddings to GPU…",
  });

  const st = parseSafetensors(rawBuf);
  let totalGpuBytes = 0;

  const embedRes = uploadQ4Linear(
    device,
    st,
    "model.embed_tokens",
    config.vocabSize,
    config.hiddenSize,
  );
  totalGpuBytes += embedRes.bytes;

  const finalNorm = uploadFloat32Tensor(device, st, "model.norm.weight");
  totalGpuBytes += finalNorm.size;

  const cpuLayer0InputNorm = getTensorAsFloat32(st, "model.layers.0.input_layernorm.weight");
  const cpuLayer0QNorm = getTensorAsFloat32(st, "model.layers.0.self_attn.q_norm.weight");

  const H = config.hiddenSize;
  const qOut = config.numAttentionHeads * config.headDim; // 2048
  const kvOut = config.numKeyValueHeads * config.headDim; // 1024
  const I = config.intermediateSize; // 3072

  const layers: Qwen3LayerWeights[] = [];
  for (let l = 0; l < config.numHiddenLayers; l++) {
    const p = `model.layers.${l}`;
    const inputLayernorm = uploadFloat32Tensor(device, st, `${p}.input_layernorm.weight`);
    const postAttentionLayernorm = uploadFloat32Tensor(
      device,
      st,
      `${p}.post_attention_layernorm.weight`,
    );
    const qNorm = uploadFloat32Tensor(device, st, `${p}.self_attn.q_norm.weight`);
    const kNorm = uploadFloat32Tensor(device, st, `${p}.self_attn.k_norm.weight`);
    const qProj = uploadQ4Linear(device, st, `${p}.self_attn.q_proj`, qOut, H);
    const kProj = uploadQ4Linear(device, st, `${p}.self_attn.k_proj`, kvOut, H);
    const vProj = uploadQ4Linear(device, st, `${p}.self_attn.v_proj`, kvOut, H);
    const oProj = uploadQ4Linear(device, st, `${p}.self_attn.o_proj`, H, qOut);
    const gateProj = uploadQ4Linear(device, st, `${p}.mlp.gate_proj`, I, H);
    const upProj = uploadQ4Linear(device, st, `${p}.mlp.up_proj`, I, H);
    const downProj = uploadQ4Linear(device, st, `${p}.mlp.down_proj`, H, I);

    totalGpuBytes +=
      inputLayernorm.size +
      postAttentionLayernorm.size +
      qNorm.size +
      kNorm.size +
      qProj.bytes +
      kProj.bytes +
      vProj.bytes +
      oProj.bytes +
      gateProj.bytes +
      upProj.bytes +
      downProj.bytes;

    layers.push({
      inputLayernorm,
      postAttentionLayernorm,
      qNorm,
      kNorm,
      qProj: qProj.weights,
      kProj: kProj.weights,
      vProj: vProj.weights,
      oProj: oProj.weights,
      gateProj: gateProj.weights,
      upProj: upProj.weights,
      downProj: downProj.weights,
    });

    if ((l + 1) % 4 === 0 || l === config.numHiddenLayers - 1) {
      onProgress?.({
        stage: "upload",
        loadedBytes: l + 2,
        totalBytes: config.numHiddenLayers + 2,
        message: `Uploaded layer ${l + 1}/${config.numHiddenLayers} (${(totalGpuBytes / 1048576).toFixed(0)} MB GPU buffers)`,
      });
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  onProgress?.({
    stage: "ready",
    loadedBytes: config.numHiddenLayers + 2,
    totalBytes: config.numHiddenLayers + 2,
    message: `Ready — ${config.repoId} (${(totalGpuBytes / 1048576).toFixed(0)} MB on GPU)`,
  });

  const destroyQ4 = (q: Q4LinearWeights) => {
    q.weight.destroy();
    q.scales.destroy();
    q.biases.destroy();
  };

  return {
    config,
    embed: embedRes.weights,
    layers,
    finalNorm,
    cpuLayer0InputNorm,
    cpuLayer0QNorm,
    totalGpuBytes,
    destroy() {
      destroyQ4(embedRes.weights);
      finalNorm.destroy();
      for (const layer of layers) {
        layer.inputLayernorm.destroy();
        layer.postAttentionLayernorm.destroy();
        layer.qNorm.destroy();
        layer.kNorm.destroy();
        destroyQ4(layer.qProj);
        destroyQ4(layer.kProj);
        destroyQ4(layer.vProj);
        destroyQ4(layer.oProj);
        destroyQ4(layer.gateProj);
        destroyQ4(layer.upProj);
        destroyQ4(layer.downProj);
      }
    },
  };
}
