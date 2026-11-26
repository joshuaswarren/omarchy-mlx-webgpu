export interface Qwen3Config {
  repoId: string;
  baseUrl: string;
  numHiddenLayers: number;
  hiddenSize: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
  rmsNormEps: number;
  ropeTheta: number;
  groupSize: number;
  bits: number;
  bosTokenId: number;
  eosTokenIds: number[];
  weightsFile: string;
  weightsBytes: number;
}

export const QWEN3_0_6B_4BIT: Qwen3Config = {
  repoId: "mlx-community/Qwen3-0.6B-4bit",
  baseUrl: "https://huggingface.co/mlx-community/Qwen3-0.6B-4bit/resolve/main",
  numHiddenLayers: 28,
  hiddenSize: 1024,
  numAttentionHeads: 16,
  numKeyValueHeads: 8,
  headDim: 128,
  intermediateSize: 3072,
  vocabSize: 151936,
  rmsNormEps: 1e-6,
  ropeTheta: 1_000_000,
  groupSize: 64,
  bits: 4,
  bosTokenId: 151643,
  eosTokenIds: [151645, 151643],
  weightsFile: "model.safetensors",
  weightsBytes: 335_449_000,
};
