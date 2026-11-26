/**
 * Verified public Apple Silicon Linux benchmark numbers from joshuaswarren/omarchy-mlx.
 * Every row cites its public PR or docs/BENCHMARKS.md source and names the exact model.
 */

export interface Pr997Row {
  context: number;
  genTokens: number;
  baseTokPerSec: number;
  baseMsPerTok: number;
  onlineTokPerSec: number;
  onlineMsPerTok: number;
  gainPct: number;
  wins: string;
  macOsRefTokPerSec?: number;
  greedyShaPrefix: string;
}

export const PR997_BENCHMARK = {
  prNumber: 997,
  prUrl: "https://github.com/joshuaswarren/omarchy-mlx/pull/997",
  repoUrl: "https://github.com/joshuaswarren/omarchy-mlx",
  title: "One-dispatch online-softmax decode SDPA (sdpa_decode_online.comp)",
  hardware: "Apple M1 Max 16-inch (32-core GPU, 64 GB) — Linux Vulkan (Mesa Honeykrisp)",
  model: "Qwen3-4B-Instruct-2507-4bit",
  dispatchDeltaPerToken: -36,
  rows: [
    {
      context: 256,
      genTokens: 64,
      baseTokPerSec: 85.32,
      baseMsPerTok: 11.72,
      onlineTokPerSec: 92.97,
      onlineMsPerTok: 10.76,
      gainPct: 8.97,
      wins: "5/5",
      greedyShaPrefix: "2b1a1b75ab45de32",
    },
    {
      context: 512,
      genTokens: 64,
      baseTokPerSec: 84.50,
      baseMsPerTok: 11.83,
      onlineTokPerSec: 91.71,
      onlineMsPerTok: 10.90,
      gainPct: 8.53,
      wins: "5/5",
      macOsRefTokPerSec: 104.53,
      greedyShaPrefix: "f35142981fe6cba3",
    },
    {
      context: 1024,
      genTokens: 64,
      baseTokPerSec: 80.35,
      baseMsPerTok: 12.44,
      onlineTokPerSec: 88.96,
      onlineMsPerTok: 11.24,
      gainPct: 10.71,
      wins: "5/5",
      greedyShaPrefix: "09a8d2e13cff20ad",
    },
    {
      context: 2048,
      genTokens: 32,
      baseTokPerSec: 75.49,
      baseMsPerTok: 13.25,
      onlineTokPerSec: 83.72,
      onlineMsPerTok: 11.94,
      gainPct: 10.91,
      wins: "5/5",
      greedyShaPrefix: "05b2eac932f527c7",
    },
  ] as Pr997Row[],
} as const;

export const PUBLIC_APPLE_SILICON_HIGHLIGHTS = [
  {
    machine: "Apple M1 Max (16-inch, 64 GB) — Linux Vulkan",
    model: "Qwen3-4B-Instruct-2507-4bit (PR #997, ctx=512)",
    decodeTokPerSec: 91.71,
    reference: "104.53 tok/s macOS Metal (87.7%)",
    sourceUrl: "https://github.com/joshuaswarren/omarchy-mlx/pull/997",
  },
  {
    machine: "Apple M1 (13-inch, 16 GB) — Linux Vulkan",
    model: "Qwen3.5-9B (MLX 4-bit)",
    decodeTokPerSec: 12.4,
    reference: "12.9 tok/s macOS Metal (96%)",
    sourceUrl: "https://github.com/joshuaswarren/omarchy-mlx/blob/main/docs/BENCHMARKS.md",
  },
  {
    machine: "Apple M1 Pro (16-inch, 16 GB) — Linux Vulkan",
    model: "Qwen3.8-2B (MLX 4-bit)",
    decodeTokPerSec: 61.63,
    reference: "279.96 tok/s prefill (512 tok)",
    sourceUrl: "https://github.com/joshuaswarren/omarchy-mlx/blob/main/docs/BENCHMARKS.md",
  },
  {
    machine: "Apple M2 Max (14-inch, 94 GB) — Linux Vulkan",
    model: "Qwen3.5-9B (MLX 4-bit)",
    decodeTokPerSec: 38.8,
    reference: "1.74x llama.cpp Vulkan on same Mac",
    sourceUrl: "https://github.com/joshuaswarren/omarchy-mlx/blob/main/docs/BENCHMARKS.md",
  },
] as const;
