// Shared matrix table: playwright.config.ts builds a project per row and
// gpu_test.ts resolves the same row when launching or connecting a browser.

export interface GpuMatrixRow {
  /** Chromium launch args for local mode; ignored when GPU_CDP_<NAME> is set. */
  launchArgs: string[];
  /** Regex matched against the joined adapter info label in the spec. */
  expect: RegExp;
}

const WEBGPU_BASE_ARGS = [
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
];

const VULKAN_ARGS = [
  ...WEBGPU_BASE_ARGS,
  "--enable-features=Vulkan",
  "--use-webgpu-adapter=vulkan",
];

const METAL_ARGS = [...WEBGPU_BASE_ARGS, "--use-webgpu-adapter=metal"];

const SWIFTSHADER_ARGS = [
  ...WEBGPU_BASE_ARGS,
  "--use-webgpu-adapter=swiftshader",
  // Software rasterizer gate: required since Chrome 120+ for Dawn/SwiftShader
  // on hosts with no GPU. Playwright's local launcher adds it already; a
  // manual chromium launch for CDP mode must pass it explicitly.
  "--enable-unsafe-swiftshader",
];

export const GPU_MATRIX: Record<string, GpuMatrixRow> = {
  swiftshader: {
    launchArgs: SWIFTSHADER_ARGS,
    expect: /swiftshader|llvmpipe|software|google/i,
  },
  "apple-metal-3": {
    launchArgs: METAL_ARGS,
    expect: /apple/i,
  },
  "intel-anv": {
    launchArgs: VULKAN_ARGS,
    expect: /intel/i,
  },
  "nvidia-vulkan": {
    launchArgs: VULKAN_ARGS,
    expect: /nvidia/i,
  },
  // More classes join the matrix with one row each:
  // "amd-radv": { launchArgs: VULKAN_ARGS, expect: /amd|radeon/i },
};

export function projectEnvKey(projectName: string): string {
  return projectName.toUpperCase().replace(/-/g, "_");
}
