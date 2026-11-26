import { defineConfig } from "@playwright/test";
import { GPU_MATRIX } from "./tests/gpu-matrix/matrix";

/**
 * GPU matrix for the boot parity suite.
 *
 * The page runs runFullParitySuite on boot and publishes
 * window.__PARITY_REPORT__ plus a PARITY_PASS / PARITY_FAIL document title,
 * so every project below reuses one spec and only differs in how its
 * browser reaches a real GPU class.
 *
 * The matrix is serial (workers 1, one adapter per host) and runs on
 * demand; there is no CI workflow. Browser wiring lives in the gpuPage
 * fixture (tests/gpu-matrix/gpu_test.ts):
 *
 *   local  - launches Chromium with the row's launchArgs; run on the GPU
 *            host itself with VK_ICD_FILENAMES selecting the driver.
 *   remote - set GPU_CDP_<NAME> (e.g. GPU_CDP_INTEL_ANV) to the ws:// or
 *            http:// CDP endpoint of a Chromium running on the GPU host,
 *            and GPU_BASE_URL to a page URL that host can reach.
 *
 * GPU_BASE_URL set => the local webServer below is skipped entirely.
 * To change the class set, edit GPU_MATRIX only - the spec keys off the
 * project name.
 */

const LOCAL_BASE_URL = "http://127.0.0.1:4173";

export default defineConfig({
  testDir: "tests/gpu-matrix",
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["line"]],
  use: {
    baseURL: process.env.GPU_BASE_URL ?? LOCAL_BASE_URL,
    viewport: { width: 1440, height: 920 },
  },
  webServer: process.env.GPU_BASE_URL
    ? undefined
    : {
        command:
          "npm run build && npm run preview -- --port 4173 --strictPort --host 127.0.0.1",
        url: LOCAL_BASE_URL,
        timeout: 120_000,
        reuseExistingServer: true,
      },
  projects: Object.keys(GPU_MATRIX).map((name) => ({ name })),
});
