import { expect, test } from "./gpu_test";
import { GPU_MATRIX } from "./matrix";

interface ParityReport {
  adapter: {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  };
  model: string;
  results: { kernel: string; label: string; passed: boolean; maxAbsDiff: number }[];
  faultResults: { kernel: string; label: string; passed: boolean }[];
  allCleanPassed: boolean;
  allFaultsDetected: boolean;
}

test.describe("GPU matrix: boot parity on the project adapter class", () => {
  test("boot parity passes and the adapter is the expected class", async ({ gpuPage }) => {
    test.setTimeout(180_000);
    await gpuPage.goto("/");
    await gpuPage.waitForFunction(
      () => /PARITY_(PASS|FAIL|ERROR)/.test(document.title),
      undefined,
      { timeout: 120_000 },
    );

    const report = await gpuPage.evaluate(
      () => (window as unknown as { __PARITY_REPORT__?: ParityReport }).__PARITY_REPORT__,
    );
    expect(report, "window.__PARITY_REPORT__ is populated").toBeTruthy();

    const label = [
      report!.adapter.vendor,
      report!.adapter.architecture,
      report!.adapter.device,
      report!.adapter.description,
    ].join(" ");
    const expectRe = GPU_MATRIX[test.info().project.name]?.expect;
    if (expectRe) {
      expect(label, `adapter "${label}" matches ${expectRe}`).toMatch(expectRe);
    }

    const failed = report!.results.filter((r) => !r.passed);
    expect(
      report!.allCleanPassed,
      `clean parity failed for: ${failed.map((r) => `${r.kernel} maxAbsDiff=${r.maxAbsDiff}`).join(", ")}`,
    ).toBe(true);
    expect(
      report!.allFaultsDetected,
      "fault-injected legs were not all detected as failing",
    ).toBe(true);
  });

  // Optional heavier smoke: real 320 MB model load plus a short decode.
  // Off by default - enable with GPU_MATRIX_GENERATE=1.
  test("short generation on the same adapter", async ({ gpuPage }) => {
    test.skip(!process.env.GPU_MATRIX_GENERATE, "set GPU_MATRIX_GENERATE=1 for the model-load smoke");
    test.setTimeout(600_000);
    await gpuPage.goto("/");
    await gpuPage.waitForFunction(
      () => /PARITY_(PASS|FAIL|ERROR)/.test(document.title),
      undefined,
      { timeout: 120_000 },
    );

    const out = await gpuPage.evaluate(async () => {
      const gen = (window as unknown as {
        __SMOKE_GENERATE__?: (
          p: string,
          n: number,
        ) => Promise<{ tokens: number[]; text: string; avgTokPerSec: number }>;
      }).__SMOKE_GENERATE__;
      if (!gen) return null;
      return gen("online softmax", 24);
    });
    expect(out, "window.__SMOKE_GENERATE__ is available").toBeTruthy();
    expect(out!.tokens.length).toBeGreaterThan(0);
    expect(out!.avgTokPerSec).toBeGreaterThan(0);
  });
});
