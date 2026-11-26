import { test as base, chromium, type Page } from "@playwright/test";
import { GPU_MATRIX, projectEnvKey } from "./matrix";

/**
 * gpuPage resolves the project's browser itself instead of using the default
 * page fixture, so remote mode can use connectOverCDP (Chrome DevTools
 * Protocol) rather than the Playwright server protocol behind connect().
 *
 *   GPU_CDP_<NAME>  - connect over CDP to a Chromium already running on the
 *                     GPU host (ws:// or http:// endpoint). The page must be
 *                     reachable from that host via GPU_BASE_URL.
 *   GPU_BASE_URL    - page URL the browser loads; skips the local webServer.
 *   unset           - local Chromium launched with the project's launchArgs;
 *                     VK_ICD_FILENAMES in the runner env selects the driver.
 */
export const test = base.extend<{ gpuPage: Page }>({
  gpuPage: async ({}, use, testInfo) => {
    const name = testInfo.project.name;
    const row = GPU_MATRIX[name];
    if (!row) throw new Error(`project "${name}" has no GPU_MATRIX row`);

    const cdp = process.env[`GPU_CDP_${projectEnvKey(name)}`];
    const baseURL =
      process.env.GPU_BASE_URL ??
      (testInfo.project.use?.baseURL as string | undefined) ??
      "http://127.0.0.1:4173";

    const browser = cdp
      ? await chromium.connectOverCDP(cdp)
      : await chromium.launch({ args: row.launchArgs });
    const context = await browser.newContext({
      baseURL,
      viewport: { width: 1440, height: 920 },
    });
    const page = await context.newPage();
    await use(page);
    await context.close();
    await browser.close();
  },
});

export { expect } from "@playwright/test";
