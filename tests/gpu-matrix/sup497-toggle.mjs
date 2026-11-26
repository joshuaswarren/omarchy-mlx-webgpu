// SUP-497: #997 toggle leg runner — argv[2] = "on"|"off".
import { chromium } from "@playwright/test";
const SWIFT_ARGS = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--use-webgpu-adapter=swiftshader", "--enable-unsafe-swiftshader"];
const attn = process.argv[2] === "on";
const browser = await chromium.launch({ args: SWIFT_ARGS });
const page = await browser.newPage();
await page.goto("http://127.0.0.1:4173/?tab=tuning");
await page.waitForFunction(() => /PARITY_(PASS|FAIL|ERROR)/.test(document.title), undefined, { timeout: 180_000 });
const adapter = await page.evaluate(() => window.omarchyLab.getState().adapter);
await page.evaluate((a) => window.omarchyLab.set({ oneDispatchAttention: a }), attn);
const res = await page.evaluate(() => window.omarchyLab.run({ tokens: 16 }));
console.log(JSON.stringify({ adapter, oneDispatchAttention: attn, result: res }));
await browser.close();
