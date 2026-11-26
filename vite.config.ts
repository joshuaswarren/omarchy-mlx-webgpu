import { defineConfig } from "vite";

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? "/omarchy-mlx-webgpu/" : "./",
  build: {
    target: "es2022",
    outDir: "dist",
    sourcemap: true,
  },
});
