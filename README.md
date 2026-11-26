# omarchy-mlx WebGPU

**Live demo:** https://joshuaswarren.github.io/omarchy-mlx-webgpu/

Run **`mlx-community/Qwen3-0.6B-4bit`** (28 layers, `hidden=1024`, 16 Q / 8 KV heads, `head_dim=128`, MLX affine 4-bit `group_size=64`) directly on your browser GPU through compute kernels ported from [omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx) to portable WebGPU WGSL.

Zero backend servers. Zero paid APIs. Weights stream once from Hugging Face and stay in your browser cache.

## What you can inspect live

1. **Zoom Level 1 — Token stream**: Watch `Qwen3-0.6B-4bit` generate text live on your GPU with per-token `tok/s`, live `dispatches/token`, and a scrolling throughput/dispatch waveform.
2. **Zoom Level 2 — Kernel graph**: Inspect all 28 transformer layers and the per-layer dispatch waterfall (`537` unfused dispatches/token down to `397` with all three fusions enabled).
3. **Zoom Level 3 — Dispatch & shader inspector**: Click any kernel to inspect its workgroup grid, 64-dim WebGPU-vs-CPU diff heatmap, and side-by-side `omarchy-mlx` Vulkan GLSL vs browser WebGPU WGSL source.
4. **Real optimization toggles**:
   - **[PR #997](https://github.com/joshuaswarren/omarchy-mlx/pull/997) one-dispatch online-softmax attention** (`sdpa_decode_online.wgsl` vs 2-dispatch `sdpa_decode_split_p1/p2.wgsl`, `-28` dispatches/token)
   - **Fused QK-Norm + RoPE** (`fast_rope_norm.wgsl`, `-56` dispatches/token)
   - **Fused Gate+Up Q4 QMM + SwiGLU** (`qmm_vec_swiglu.wgsl`, `-56` dispatches/token)
5. **Fault switch (`F`)**: Invert the online-softmax rescale mid-generation (TDR-safe) to watch live text corrupt and the in-page CPU parity check turn red immediately; flip it back off to recover.
6. **In-page CPU parity**: Verifies all 10 WebGPU kernels (`sdpa_decode_online`, `sdpa_decode_split`, `qmm_vec`, `fast_norm`, `fast_rope_norm`, `qmm_vec_swiglu`, `fast_rope`, `swiglu`, `embed_q4`, `argmax`) against an independent Float64/Float32 CPU reference inside the page.

## Measured numbers

### Browser WebGPU (`mlx-community/Qwen3-0.6B-4bit`, greedy decode, Apple Silicon Metal-3 WebGPU)

| Configuration | Dispatches / token | Attention dispatches | Decode speed | Greedy output vs CPU ref |
|---|---:|---:|---:|---|
| Unfused baseline (2-dispatch split-K SDPA, separate QK-Norm/RoPE, separate Gate/Up/SwiGLU) | 537 | 56 | 43.8 tok/s | Identical (32/32 tokens) |
| Optimized (PR #997 1-dispatch online SDPA + Fused QK-Norm+RoPE + Fused SwiGLU QMM) | 397 (`-140`) | 28 (`-28`) | 68.7 tok/s (`+57%`) | Identical (32/32 tokens) |

### Upstream `omarchy-mlx` Linux Vulkan reference (public [PR #997](https://github.com/joshuaswarren/omarchy-mlx/pull/997))

Apple M1 Max 16-inch (32-core GPU, 64 GB), `Qwen3-4B-Instruct-2507-4bit`, Mesa Honeykrisp Vulkan:

- `ctx=256`: `85.32 -> 92.97 tok/s` (`+8.97%`, 5/5 wins)
- `ctx=512`: `84.50 -> 91.71 tok/s` (`+8.53%`, 5/5 wins; macOS Metal ref `104.53 tok/s`)
- `ctx=1024`: `80.35 -> 88.96 tok/s` (`+10.71%`, 5/5 wins)
- `ctx=2048`: `75.49 -> 83.72 tok/s` (`+10.91%`, 5/5 wins)

## Development

```bash
npm install
npm run dev
npm run build
npm run test:cpu
```

## License

MIT
