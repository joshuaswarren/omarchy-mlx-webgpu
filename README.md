# omarchy-mlx-webgpu

Run a real language model (`mlx-community/Qwen3-0.6B-4bit`) directly on your browser GPU using compute kernels ported from [omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx) (`RMSNorm`, `RoPE`, 4-bit affine `QuantizedMatmul`, and [PR #997](https://github.com/joshuaswarren/omarchy-mlx/pull/997)'s single-dispatch online-softmax decode attention).

## What is in this demo

- **Real model in the browser**: `mlx-community/Qwen3-0.6B-4bit` (28 layers, `hidden=1024`, 16 Q heads / 8 KV heads, `head_dim=128`, MLX 4-bit `group_size=64`) loaded straight from Hugging Face safetensors with browser caching.
- **3-level live zoom**: token stream -> kernel execution graph -> per-dispatch workgroup & WGSL/GLSL shader inspector.
- **Real optimization toggles**: toggle [PR #997](https://github.com/joshuaswarren/omarchy-mlx/pull/997) one-dispatch online-softmax attention vs 2-dispatch split-K attention (plus fused QK-Norm+RoPE and fused SwiGLU QMM) and watch live dispatches/token and live tok/s change on your GPU.
- **Live kernel fault switch**: inject a safe attention softmax fault mid-generation to watch output corrupt and CPU parity flip red in real time, then switch it back off.
- **In-page CPU parity**: verify WebGPU kernel outputs directly against an independent Float64/Float32 CPU reference in your browser.

## Development

```bash
npm install
npm run dev
npm run build
npm run test:cpu
```

## License

MIT
