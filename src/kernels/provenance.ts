import type { DispatchEvent, KernelId, OptFlags } from "./types";
import { QWEN3_0_6B_4BIT } from "../model/config";
import fastNormWgsl from "./wgsl/fast_norm.wgsl?raw";
import sdpaOnlineWgsl from "./wgsl/sdpa_decode_online.wgsl?raw";
import sdpaSplitP1Wgsl from "./wgsl/sdpa_decode_split_p1.wgsl?raw";
import sdpaSplitP2Wgsl from "./wgsl/sdpa_decode_split_p2.wgsl?raw";

export type KernelFamily = "attention" | "qmm" | "norm_rope" | "swiglu" | "memory";

export interface KernelProvenance {
  id: KernelId;
  title: string;
  shortLabel: string;
  family: KernelFamily;
  omarchyPath: string;
  omarchyUrl: string;
  prNumber?: number;
  prUrl?: string;
  wgslPath: string;
  workgroupSize: [number, number, number];
  summary: string;
  portNotes: string;
  glslSnippet: string;
  wgslSnippet: string;
}

const REPO_BLOB = "https://github.com/joshuaswarren/omarchy-mlx/blob/main";

function extractHotSection(source: string, anchor: string, before = 4, maxLines = 22): string {
  const lines = source.trim().split("\n");
  const hit = lines.findIndex((l) => l.includes(anchor));
  if (hit < 0) {
    return lines.slice(0, maxLines).join("\n");
  }
  const start = Math.max(0, hit - before);
  const end = Math.min(lines.length, start + maxLines);
  const slice = lines.slice(start, end);
  const nonEmpty = slice.filter((l) => l.trim().length > 0);
  const minIndent = nonEmpty.reduce(
    (min, l) => Math.min(min, l.match(/^ */)?.[0].length ?? 0),
    Infinity,
  );
  const dedented =
    Number.isFinite(minIndent) && minIndent > 0
      ? slice.map((l) => l.slice(minIndent))
      : slice;
  const prefix = start > 0 ? [`// lines ${start + 1}-${end} of ${lines.length}`] : [];
  return [...prefix, ...dedented].join("\n");
}

export const KERNEL_PROVENANCE: Record<KernelId, KernelProvenance> = {
  sdpa_decode_online: {
    id: "sdpa_decode_online",
    title: "PR #997 One-Dispatch Online-Softmax SDPA",
    shortLabel: "SDPA online",
    family: "attention",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/sdpa_decode_online.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/sdpa_decode_online.comp`,
    prNumber: 997,
    prUrl: "https://github.com/joshuaswarren/omarchy-mlx/pull/997",
    wgslPath: "src/kernels/wgsl/sdpa_decode_online.wgsl",
    workgroupSize: [256, 1, 1],
    summary:
      "Single-pass online-softmax decode attention: one workgroup per query head, zero scratch buffer, zero Pass-2 dispatch.",
    portNotes:
      "Replaces GL_KHR_shader_subgroup arithmetic with portable 32-lane workgroup shared-memory tree reduction so it runs identically across Metal, Vulkan, and D3D12 WebGPU adapters.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/sdpa_decode_online.comp (PR #997)
layout(local_size_x = SUBGROUPS * 32, local_size_y = 1, local_size_z = 1) in;

for (uint key = sg + uint(SUBGROUPS); key < k_len; key += uint(SUBGROUPS)) {
  kw = key_data.values[k_idx];
  vw = value_data.values[v_idx];
  dot = fma(q0, uintBitsToFloat(kw.x << 16u),
        fma(q1, uintBitsToFloat(kw.x & 0xffff0000u),
        fma(q2, uintBitsToFloat(kw.y << 16u),
            q3 * uintBitsToFloat(kw.y & 0xffff0000u))));
  float score = subgroupAdd(dot);
  float new_max = max(m_i, score);
  float factor = exp2(m_i - new_max);
  float p = exp2(score - new_max);
  m_i = new_max;
  l_i = fma(l_i, factor, p);
  o0 = fma(o0, factor, p * uintBitsToFloat(vw.x << 16u));
  o1 = fma(o1, factor, p * uintBitsToFloat(vw.x & 0xffff0000u));
  o2 = fma(o2, factor, p * uintBitsToFloat(vw.y << 16u));
  o3 = fma(o3, factor, p * uintBitsToFloat(vw.y & 0xffff0000u));
}
barrier();
// 128 output threads merge SUBGROUPS partial (max, sum, o) states directly`,
    wgslSnippet: extractHotSection(sdpaOnlineWgsl, "let new_max = max", 2, 18),
  },

  sdpa_decode_split_p1: {
    id: "sdpa_decode_split_p1",
    title: "Baseline Split-K Decode SDPA (Pass 1)",
    shortLabel: "SDPA p1",
    family: "attention",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/sdpa_decode_split_p1.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/sdpa_decode_split_p1.comp`,
    wgslPath: "src/kernels/wgsl/sdpa_decode_split_p1.wgsl",
    workgroupSize: [128, 1, 1],
    summary:
      "First pass of the 2-dispatch split-K baseline: computes block max, exp sum, and weighted V per 64-key chunk into a global scratch buffer.",
    portNotes:
      "Requires a global scratch write of (head_dim + 4) floats per (head, block) plus a full barrier before Pass 2 can reduce.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/sdpa_decode_split_p1.comp
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
shared float s_q[SPLIT_R * D];
shared float s_p[SPLIT_R * BK];
shared float s_max[SPLIT_R * SUBGROUPS];
shared float s_sum[SPLIT_R * SUBGROUPS];

// Each workgroup owns (kv_head, key_block) and writes (D + 4) floats
// to scratch_data for sdpa_decode_split_p2.comp to merge in a second pass.`,
    wgslSnippet: extractHotSection(sdpaSplitP1Wgsl, "Softmax numerators", 4, 28),
  },

  sdpa_decode_split_p2: {
    id: "sdpa_decode_split_p2",
    title: "Baseline Split-K Decode SDPA (Pass 2 Merge)",
    shortLabel: "SDPA p2",
    family: "attention",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/sdpa_decode_split_p2.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/sdpa_decode_split_p2.comp`,
    wgslPath: "src/kernels/wgsl/sdpa_decode_split_p2.wgsl",
    workgroupSize: [128, 1, 1],
    summary:
      "Second pass of the split-K baseline: folds per-block (max, sum, weighted V) across scratch memory. Eliminated completely by PR #997.",
    portNotes:
      "Removed (-28 dispatches/token on 28-layer Qwen3-0.6B; -36/token on 36-layer Qwen3-4B) when PR #997 online softmax is active.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/sdpa_decode_split_p2.comp
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
for (uint b = t; b < nblocks; b += 128u) {
  float m = scratch_data.values4[(base + b * STRIDE) / 4u + D / 4u].x;
  part_max = isnan(m) ? uintBitsToFloat(0x7f800000u) : max(part_max, m);
}
part_max = subgroupMax(part_max);
// Rescale every block weighted-V row by exp(block_max - global_max) and divide.`,
    wgslSnippet: extractHotSection(sdpaSplitP2Wgsl, "var global_max", 4, 26),
  },

  fast_norm: {
    id: "fast_norm",
    title: "Fused RMSNorm (fast_norm)",
    shortLabel: "RMSNorm",
    family: "norm_rope",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/fast_norm.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/fast_norm.comp`,
    wgslPath: "src/kernels/wgsl/fast_norm.wgsl",
    workgroupSize: [256, 1, 1],
    summary:
      "One 256-thread workgroup per row computes float32 sum-of-squares tree reduction, inverseSqrt(mean + eps), and weighted scale.",
    portNotes:
      "Direct 1:1 port of the stride-128..1 workgroup tree reduction and exact float32 arithmetic.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/fast_norm.comp
layout(local_size_x = 256) in;
shared float block_values[256];

float local_sq = 0.0;
for (uint index = thread; index < row_length; index += 256u) {
  float value = LOAD_VALUE(input_data.values[start + index]);
  local_sq += value * value;
}
block_values[thread] = local_sq;
barrier();
for (uint stride = 128u; stride > 0u; stride >>= 1u) {
  if (thread < stride) block_values[thread] += block_values[thread + stride];
  barrier();
}
float norm = inversesqrt(block_values[0] / float(row_length) + eps);`,
    wgslSnippet: extractHotSection(fastNormWgsl, "let mean_square", 12, 26),
  },

  fast_rope: {
    id: "fast_rope",
    title: "Rotary Position Embedding (fast_rope)",
    shortLabel: "RoPE",
    family: "norm_rope",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/fast_rope.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/fast_rope.comp`,
    wgslPath: "src/kernels/wgsl/fast_rope.wgsl",
    workgroupSize: [128, 1, 1],
    summary:
      "Applies half-split rotary angles (cos/sin at theta = 1e6) across Q and K heads with no shared memory or barriers.",
    portNotes:
      "Standalone path used when Fused QK-Norm+RoPE is turned off; adds 56 dispatches/token across 28 layers.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/fast_rope.comp
// Half-split rotary embedding matching mlx::core::fast::RoPE
float inv_freq = exp(-float(pair_idx) * params.beta);
float theta = float(pos) * params.alpha * inv_freq;
float c = cos(theta);
float s = sin(theta);
float y0 = x0 * c - x1 * s;
float y1 = x0 * s + x1 * c;`,
    wgslSnippet: `// Ported from omarchy-mlx fast_rope.comp
@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let pair = lid.x; // 0..63 for head_dim=128
  if (pair >= 64u) { return; }
  let inv_freq = exp(-f32(pair) * params.log_theta_over_half);
  let angle = f32(params.position) * inv_freq;
  let c = cos(angle);
  let s = sin(angle);
  let base = head * 128u;
  let x0 = data[base + pair];
  let x1 = data[base + pair + 64u];
  out_data[base + pair] = x0 * c - x1 * s;
  out_data[base + pair + 64u] = x0 * s + x1 * c;
}`,
  },

  fast_rope_norm: {
    id: "fast_rope_norm",
    title: "Fused Per-Head QK-Norm + RoPE (fast_rope_norm)",
    shortLabel: "Norm+RoPE",
    family: "norm_rope",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/fast_rope_norm.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/fast_rope_norm.comp`,
    wgslPath: "src/kernels/wgsl/fast_rope_norm.wgsl",
    workgroupSize: [128, 1, 1],
    summary:
      "Fuses per-head RMSNorm and rotary embedding in one workgroup pass per Q/K head, eliminating 2 intermediate writes and 56 dispatches/token.",
    portNotes:
      "Keeps the normalized head vector in workgroup shared memory and rotates in-register before a single global store.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/fast_rope_norm.comp
// Fused Q/K RMSNorm + RoPE: one workgroup owns one head row (D = 128)
float mean_square = block_values[0] / float(row_length);
barrier();
precise float norm = inversesqrt(mean_square + eps);
precise float value = rope_load(params.lhs_offset + in_base + thread);
precise float weight = weight_load(params.aux_offset + thread);
rounded[thread] = rope_round((weight * norm) * value);
barrier();
// Rotate (rounded[i], rounded[i + D/2]) by (cos(theta), sin(theta)) in-place`,
    wgslSnippet: `// Ported from omarchy-mlx fast_rope_norm.comp
@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let t = lid.x;
  let v = input_data[head * 128u + t];
  s_sq[t] = v * v;
  workgroupBarrier();
  // 128-lane tree sum -> inv_rms
  let inv_rms = inverseSqrt(s_sq[0] / 128.0 + params.eps);
  s_norm[t] = v * inv_rms * weight_data[t];
  workgroupBarrier();
  if (t < 64u) {
    let angle = f32(params.position) * exp(-f32(t) * params.log_theta_over_half);
    let c = cos(angle); let s = sin(angle);
    out_data[head * 128u + t] = s_norm[t] * c - s_norm[t + 64u] * s;
    out_data[head * 128u + t + 64u] = s_norm[t] * s + s_norm[t + 64u] * c;
  }
}`,
  },

  qmm_vec: {
    id: "qmm_vec",
    title: "4-Bit Affine Quantized GEMV (qmm_vec)",
    shortLabel: "Q4 GEMV",
    family: "qmm",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/qmm_vec.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/qmm_vec.comp`,
    wgslPath: "src/kernels/wgsl/qmm_vec.wgsl",
    workgroupSize: [256, 1, 1],
    summary:
      "Decodes packed 4-bit weights (8 nibbles per uint32, group_size=64 affine scale + bias) in registers inside the dot product.",
    portNotes:
      "Every 32-lane slot walks contiguous packed weight words and input activations for coalesced memory bandwidth.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/qmm_vec.comp (QMM_VEC_Q4_WORD)
// Unpack 8 affine 4-bit nibbles per uint32 word in registers (group_size = 64)
uint packed = weights.values[row * words_per_row + w];
float scale = LOAD_VALUE(scales.values[row * groups_per_row + g]);
float bias  = LOAD_VALUE(biases.values[row * groups_per_row + g]);
float dot = fma(x3, float((packed >> 12u) & 0xfu),
            fma(x2, float((packed >> 8u)  & 0xfu),
            fma(x1, float((packed >> 4u)  & 0xfu),
                x0 * float(packed & 0xfu))));
acc += fma(scale, dot, input_sum * bias);`,
    wgslSnippet: `// Ported from omarchy-mlx qmm_vec.comp (MLX 4-bit group_size=64)
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  // 8 groups of 32 lanes per workgroup; each group reduces one output row
  let word = w_packed[row * words_per_row + w_idx];
  let g = (w_idx * 8u) / 64u;
  let scale = scales[row * groups_per_row + g];
  let bias = biases[row * groups_per_row + g];
  var acc = 0.0;
  for (var n = 0u; n < 8u; n = n + 1u) {
    let q = f32((word >> (n * 4u)) & 0xfu);
    acc += x[w_idx * 8u + n] * (q * scale + bias);
  }
}`,
  },

  qmm_vec_swiglu: {
    id: "qmm_vec_swiglu",
    title: "Fused Gate + Up Q4 GEMV + SwiGLU (QMM_VEC_SWIGLU)",
    shortLabel: "Q4 SwiGLU",
    family: "swiglu",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/qmm_vec.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/qmm_vec.comp#L2225`,
    wgslPath: "src/kernels/wgsl/qmm_vec_swiglu.wgsl",
    workgroupSize: [256, 1, 1],
    summary:
      "Reads the MLP input vector once, computes both Gate and Up 4-bit dot products in lockstep, and applies SiLU(gate) * up before storing.",
    portNotes:
      "Replaces 3 dispatches (gate_proj + up_proj + swiglu) with 1 dispatch per layer (-56 dispatches/token) and removes 2 intermediate MLP buffers.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/qmm_vec.comp (#elif QMM_VEC_SWIGLU)
// Single pass over input_x feeds both Gate (w0) and Up (w1) Q4 dot products
Q4_QUAD_SWIGLU(wvg[r].x, wvu[r].x, xa.x, xa.y)
Q4_QUAD_SWIGLU(wvg[r].x >> 16u, wvu[r].x >> 16u, xa.z, xa.w)
acc_g[r] += fma(sg[r], dot_g[r], input_sum * bg[r]);
acc_u[r] += fma(su[r], dot_u[r], input_sum * bu[r]);
// Epilogue: out = silu(acc_g) * acc_u in registers (zero intermediate store)`,
    wgslSnippet: `// Ported from omarchy-mlx qmm_vec.comp (QMM_VEC_SWIGLU)
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  // Stream input x once against gate_w and up_w simultaneously
  let gate_val = reduce_group(acc_g);
  let up_val = reduce_group(acc_u);
  if (lane == 0u && row < params.n) {
    let silu = gate_val / (1.0 + exp(-gate_val));
    output[row] = silu * up_val;
  }
}`,
  },

  swiglu: {
    id: "swiglu",
    title: "Standalone SwiGLU Activation (swiglu)",
    shortLabel: "SwiGLU",
    family: "swiglu",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/swiglu.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/swiglu.comp`,
    wgslPath: "src/kernels/wgsl/swiglu.wgsl",
    workgroupSize: [256, 1, 1],
    summary:
      "Vectorized elementwise silu(gate) * up over intermediate_size=3072. Used only when Fused SwiGLU QMM is switched off.",
    portNotes:
      "Eliminated when QMM_VEC_SWIGLU fuses gate_proj, up_proj, and activation into one kernel.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/swiglu.comp
layout(local_size_x = 256) in;
vec4 gate = LOAD_VEC(gate_buf.values[index]);
vec4 up   = LOAD_VEC(up_buf.values[index]);
vec4 sig  = round4(1.0 / (1.0 + exp(-gate)));
vec4 silu = round4(gate * sig);
output_data.values[index] = STORE_VEC(round4(silu * up));`,
    wgslSnippet: `// Ported from omarchy-mlx swiglu.comp
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let g = gate_buf[i];
  out_buf[i] = (g / (1.0 + exp(-g))) * up_buf[i];
}`,
  },

  embed_q4: {
    id: "embed_q4",
    title: "4-Bit Token Embedding Lookup",
    shortLabel: "Embed Q4",
    family: "memory",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/dequant.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/dequant.comp`,
    wgslPath: "src/kernels/wgsl/embed_q4.wgsl",
    workgroupSize: [256, 1, 1],
    summary:
      "Dequantizes one 1024-wide row from the 4-bit affine token embedding table (151,936 vocab entries) on the GPU.",
    portNotes: "Runs once per decode step before Layer 0.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/dequant.comp
uint word = weights[token_id * words_per_row + (col >> 3u)];
float q = float((word >> ((col & 7u) * 4u)) & 0xfu);
output[col] = fma(q, scales[group_idx], biases[group_idx]);`,
    wgslSnippet: `// Q4 affine single-row embedding dequantization
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  if (col >= params.hidden_size) { return; }
  let word = embed_w[params.token_id * (params.hidden_size / 8u) + (col >> 3u)];
  let q = f32((word >> ((col & 7u) * 4u)) & 0xfu);
  let g = params.token_id * (params.hidden_size / 64u) + (col >> 6u);
  hidden[col] = q * embed_scales[g] + embed_biases[g];
}`,
  },

  residual_add: {
    id: "residual_add",
    title: "Residual Vector Add",
    shortLabel: "ResAdd",
    family: "memory",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/binary_vec.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/binary_vec.comp`,
    wgslPath: "src/kernels/wgsl/residual_add.wgsl",
    workgroupSize: [256, 1, 1],
    summary: "Accumulates attention or MLP projection output back onto the 1024-dim residual stream.",
    portNotes: "Runs twice per transformer layer (post-attention and post-MLP).",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/binary_vec.comp
uint idx = gl_GlobalInvocationID.x;
out_data.values[idx] = lhs.values[idx] + rhs.values[idx];`,
    wgslSnippet: `@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.count) {
    residual[i] = residual[i] + delta[i];
  }
}`,
  },

  kv_cache_write: {
    id: "kv_cache_write",
    title: "KV Cache Append",
    shortLabel: "KV Write",
    family: "memory",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/slice_update_pair.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/slice_update_pair.comp`,
    wgslPath: "src/kernels/wgsl/kv_cache_write.wgsl",
    workgroupSize: [128, 1, 1],
    summary: "Appends the newly rotated K and V vectors (8 KV heads x 128 dims) at slot `pos` in the layer KV cache.",
    portNotes: "Writes K and V in one dispatch so SDPA immediately sees the current token.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/slice_update_pair.comp
uint dst = kv_head * max_seq * D + pos * D + d;
k_cache[dst] = k_new[kv_head * D + d];
v_cache[dst] = v_new[kv_head * D + d];`,
    wgslSnippet: `@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let kv_head = wg.x;
  let d = lid.x;
  let dst = kv_head * params.max_seq * 128u + params.pos * 128u + d;
  k_cache[dst] = k_in[kv_head * 128u + d];
  v_cache[dst] = v_in[kv_head * 128u + d];
}`,
  },

  argmax: {
    id: "argmax",
    title: "Greedy Logit Argmax",
    shortLabel: "Argmax",
    family: "memory",
    omarchyPath: "overlay/mlx/backend/omarchy/shaders/argreduce_chunk.comp",
    omarchyUrl: `${REPO_BLOB}/overlay/mlx/backend/omarchy/shaders/argreduce_chunk.comp`,
    wgslPath: "src/kernels/wgsl/argmax.wgsl",
    workgroupSize: [256, 1, 1],
    summary: "Selects the highest-scoring token ID over 151,936 vocabulary logits on the GPU.",
    portNotes: "Returns a single uint32 token ID back to the host per decode step.",
    glslSnippet: `// overlay/mlx/backend/omarchy/shaders/argreduce_chunk.comp
// Shared-memory tree argmax over vocab logits (lowest index wins ties)`,
    wgslSnippet: `@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  // Strided reduction across 151,936 logits -> single token_id
}`,
  },
};

export interface DispatchSummary {
  totalDispatches: number;
  baselineDispatches: number;
  savedDispatches: number;
  attentionDispatches: number;
  perLayerDispatches: number;
  baselinePerLayer: number;
}

export function countDispatchesForOpts(opts: OptFlags): DispatchSummary {
  const L = QWEN3_0_6B_4BIT.numHiddenLayers; // 28
  const baselinePerLayer = 19;
  const baselineDispatches = 4 + L * baselinePerLayer; // 536
  const perLayerDispatches =
    baselinePerLayer -
    (opts.oneDispatchAttention ? 1 : 0) -
    (opts.fusedRopeNorm ? 2 : 0) -
    (opts.fusedSwigluQmm ? 2 : 0);
  const totalDispatches = 4 + L * perLayerDispatches;
  return {
    totalDispatches,
    baselineDispatches,
    savedDispatches: baselineDispatches - totalDispatches,
    attentionDispatches: L * (opts.oneDispatchAttention ? 1 : 2),
    perLayerDispatches,
    baselinePerLayer,
  };
}

/**
 * Builds the exact Qwen3-0.6B-4bit per-token dispatch schedule for the given
 * optimization flags and sequence length. Used before live generation starts
 * and whenever a toggle flips between tokens.
 */
export function buildSyntheticSchedule(opts: OptFlags, seqLen = 192): DispatchEvent[] {
  const cfg = QWEN3_0_6B_4BIT;
  const H = cfg.hiddenSize; // 1024
  const Q = cfg.numAttentionHeads; // 16
  const KV = cfg.numKeyValueHeads; // 8
  const D = cfg.headDim; // 128
  const I = cfg.intermediateSize; // 3072
  const V = cfg.vocabSize; // 151936
  const nblocks = Math.max(1, Math.ceil(seqLen / 64));
  const q4Bytes = (rows: number, cols: number) =>
    Math.round(rows * cols * 0.5 + rows * (cols / cfg.groupSize) * 4);

  const dispatches: DispatchEvent[] = [];
  let idx = 0;
  const push = (
    layer: number,
    kernel: KernelId,
    label: string,
    workgroups: [number, number, number],
    workgroupSize: [number, number, number],
    bytesRead: number,
    bytesWritten: number,
    durationUs: number,
  ) => {
    dispatches.push({
      index: idx++,
      layer,
      kernel,
      label,
      workgroups,
      workgroupSize,
      bytesRead,
      bytesWritten,
      durationUs,
    });
  };

  // Token embedding
  push(-1, "embed_q4", "embed_tokens (Q4)", [4, 1, 1], [256, 1, 1], q4Bytes(1, H), H * 4, 14);

  for (let layer = 0; layer < cfg.numHiddenLayers; layer++) {
    push(layer, "fast_norm", "input_layernorm", [1, 1, 1], [256, 1, 1], H * 8, H * 4, 11);
    push(layer, "qmm_vec", "q_proj (Q4)", [Math.ceil((Q * D) / 8), 1, 1], [256, 1, 1], q4Bytes(Q * D, H), Q * D * 4, 38);
    push(layer, "qmm_vec", "k_proj (Q4)", [Math.ceil((KV * D) / 8), 1, 1], [256, 1, 1], q4Bytes(KV * D, H), KV * D * 4, 24);
    push(layer, "qmm_vec", "v_proj (Q4)", [Math.ceil((KV * D) / 8), 1, 1], [256, 1, 1], q4Bytes(KV * D, H), KV * D * 4, 24);

    if (opts.fusedRopeNorm) {
      push(layer, "fast_rope_norm", "Q norm+RoPE (fused)", [Q, 1, 1], [128, 1, 1], Q * D * 4 + D * 4, Q * D * 4, 13);
      push(layer, "fast_rope_norm", "K norm+RoPE (fused)", [KV, 1, 1], [128, 1, 1], KV * D * 4 + D * 4, KV * D * 4, 11);
    } else {
      push(layer, "fast_norm", "q_norm", [Q, 1, 1], [256, 1, 1], Q * D * 4 + D * 4, Q * D * 4, 12);
      push(layer, "fast_rope", "q_rope", [Q, 1, 1], [128, 1, 1], Q * D * 4, Q * D * 4, 11);
      push(layer, "fast_norm", "k_norm", [KV, 1, 1], [256, 1, 1], KV * D * 4 + D * 4, KV * D * 4, 10);
      push(layer, "fast_rope", "k_rope", [KV, 1, 1], [128, 1, 1], KV * D * 4, KV * D * 4, 10);
    }

    push(layer, "kv_cache_write", "kv_cache append", [KV, 1, 1], [128, 1, 1], KV * D * 8, KV * D * 8, 9);

    if (opts.oneDispatchAttention) {
      push(
        layer,
        "sdpa_decode_online",
        "PR #997 SDPA online (1-dispatch)",
        [Q, 1, 1],
        [256, 1, 1],
        Q * D * 4 + KV * seqLen * D * 8,
        Q * D * 4,
        29,
      );
    } else {
      push(
        layer,
        "sdpa_decode_split_p1",
        "SDPA split-K pass 1",
        [Q, nblocks, 1],
        [128, 1, 1],
        Q * D * 4 + KV * seqLen * D * 8,
        Q * nblocks * (D + 4) * 4,
        33,
      );
      push(
        layer,
        "sdpa_decode_split_p2",
        "SDPA split-K pass 2 merge",
        [Q, 1, 1],
        [128, 1, 1],
        Q * nblocks * (D + 4) * 4,
        Q * D * 4,
        19,
      );
    }

    push(layer, "qmm_vec", "o_proj (Q4)", [Math.ceil(H / 8), 1, 1], [256, 1, 1], q4Bytes(H, Q * D), H * 4, 38);
    push(layer, "residual_add", "residual_add (attn)", [Math.ceil(H / 256), 1, 1], [256, 1, 1], H * 8, H * 4, 8);
    push(layer, "fast_norm", "post_attn_layernorm", [1, 1, 1], [256, 1, 1], H * 8, H * 4, 11);

    if (opts.fusedSwigluQmm) {
      push(
        layer,
        "qmm_vec_swiglu",
        "gate+up+SwiGLU (fused Q4)",
        [Math.ceil(I / 8), 1, 1],
        [256, 1, 1],
        q4Bytes(I, H) * 2 + H * 4,
        I * 4,
        74,
      );
    } else {
      push(layer, "qmm_vec", "gate_proj (Q4)", [Math.ceil(I / 8), 1, 1], [256, 1, 1], q4Bytes(I, H) + H * 4, I * 4, 48);
      push(layer, "qmm_vec", "up_proj (Q4)", [Math.ceil(I / 8), 1, 1], [256, 1, 1], q4Bytes(I, H) + H * 4, I * 4, 48);
      push(layer, "swiglu", "swiglu activation", [Math.ceil(I / 256), 1, 1], [256, 1, 1], I * 8, I * 4, 14);
    }

    push(layer, "qmm_vec", "down_proj (Q4)", [Math.ceil(H / 8), 1, 1], [256, 1, 1], q4Bytes(H, I) + I * 4, H * 4, 46);
    push(layer, "residual_add", "residual_add (mlp)", [Math.ceil(H / 256), 1, 1], [256, 1, 1], H * 8, H * 4, 8);
  }

  // Output head
  push(-1, "fast_norm", "final_norm", [1, 1, 1], [256, 1, 1], H * 8, H * 4, 12);
  push(-1, "qmm_vec", "lm_head (151,936 vocab Q4)", [Math.ceil(V / 8), 1, 1], [256, 1, 1], q4Bytes(V, H), V * 4, 420);
  push(-1, "argmax", "greedy_argmax", [1, 1, 1], [256, 1, 1], V * 4, 4, 16);

  return dispatches;
}
