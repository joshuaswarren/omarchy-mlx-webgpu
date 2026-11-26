// Ported from joshuaswarren/omarchy-mlx:
//   overlay/mlx/backend/omarchy/shaders/fast_rope_norm.comp
// SPDX-License-Identifier: MIT
//
// Fused per-head Q/K RMSNorm + half-split RoPE in a single dispatch
// (eliminates 56 dispatches/token across 28 Qwen3 layers):
//   1. Compute row RMS over head_dim elements (portable 128-thread tree reduction).
//   2. Normalize & scale by per-dim weight into workgroup memory.
//   3. Rotate half-split pairs (i, i + head_dim/2) with theta=rope_theta and store.
//
// Portable WebGPU WGSL: f32 arithmetic, 1.5 KB workgroup memory, no subgroup ext.
// Supports any even head_dim <= 256 (Qwen3 uses head_dim = 128).

struct Params {
  heads: u32,
  head_dim: u32,
  position: u32,
  weight_broadcast: u32,
  eps: f32,
  rope_theta: f32,
  scale: f32,
  fault_mode: u32,
};

@group(0) @binding(0) var<storage, read> src_data: array<f32>;
@group(0) @binding(1) var<storage, read> weight_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_data: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> s_sq: array<f32, 128>;
var<workgroup> s_normed: array<f32, 256>;

const INV_TWO_PI: f32 = 0.15915494309189535;
const TWO_PI_HI: f32 = 6.283185005187988;  // exact 21-bit top of 2*pi
const TWO_PI_LO: f32 = 3.0199160587e-7;    // 2*pi - TWO_PI_HI

fn reduce_angle(angle: f32) -> f32 {
  let k_turn = round(angle * INV_TWO_PI);
  return (angle - k_turn * TWO_PI_HI) - k_turn * TWO_PI_LO;
}

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  let tid = lid.x;
  let head_dim = params.head_dim;
  let half_dim = head_dim >> 1u;
  if (half_dim == 0u || head_dim > 256u) {
    return;
  }

  let log2_theta_over_half = log2(params.rope_theta) / f32(half_dim);
  let pos_scaled = f32(params.position) * params.scale;

  for (var h = wg.x; h < params.heads; h = h + num_wg.x) {
    let base = h * head_dim;

    var local_sq = 0.0;
    for (var d = tid; d < head_dim; d = d + 128u) {
      let v = src_data[base + d];
      local_sq = local_sq + v * v;
    }
    s_sq[tid] = local_sq;
    workgroupBarrier();

    if (tid < 64u) { s_sq[tid] = s_sq[tid] + s_sq[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { s_sq[tid] = s_sq[tid] + s_sq[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { s_sq[tid] = s_sq[tid] + s_sq[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { s_sq[tid] = s_sq[tid] + s_sq[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { s_sq[tid] = s_sq[tid] + s_sq[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { s_sq[tid] = s_sq[tid] + s_sq[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) { s_sq[0] = s_sq[0] + s_sq[1]; }
    workgroupBarrier();

    let mean_sq = s_sq[0] / f32(head_dim);
    // Fault mode drops the RMS denominator (scale = 1.0) while staying finite.
    let inv_rms = select(inverseSqrt(mean_sq + params.eps), 1.0, params.fault_mode != 0u);

    for (var d = tid; d < head_dim; d = d + 128u) {
      let v = src_data[base + d];
      var w = weight_data[0];
      if (params.weight_broadcast == 0u) {
        w = weight_data[d];
      }
      s_normed[d] = (v * inv_rms) * w;
    }
    workgroupBarrier();

    for (var pair_idx = tid; pair_idx < half_dim; pair_idx = pair_idx + 128u) {
      let freq = exp2(-f32(pair_idx) * log2_theta_over_half);
      let angle = reduce_angle(pos_scaled * freq);
      let c = cos(angle);
      let s = sin(angle);

      let y0 = s_normed[pair_idx];
      let y1 = s_normed[pair_idx + half_dim];
      dst_data[base + pair_idx] = y0 * c - y1 * s;
      dst_data[base + pair_idx + half_dim] = y1 * c + y0 * s;
    }
    workgroupBarrier();
  }
}
