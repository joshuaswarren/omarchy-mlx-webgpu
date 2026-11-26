// Ported from joshuaswarren/omarchy-mlx:
//   overlay/mlx/backend/omarchy/shaders/fast_rope.comp
// SPDX-License-Identifier: MIT
//
// Half-split Rotary Positional Embedding (Qwen3 traditional=false):
//   For each head h in [0, heads) and pair i in [0, head_dim / 2):
//     freq  = exp(-f32(i) * (log(rope_theta) / f32(head_dim / 2)))
//     angle = f32(position) * scale * freq
//     y[i]            = x[i] * cos(angle) - x[i + half] * sin(angle)
//     y[i + half]     = x[i + half] * cos(angle) + x[i] * sin(angle)
//
// Portable WebGPU WGSL: f32 arithmetic, 0 B workgroup memory, no subgroup ext.

struct Params {
  heads: u32,
  head_dim: u32,
  position: u32,
  fault_mode: u32,
  rope_theta: f32,
  scale: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> src_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const INV_TWO_PI: f32 = 0.15915494309189535;
const TWO_PI_HI: f32 = 6.283185005187988;  // exact 21-bit top of 2*pi
const TWO_PI_LO: f32 = 3.0199160587e-7;    // 2*pi - TWO_PI_HI

fn reduce_angle(angle: f32) -> f32 {
  let k_turn = round(angle * INV_TWO_PI);
  return (angle - k_turn * TWO_PI_HI) - k_turn * TWO_PI_LO;
}

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  let head_dim = params.head_dim;
  let half_dim = head_dim >> 1u;
  if (half_dim == 0u) {
    return;
  }

  let log2_theta_over_half = log2(params.rope_theta) / f32(half_dim);
  let pos_scaled = f32(params.position) * params.scale;

  for (var h = wg.x; h < params.heads; h = h + num_wg.x) {
    let base = h * head_dim;
    for (var pair_idx = lid.x; pair_idx < half_dim; pair_idx = pair_idx + 64u) {
      let freq = exp2(-f32(pair_idx) * log2_theta_over_half);
      let angle = reduce_angle(pos_scaled * freq);
      let c = cos(angle);
      // Fault mode flips the rotation sign (applies inverse RoPE) without OOB/NaN.
      let s_raw = sin(angle);
      let s = select(s_raw, -s_raw, params.fault_mode != 0u);

      let x0 = src_data[base + pair_idx];
      let x1 = src_data[base + pair_idx + half_dim];
      dst_data[base + pair_idx] = x0 * c - x1 * s;
      dst_data[base + pair_idx + half_dim] = x1 * c + x0 * s;
    }
  }
}
