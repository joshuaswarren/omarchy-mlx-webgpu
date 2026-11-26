// Ported from joshuaswarren/omarchy-mlx:
//   overlay/mlx/backend/omarchy/shaders/qmm_vec.comp (QMM_VEC_SWIGLU path)
// SPDX-License-Identifier: MIT
//
// Fused Gate + Up affine 4-bit (group_size = 64) QMM + SwiGLU epilogue:
//   gate[n] = x @ dequant(W_gate[n])
//   up[n]   = x @ dequant(W_up[n])
//   dst[n]  = (gate[n] / (1 + exp(-gate[n]))) * up[n]
// Eliminates 2 intermediate buffers and 56 dispatches/token across 28 Qwen3 layers.
//
// Uses exactly 8 storage buffers + 1 uniform buffer (fits WebGPU's guaranteed
// maxStorageBuffersPerShaderStage = 8 on all adapters), 3 KB workgroup memory,
// no subgroup extensions, and 2D-safe workgroup indexing.

struct Params {
  n: u32,
  k: u32,
  group_size: u32,
  fault_mode: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};

@group(0) @binding(0) var<storage, read> x_data: array<f32>;
@group(0) @binding(1) var<storage, read> gate_w: array<u32>;
@group(0) @binding(2) var<storage, read> gate_scales: array<f32>;
@group(0) @binding(3) var<storage, read> gate_biases: array<f32>;
@group(0) @binding(4) var<storage, read> up_w: array<u32>;
@group(0) @binding(5) var<storage, read> up_scales: array<f32>;
@group(0) @binding(6) var<storage, read> up_biases: array<f32>;
@group(0) @binding(7) var<storage, read_write> dst_data: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

const SLOTS: u32 = 8u;
const TILE_K: u32 = 256u;

var<workgroup> s_x: array<f32, 256>;
var<workgroup> s_gate: array<f32, 256>;
var<workgroup> s_up: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  let tid = lid.x;
  let slot = tid >> 5u;   // 0..7
  let lane = tid & 31u;   // 0..31

  let wg_linear = wg.y * num_wg.x + wg.x;
  let row = wg_linear * SLOTS + slot;
  let row_valid = row < params.n;

  let k = params.k;
  let words_per_row = k >> 3u;
  let groups_per_row = k / params.group_size;
  let w_row_base = row * words_per_row;
  let g_row_base = row * groups_per_row;

  var acc_gate = 0.0;
  var acc_up = 0.0;

  for (var tile_k = 0u; tile_k < k; tile_k = tile_k + TILE_K) {
    let x_idx = tile_k + tid;
    if (x_idx < k) {
      s_x[tid] = x_data[x_idx];
    } else {
      s_x[tid] = 0.0;
    }
    workgroupBarrier();

    let w_col = (tile_k >> 3u) + lane;
    if (row_valid && w_col < words_per_row) {
      let g_col = (w_col << 3u) / params.group_size;
      let w_off = w_row_base + w_col;
      let g_off = g_row_base + g_col;

      let gw = gate_w[w_off];
      let gs = gate_scales[g_off];
      let gb = gate_biases[g_off];

      let uw = up_w[w_off];
      let us = up_scales[g_off];
      let ub = up_biases[g_off];

      let lx = lane << 3u;
      let x0 = s_x[lx + 0u];
      let x1 = s_x[lx + 1u];
      let x2 = s_x[lx + 2u];
      let x3 = s_x[lx + 3u];
      let x4 = s_x[lx + 4u];
      let x5 = s_x[lx + 5u];
      let x6 = s_x[lx + 6u];
      let x7 = s_x[lx + 7u];

      let sum_x = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));

      let dot_g =
          x0 * f32(gw & 0xfu)
        + x1 * f32((gw >> 4u) & 0xfu)
        + x2 * f32((gw >> 8u) & 0xfu)
        + x3 * f32((gw >> 12u) & 0xfu)
        + x4 * f32((gw >> 16u) & 0xfu)
        + x5 * f32((gw >> 20u) & 0xfu)
        + x6 * f32((gw >> 24u) & 0xfu)
        + x7 * f32((gw >> 28u) & 0xfu);
      acc_gate = acc_gate + dot_g * gs + sum_x * gb;

      let dot_u =
          x0 * f32(uw & 0xfu)
        + x1 * f32((uw >> 4u) & 0xfu)
        + x2 * f32((uw >> 8u) & 0xfu)
        + x3 * f32((uw >> 12u) & 0xfu)
        + x4 * f32((uw >> 16u) & 0xfu)
        + x5 * f32((uw >> 20u) & 0xfu)
        + x6 * f32((uw >> 24u) & 0xfu)
        + x7 * f32((uw >> 28u) & 0xfu);
      acc_up = acc_up + dot_u * us + sum_x * ub;
    }
    workgroupBarrier();
  }

  // Co-reduce gate and up across the 32 lanes of each slot in 5 shared barrier rounds.
  s_gate[tid] = acc_gate;
  s_up[tid] = acc_up;
  workgroupBarrier();
  if (lane < 16u) {
    s_gate[tid] = s_gate[tid] + s_gate[tid + 16u];
    s_up[tid] = s_up[tid] + s_up[tid + 16u];
  }
  workgroupBarrier();
  if (lane < 8u) {
    s_gate[tid] = s_gate[tid] + s_gate[tid + 8u];
    s_up[tid] = s_up[tid] + s_up[tid + 8u];
  }
  workgroupBarrier();
  if (lane < 4u) {
    s_gate[tid] = s_gate[tid] + s_gate[tid + 4u];
    s_up[tid] = s_up[tid] + s_up[tid + 4u];
  }
  workgroupBarrier();
  if (lane < 2u) {
    s_gate[tid] = s_gate[tid] + s_gate[tid + 2u];
    s_up[tid] = s_up[tid] + s_up[tid + 2u];
  }
  workgroupBarrier();
  if (lane == 0u) {
    let g_val = s_gate[tid] + s_gate[tid + 1u];
    let u_val = s_up[tid] + s_up[tid + 1u];
    if (row_valid) {
      let silu = g_val / (1.0 + exp(-g_val));
      // Fault mode drops the sigmoid gate (outputs linear g * u) without NaN/OOB.
      dst_data[row] = select(silu * u_val, g_val * u_val, params.fault_mode != 0u);
    }
  }
}
