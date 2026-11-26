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
// Performance architecture (portable WebGPU WGSL, 2 KB workgroup memory, no subgroup ext):
// - 128 threads = 4 slots of 32 lanes; each slot computes 2 output rows (8 rows/WG,
//   matching qmmDispatchGrid).
// - Zero workgroupBarrier() calls inside the K loop: L1-hot x_data is loaded via
//   vec4<f32> once per 16 elements and reused across 4 projections (gate0, up0, gate1, up1).
// - vec2<u32> coalesced weight loads (256 B per 32-lane SIMDgroup) + power-of-2 bitshift
//   for group index + co-reduction of all 4 accumulators in 5 barrier rounds.
// - Uses exactly 8 storage buffers + 1 uniform buffer (fits WebGPU's guaranteed
//   maxStorageBuffersPerShaderStage = 8 on all adapters).

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

@group(0) @binding(0) var<storage, read> x_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> gate_w: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> gate_scales: array<f32>;
@group(0) @binding(3) var<storage, read> gate_biases: array<f32>;
@group(0) @binding(4) var<storage, read> up_w: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> up_scales: array<f32>;
@group(0) @binding(6) var<storage, read> up_biases: array<f32>;
@group(0) @binding(7) var<storage, read_write> dst_data: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

const ROWS_PER_WG: u32 = 8u;
const ROWS_PER_SLOT: u32 = 2u;

// Packing: (gate0, up0, gate1, up1) -> 128 * 16 B = 2 KB workgroup memory.
var<workgroup> s_red: array<vec4<f32>, 128>;

fn dot_word_q4(word: u32, xv0: vec4<f32>, xv1: vec4<f32>) -> f32 {
  let q0 = vec4<f32>(
    f32(word & 0xfu),
    f32((word >> 4u) & 0xfu),
    f32((word >> 8u) & 0xfu),
    f32((word >> 12u) & 0xfu),
  );
  let q1 = vec4<f32>(
    f32((word >> 16u) & 0xfu),
    f32((word >> 20u) & 0xfu),
    f32((word >> 24u) & 0xfu),
    f32((word >> 28u) & 0xfu),
  );
  return dot(xv0, q0) + dot(xv1, q1);
}

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  let tid = lid.x;
  let slot = tid >> 5u;   // 0..3
  let lane = tid & 31u;   // 0..31

  let wg_linear = wg.y * num_wg.x + wg.x;
  let row0 = wg_linear * ROWS_PER_WG + slot * ROWS_PER_SLOT;
  let row1 = row0 + 1u;
  let row0_valid = row0 < params.n;
  let row1_valid = row1 < params.n;

  let words2_per_row = params.k >> 4u;
  let words2_per_group = params.group_size >> 4u;
  let is_pow2_group = (words2_per_group & (words2_per_group - 1u)) == 0u;
  let group_shift = countTrailingZeros(words2_per_group);
  let groups_per_row = params.k / params.group_size;

  let w_base0 = row0 * words2_per_row;
  let w_base1 = row1 * words2_per_row;
  let g_base0 = row0 * groups_per_row;
  let g_base1 = row1 * groups_per_row;

  var acc_g0 = 0.0;
  var acc_u0 = 0.0;
  var acc_g1 = 0.0;
  var acc_u1 = 0.0;

  if (row0_valid) {
    for (var w2_col = lane; w2_col < words2_per_row; w2_col = w2_col + 32u) {
      let v_base = w2_col << 2u;
      let xv0 = x_data[v_base + 0u];
      let xv1 = x_data[v_base + 1u];
      let xv2 = x_data[v_base + 2u];
      let xv3 = x_data[v_base + 3u];

      let sx_v = (xv0 + xv1) + (xv2 + xv3);
      let sum_x = (sx_v.x + sx_v.y) + (sx_v.z + sx_v.w);

      var g_col = 0u;
      if (is_pow2_group) {
        g_col = w2_col >> group_shift;
      } else {
        g_col = w2_col / words2_per_group;
      }

      let w_off0 = w_base0 + w2_col;
      let g_off0 = g_base0 + g_col;
      let gw0 = gate_w[w_off0];
      let uw0 = up_w[w_off0];
      let dot_g0 = dot_word_q4(gw0.x, xv0, xv1) + dot_word_q4(gw0.y, xv2, xv3);
      let dot_u0 = dot_word_q4(uw0.x, xv0, xv1) + dot_word_q4(uw0.y, xv2, xv3);
      acc_g0 = acc_g0 + dot_g0 * gate_scales[g_off0] + sum_x * gate_biases[g_off0];
      acc_u0 = acc_u0 + dot_u0 * up_scales[g_off0] + sum_x * up_biases[g_off0];

      if (row1_valid) {
        let w_off1 = w_base1 + w2_col;
        let g_off1 = g_base1 + g_col;
        let gw1 = gate_w[w_off1];
        let uw1 = up_w[w_off1];
        let dot_g1 = dot_word_q4(gw1.x, xv0, xv1) + dot_word_q4(gw1.y, xv2, xv3);
        let dot_u1 = dot_word_q4(uw1.x, xv0, xv1) + dot_word_q4(uw1.y, xv2, xv3);
        acc_g1 = acc_g1 + dot_g1 * gate_scales[g_off1] + sum_x * gate_biases[g_off1];
        acc_u1 = acc_u1 + dot_u1 * up_scales[g_off1] + sum_x * up_biases[g_off1];
      }
    }
  }

  // Co-reduce (gate0, up0, gate1, up1) across the 32 lanes of each slot in 5 rounds.
  s_red[tid] = vec4<f32>(acc_g0, acc_u0, acc_g1, acc_u1);
  workgroupBarrier();
  if (lane < 16u) { s_red[tid] = s_red[tid] + s_red[tid + 16u]; }
  workgroupBarrier();
  if (lane < 8u) { s_red[tid] = s_red[tid] + s_red[tid + 8u]; }
  workgroupBarrier();
  if (lane < 4u) { s_red[tid] = s_red[tid] + s_red[tid + 4u]; }
  workgroupBarrier();
  if (lane < 2u) { s_red[tid] = s_red[tid] + s_red[tid + 2u]; }
  workgroupBarrier();
  if (lane == 0u) {
    let totals = s_red[tid] + s_red[tid + 1u];
    let fault = params.fault_mode != 0u;
    if (row0_valid) {
      let g0 = totals.x;
      let u0 = totals.y;
      let silu0 = g0 / (1.0 + exp(-g0));
      // Fault mode drops the sigmoid gate (outputs linear g * u) without NaN/OOB.
      dst_data[row0] = select(silu0 * u0, g0 * u0, fault);
    }
    if (row1_valid) {
      let g1 = totals.z;
      let u1 = totals.w;
      let silu1 = g1 / (1.0 + exp(-g1));
      dst_data[row1] = select(silu1 * u1, g1 * u1, fault);
    }
  }
}
