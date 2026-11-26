// Ported from joshuaswarren/omarchy-mlx:
//   overlay/mlx/backend/omarchy/shaders/qmm_vec.comp (QMM_VEC_Q4_WORD path)
// SPDX-License-Identifier: MIT
//
// Affine 4-bit (group_size = 64) quantized matrix-vector product:
//   dst[n] = sum_{k=0..K-1} x[k] * (nibble(w[n, k]) * scales[n, k/g] + biases[n, k/g])
// where each u32 word in w_data holds 8 little-endian 4-bit unsigned nibbles along K.
//
// Performance architecture (portable WebGPU WGSL, 1 KB workgroup memory, no subgroup ext):
// - 128 threads = 4 slots of 32 lanes; each slot computes 2 output rows (8 rows/WG,
//   matching qmmDispatchGrid -> 18992 WGs <= 65535 for Qwen3 vocab N = 151936).
// - Zero workgroupBarrier() calls inside the K loop: x_data (4-12 KB) stays L1-hot
//   and is loaded via 128-bit vec4<f32> reads reused across both rows of each slot.
// - Each lane reads vec2<u32> (2 words = 16 nibbles = 8 B -> 256 B coalesced per 32-lane
//   SIMDgroup), cutting weight loads, scale/bias loads, and affine FMAs in half.
// - Power-of-2 group_size (32/64/128) uses a 1-cycle bitshift instead of runtime division.
// - Portable 32-lane tree reduction of both rows in 1 KB workgroup memory (5 barriers total).

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
@group(0) @binding(1) var<storage, read> w_data: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> scales_data: array<f32>;
@group(0) @binding(3) var<storage, read> biases_data: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_data: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const ROWS_PER_WG: u32 = 8u;
const ROWS_PER_SLOT: u32 = 2u;

var<workgroup> s_red: array<vec2<f32>, 128>;

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

  let words2_per_row = params.k >> 4u; // 16 nibbles per vec2<u32>
  let words2_per_group = params.group_size >> 4u; // 4 for g64
  let is_pow2_group = (words2_per_group & (words2_per_group - 1u)) == 0u;
  let group_shift = countTrailingZeros(words2_per_group);
  let groups_per_row = params.k / params.group_size;

  let w_base0 = row0 * words2_per_row;
  let w_base1 = row1 * words2_per_row;
  let g_base0 = row0 * groups_per_row;
  let g_base1 = row1 * groups_per_row;
  let fault = params.fault_mode != 0u;

  var acc0 = 0.0;
  var acc1 = 0.0;

  if (row0_valid) {
    for (var w2_col = lane; w2_col < words2_per_row; w2_col = w2_col + 32u) {
      let v_base = w2_col << 2u; // 4 vec4<f32> = 16 floats
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

      let w0 = w_data[w_base0 + w2_col];
      let s0 = scales_data[g_base0 + g_col];
      // Fault mode drops the affine bias term while remaining finite and TDR-safe.
      let b0 = select(biases_data[g_base0 + g_col], 0.0, fault);
      let dot0 = dot_word_q4(w0.x, xv0, xv1) + dot_word_q4(w0.y, xv2, xv3);
      acc0 = acc0 + dot0 * s0 + sum_x * b0;

      if (row1_valid) {
        let w1 = w_data[w_base1 + w2_col];
        let s1 = scales_data[g_base1 + g_col];
        let b1 = select(biases_data[g_base1 + g_col], 0.0, fault);
        let dot1 = dot_word_q4(w1.x, xv0, xv1) + dot_word_q4(w1.y, xv2, xv3);
        acc1 = acc1 + dot1 * s1 + sum_x * b1;
      }
    }
  }

  // Portable 32-lane tree reduction within each slot (co-reduces row0 and row1).
  s_red[tid] = vec2<f32>(acc0, acc1);
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
    let total = s_red[tid] + s_red[tid + 1u];
    if (row0_valid) {
      dst_data[row0] = total.x;
    }
    if (row1_valid) {
      dst_data[row1] = total.y;
    }
  }
}
