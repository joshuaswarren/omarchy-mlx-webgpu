// Ported from joshuaswarren/omarchy-mlx:
//   overlay/mlx/backend/omarchy/shaders/qmm_vec.comp (QMM_VEC_Q4_WORD path)
// SPDX-License-Identifier: MIT
//
// Affine 4-bit (group_size = 64) quantized matrix-vector product:
//   dst[n] = sum_{k=0..K-1} x[k] * (nibble(w[n, k]) * scales[n, k/g] + biases[n, k/g])
// where each u32 word in w_data holds 8 little-endian 4-bit unsigned nibbles along K.
//
// Architecture:
// - 256 threads = 8 slots of 32 lanes; each slot computes 1 output row (8 rows/WG).
// - For N = 151936 (Qwen3 vocab), ceil(151936 / 8) = 18992 workgroups (<= 65535);
//   also supports 2D dispatch (wg.y * num_wg.x + wg.x) for arbitrarily large N.
// - Coalesced tile load of 256 f32 x-elements into workgroup SRAM shared by all 8 rows.
// - Consecutive lanes read consecutive u32 weight words (128 B coalesced transactions),
//   unpack 8 nibbles in registers, and apply affine scale/bias once per word.
// - Portable 32-lane tree reduction in workgroup memory (2 KB total, no subgroup ext).

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
@group(0) @binding(1) var<storage, read> w_data: array<u32>;
@group(0) @binding(2) var<storage, read> scales_data: array<f32>;
@group(0) @binding(3) var<storage, read> biases_data: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_data: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const SLOTS: u32 = 8u;
const LANES: u32 = 32u;
const TILE_K: u32 = 256u; // 32 lanes * 8 nibbles per u32 word

var<workgroup> s_x: array<f32, 256>;
var<workgroup> s_red: array<f32, 256>;

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

  var acc = 0.0;

  // All 256 threads execute identical tile loop trip counts so barriers stay uniform.
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
      let word = w_data[w_row_base + w_col];
      let g_col = (w_col << 3u) / params.group_size;
      let scale = scales_data[g_row_base + g_col];
      // Fault mode drops the affine bias term while remaining finite and TDR-safe.
      let bias = select(biases_data[g_row_base + g_col], 0.0, params.fault_mode != 0u);

      let lx = lane << 3u;
      let x0 = s_x[lx + 0u];
      let x1 = s_x[lx + 1u];
      let x2 = s_x[lx + 2u];
      let x3 = s_x[lx + 3u];
      let x4 = s_x[lx + 4u];
      let x5 = s_x[lx + 5u];
      let x6 = s_x[lx + 6u];
      let x7 = s_x[lx + 7u];

      let dot_q =
          x0 * f32(word & 0xfu)
        + x1 * f32((word >> 4u) & 0xfu)
        + x2 * f32((word >> 8u) & 0xfu)
        + x3 * f32((word >> 12u) & 0xfu)
        + x4 * f32((word >> 16u) & 0xfu)
        + x5 * f32((word >> 20u) & 0xfu)
        + x6 * f32((word >> 24u) & 0xfu)
        + x7 * f32((word >> 28u) & 0xfu);
      let sum_x = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));
      acc = acc + dot_q * scale + sum_x * bias;
    }
    workgroupBarrier();
  }

  // Portable 32-lane tree reduction within each slot.
  s_red[tid] = acc;
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
    if (row_valid) {
      dst_data[row] = total;
    }
  }
}
