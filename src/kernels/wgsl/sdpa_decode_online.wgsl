// Ported from joshuaswarren/omarchy-mlx PR #997:
//   overlay/mlx/backend/omarchy/shaders/sdpa_decode_online.comp
// SPDX-License-Identifier: MIT
//
// Single-dispatch online-softmax decode SDPA for head_dim = 128:
// - Grid = (heads, 1, 1): one 256-thread workgroup per query head.
// - 8 cooperative 32-lane groups stride keys by 8; each lane holds 4 consecutive
//   head elements (q0..q3, o0..o3) in registers.
// - Uses exp2 with pre-scaled q * scale * LOG2E (exact PR #997 online softmax).
// - Zero scratch buffer, zero atomics, zero Pass-2 dispatch.
// - Portable WebGPU: uses workgroup shared-memory tree reduction instead of
//   vendor-specific subgroup intrinsics so it runs identically on Metal, Vulkan,
//   D3D12, and SwiftShader WebGPU adapters.

struct Params {
  heads: u32,
  kv_heads: u32,
  k_len: u32,
  max_seq: u32,
  scale: f32,
  fault_mode: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> query_data: array<f32>;
@group(0) @binding(1) var<storage, read> key_cache: array<f32>;
@group(0) @binding(2) var<storage, read> value_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const D: u32 = 128u;
const GROUPS: u32 = 8u;
const LANES: u32 = 32u;
const LOG2E: f32 = 1.4426950408889634;
const NEG_INF: f32 = -3.4028234663852886e+38;

var<workgroup> s_dot: array<f32, 256>;
var<workgroup> s_max: array<f32, 8>;
var<workgroup> s_sum: array<f32, 8>;
var<workgroup> s_o: array<f32, 1024>; // GROUPS * D = 8 * 128

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let head = wg.x;
  let heads = params.heads;
  let kv_heads = params.kv_heads;
  let k_len = params.k_len;
  let max_seq = params.max_seq;
  if (head >= heads || k_len == 0u) {
    return;
  }

  let repeats = heads / kv_heads;
  let kv_head = head / repeats;

  let t = lid.x;
  let sg = t >> 5u;    // 0..7
  let lane = t & 31u;  // 0..31

  let s_scale = params.scale * LOG2E;
  let q_base = head * D + lane * 4u;
  let q0 = query_data[q_base + 0u] * s_scale;
  let q1 = query_data[q_base + 1u] * s_scale;
  let q2 = query_data[q_base + 2u] * s_scale;
  let q3 = query_data[q_base + 3u] * s_scale;

  var m_i = NEG_INF;
  var l_i = 0.0;
  var o0 = 0.0;
  var o1 = 0.0;
  var o2 = 0.0;
  var o3 = 0.0;

  let kv_head_base = kv_head * max_seq * D;
  let max_iters = (k_len + GROUPS - 1u) / GROUPS;

  for (var iter = 0u; iter < max_iters; iter = iter + 1u) {
    let key = sg + iter * GROUPS;
    let is_active = key < k_len;

    var dot = 0.0;
    var v0 = 0.0;
    var v1 = 0.0;
    var v2 = 0.0;
    var v3 = 0.0;

    if (is_active) {
      let kv_base = kv_head_base + key * D + lane * 4u;
      dot = q0 * key_cache[kv_base + 0u]
          + q1 * key_cache[kv_base + 1u]
          + q2 * key_cache[kv_base + 2u]
          + q3 * key_cache[kv_base + 3u];
      v0 = value_cache[kv_base + 0u];
      v1 = value_cache[kv_base + 1u];
      v2 = value_cache[kv_base + 2u];
      v3 = value_cache[kv_base + 3u];
    }

    // 32-lane tree reduction within each group (portable subgroupAdd)
    s_dot[t] = dot;
    workgroupBarrier();
    if (lane < 16u) { s_dot[t] += s_dot[t + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { s_dot[t] += s_dot[t + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { s_dot[t] += s_dot[t + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { s_dot[t] += s_dot[t + 2u]; }
    workgroupBarrier();
    if (lane == 0u) { s_dot[t] += s_dot[t + 1u]; }
    workgroupBarrier();

    if (is_active) {
      let score = s_dot[sg * LANES];
      if (iter == 0u) {
        m_i = score;
        l_i = 1.0;
        o0 = v0;
        o1 = v1;
        o2 = v2;
        o3 = v3;
      } else {
        let new_max = max(m_i, score);
        // FAULT SWITCH: when fault_mode != 0, drop the online-softmax max-rescale
        // (uses inverted weight / unrescaled accumulator — simulates a broken
        // online-softmax merge while remaining finite and TDR-safe).
        let factor = select(exp2(m_i - new_max), -1.0, params.fault_mode != 0u);
        let p = select(exp2(score - new_max), 1.0, params.fault_mode != 0u);
        m_i = new_max;
        l_i = l_i * factor + p;
        o0 = o0 * factor + p * v0;
        o1 = o1 * factor + p * v1;
        o2 = o2 * factor + p * v2;
        o3 = o3 * factor + p * v3;
      }
    }
  }

  if (sg < k_len) {
    if (lane == 0u) {
      s_max[sg] = m_i;
      s_sum[sg] = l_i;
    }
    let obase = sg * D + lane * 4u;
    s_o[obase + 0u] = o0;
    s_o[obase + 1u] = o1;
    s_o[obase + 2u] = o2;
    s_o[obase + 3u] = o3;
  }

  workgroupBarrier();

  // Final merge across GROUPS partial states by threads 0..127 (1 thread per head dim)
  if (t < D) {
    let active_sg = min(GROUPS, k_len);
    var gmax = NEG_INF;
    for (var s = 0u; s < active_sg; s = s + 1u) {
      gmax = max(gmax, s_max[s]);
    }
    var denom = 0.0;
    var acc = 0.0;
    for (var s = 0u; s < active_sg; s = s + 1u) {
      // FAULT SWITCH also corrupts cross-group merge weights when k_len <= 8
      let w = select(
        exp2(s_max[s] - gmax),
        select(1.0, -1.0, (s & 1u) == 1u),
        params.fault_mode != 0u
      );
      denom += s_sum[s] * w;
      acc += s_o[s * D + t] * w;
    }
    let safe_denom = select(denom, 1.0, abs(denom) < 1e-12);
    output_data[head * D + t] = acc / safe_denom;
  }
}
