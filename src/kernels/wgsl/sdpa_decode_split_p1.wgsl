// Ported from joshuaswarren/omarchy-mlx:
//   overlay/mlx/backend/omarchy/shaders/sdpa_decode_split_p1.comp
// SPDX-License-Identifier: MIT
//
// Split-K Pass 1 (2-dispatch baseline before PR #997):
// One workgroup per (query head, block of 64 keys) computes block (max, exp_sum,
// weighted value sum[128]) and writes (D + 4) = 132 floats to the scratch buffer.
// Pass 2 (sdpa_decode_split_p2.wgsl) reduces across blocks to produce the output.

struct Params {
  heads: u32,
  kv_heads: u32,
  k_len: u32,
  max_seq: u32,
  scale: f32,
  fault_mode: u32,
  nblocks: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> query_data: array<f32>;
@group(0) @binding(1) var<storage, read> key_cache: array<f32>;
@group(0) @binding(2) var<storage, read> value_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch_data: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const D: u32 = 128u;
const BK: u32 = 64u;
const STRIDE: u32 = 132u; // D + 4 floats per (head, block)
const NEG_INF: f32 = -3.4028234663852886e+38;

var<workgroup> s_q: array<f32, 128>;
var<workgroup> s_scores: array<f32, 64>;
var<workgroup> s_red: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let head = wg.x;
  let block = wg.y;
  let t = lid.x;

  if (head >= params.heads || block >= params.nblocks) {
    return;
  }

  let repeats = params.heads / params.kv_heads;
  let kv_head = head / repeats;
  let kv_head_base = kv_head * params.max_seq * D;

  let key_start = block * BK;
  let key_end = min(key_start + BK, params.k_len);
  let block_len = key_end - key_start;

  // Load query row into shared memory
  s_q[t] = query_data[head * D + t] * params.scale;
  workgroupBarrier();

  // Score phase: first 64 threads each compute one key's dot product
  if (t < BK) {
    if (t < block_len) {
      let k_base = kv_head_base + (key_start + t) * D;
      var dot = 0.0;
      for (var d = 0u; d < D; d = d + 1u) {
        dot += s_q[d] * key_cache[k_base + d];
      }
      s_scores[t] = dot;
    } else {
      s_scores[t] = NEG_INF;
    }
  }
  workgroupBarrier();

  // Block max over 64 keys
  s_red[t] = select(NEG_INF, s_scores[t], t < BK);
  workgroupBarrier();
  if (t < 32u) { s_red[t] = max(s_red[t], s_red[t + 32u]); }
  workgroupBarrier();
  if (t < 16u) { s_red[t] = max(s_red[t], s_red[t + 16u]); }
  workgroupBarrier();
  if (t < 8u) { s_red[t] = max(s_red[t], s_red[t + 8u]); }
  workgroupBarrier();
  if (t < 4u) { s_red[t] = max(s_red[t], s_red[t + 4u]); }
  workgroupBarrier();
  if (t < 2u) { s_red[t] = max(s_red[t], s_red[t + 2u]); }
  workgroupBarrier();
  if (t == 0u) { s_red[0] = max(s_red[0], s_red[1]); }
  workgroupBarrier();
  let block_max = s_red[0];

  // Softmax numerators exp(score - block_max)
  if (t < BK) {
    if (t < block_len) {
      let w = exp(s_scores[t] - block_max);
      // Fault switch: flip alternating attention weights when fault_mode != 0
      s_scores[t] = select(w, select(1.0, -1.0, (t & 1u) == 1u), params.fault_mode != 0u);
    } else {
      s_scores[t] = 0.0;
    }
  }
  workgroupBarrier();

  // Block exp sum
  s_red[t] = select(0.0, s_scores[t], t < BK);
  workgroupBarrier();
  if (t < 32u) { s_red[t] += s_red[t + 32u]; }
  workgroupBarrier();
  if (t < 16u) { s_red[t] += s_red[t + 16u]; }
  workgroupBarrier();
  if (t < 8u) { s_red[t] += s_red[t + 8u]; }
  workgroupBarrier();
  if (t < 4u) { s_red[t] += s_red[t + 4u]; }
  workgroupBarrier();
  if (t < 2u) { s_red[t] += s_red[t + 2u]; }
  workgroupBarrier();
  if (t == 0u) { s_red[0] += s_red[1]; }
  workgroupBarrier();
  let block_sum = s_red[0];

  // Weighted value accumulation: each of 128 threads owns 1 output dimension
  var acc = 0.0;
  for (var j = 0u; j < block_len; j = j + 1u) {
    let v_idx = kv_head_base + (key_start + j) * D + t;
    acc += s_scores[j] * value_cache[v_idx];
  }

  let out_base = (head * params.nblocks + block) * STRIDE;
  scratch_data[out_base + t] = acc;
  if (t == 0u) {
    scratch_data[out_base + D + 0u] = block_max;
    scratch_data[out_base + D + 1u] = block_sum;
    scratch_data[out_base + D + 2u] = 0.0;
    scratch_data[out_base + D + 3u] = 0.0;
  }
}
