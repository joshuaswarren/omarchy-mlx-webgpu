// Ported from joshuaswarren/omarchy-mlx:
//   overlay/mlx/backend/omarchy/shaders/sdpa_decode_split_p2.comp
// SPDX-License-Identifier: MIT
//
// Split-K Pass 2 (2-dispatch baseline before PR #997):
// One workgroup per query head merges per-block (max, exp_sum, weighted_val[128])
// written to scratch_data by Pass 1, divides by global exp sum, and writes output.

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

@group(0) @binding(0) var<storage, read> scratch_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const D: u32 = 128u;
const STRIDE: u32 = 132u;
const NEG_INF: f32 = -3.4028234663852886e+38;

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let head = wg.x;
  let t = lid.x;
  if (head >= params.heads) {
    return;
  }

  let nblocks = params.nblocks;
  let head_base = head * nblocks * STRIDE;

  var global_max = NEG_INF;
  for (var b = 0u; b < nblocks; b = b + 1u) {
    let m = scratch_data[head_base + b * STRIDE + D];
    global_max = max(global_max, m);
  }

  var denom = 0.0;
  var acc = 0.0;
  for (var b = 0u; b < nblocks; b = b + 1u) {
    let b_base = head_base + b * STRIDE;
    let b_max = scratch_data[b_base + D + 0u];
    let b_sum = scratch_data[b_base + D + 1u];
    let w = exp(b_max - global_max);
    denom += b_sum * w;
    acc += scratch_data[b_base + t] * w;
  }

  let safe_denom = select(denom, 1.0, abs(denom) < 1e-12);
  output_data[head * D + t] = acc / safe_denom;
}
