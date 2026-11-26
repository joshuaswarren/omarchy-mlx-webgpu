// Ported from joshuaswarren/omarchy-mlx:
//   overlay/mlx/backend/omarchy/shaders/fast_norm.comp
// SPDX-License-Identifier: MIT
//
// Fused RMSNorm forward over the last axis:
//   n = inverseSqrt(mean(x^2) + eps)
//   y = x * n * w
// One workgroup (256 threads) handles one row with shared-memory tree reduction.

struct Params {
  rows: u32,
  row_length: u32,
  weight_broadcast: u32,
  fault_mode: u32,
  eps: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> weight_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> block_values: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  let thread = lid.x;
  let row_length = params.row_length;
  let eps = params.eps;
  let threads = 256u;

  for (var row = wg.x; row < params.rows; row += num_wg.x) {
    let start = row * row_length;

    var local_sq = 0.0;
    for (var idx = thread; idx < row_length; idx += threads) {
      let v = input_data[start + idx];
      local_sq += v * v;
    }
    block_values[thread] = local_sq;
    workgroupBarrier();

    var stride = 128u;
    loop {
      if (stride == 0u) {
        break;
      }
      if (thread < stride) {
        block_values[thread] += block_values[thread + stride];
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }

    let mean_square = block_values[0] / f32(row_length);
    // Fault mode (when enabled): drop the RMS denominator (use unnormalized scale 1.0)
    let norm = select(inverseSqrt(mean_square + eps), 1.0, params.fault_mode != 0u);
    workgroupBarrier();

    for (var idx = thread; idx < row_length; idx += threads) {
      let v = input_data[start + idx];
      let w = select(weight_data[idx], weight_data[0], params.weight_broadcast != 0u);
      output_data[start + idx] = v * norm * w;
    }
    workgroupBarrier();
  }
}
