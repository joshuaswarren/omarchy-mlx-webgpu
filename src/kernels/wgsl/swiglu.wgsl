// Ported from joshuaswarren/omarchy-mlx:
//   overlay/mlx/backend/omarchy/shaders/swiglu.comp
// SPDX-License-Identifier: MIT
//
// Elementwise SwiGLU activation:
//   dst[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
//
// Portable WebGPU WGSL: f32 arithmetic, 0 B workgroup memory, no subgroup ext.

struct Params {
  count: u32,
  fault_mode: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> gate_data: array<f32>;
@group(0) @binding(1) var<storage, read> up_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_data: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  let stride = num_wg.x * 256u;
  for (var idx = gid.x; idx < params.count; idx = idx + stride) {
    let g = gate_data[idx];
    let u = up_data[idx];
    let silu = g / (1.0 + exp(-g));
    dst_data[idx] = select(silu * u, g * u, params.fault_mode != 0u);
  }
}
