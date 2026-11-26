import type { QmmPrecision, QmmWorkgroupSize } from "./types";
import qmmVecBaseWgsl from "./wgsl/qmm_vec.wgsl?raw";
import qmmVecSwigluBaseWgsl from "./wgsl/qmm_vec_swiglu.wgsl?raw";
import sdpaOnlineBaseWgsl from "./wgsl/sdpa_decode_online.wgsl?raw";

const MAX_WG_DIM = 65535;

/** Rows computed per workgroup for a given QMM workgroup size (2 rows per 32-lane slot). */
export function qmmRowsPerWg(wgSize: QmmWorkgroupSize): number {
  return (wgSize / 32) * 2; // 64 -> 4, 128 -> 8, 256 -> 16
}

/** 2D-safe dispatch grid for any QMM workgroup size (supports Qwen3 vocab N=151936). */
export function qmmDispatchGridForWg(
  n: number,
  wgSize: QmmWorkgroupSize = 128,
): [number, number, number] {
  const rowsPerWg = qmmRowsPerWg(wgSize);
  const totalWg = Math.max(1, Math.ceil(n / rowsPerWg));
  if (totalWg <= MAX_WG_DIM) {
    return [totalWg, 1, 1];
  }
  const gridX = MAX_WG_DIM;
  const gridY = Math.ceil(totalWg / gridX);
  return [gridX, gridY, 1];
}

export interface QmmVariantOpts {
  wgSize: QmmWorkgroupSize;
  precision: QmmPrecision;
  useSubgroups: boolean;
}

/**
 * Generates parameterized WGSL for `qmm_vec` supporting:
 * - workgroup sizes 64 (4 rows/WG), 128 (8 rows/WG), 256 (16 rows/WG)
 * - f32, f16 (vec4<f16> inner dot + f32 accum), and f16_accum (full f16 K-loop accum)
 * - optional hardware `subgroupAdd` reduction (0 workgroup barriers on 32-wide SIMDgroups)
 */
export function buildQmmVecWgsl(opts: QmmVariantOpts): string {
  const { wgSize, precision, useSubgroups } = opts;
  if (wgSize === 128 && precision === "f32" && !useSubgroups) {
    return qmmVecBaseWgsl;
  }

  const rowsPerWg = qmmRowsPerWg(wgSize);
  const enableLines: string[] = [];
  if (precision !== "f32") enableLines.push("enable f16;");
  if (useSubgroups) enableLines.push("enable subgroups;");
  const header = enableLines.length > 0 ? `${enableLines.join("\n")}\n\n` : "";

  const dotFn =
    precision === "f32"
      ? `fn dot_word_q4(word: u32, xv0: vec4<f32>, xv1: vec4<f32>) -> f32 {
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
}`
      : `fn dot_word_q4_f16(word: u32, xv0: vec4<f16>, xv1: vec4<f16>) -> f16 {
  let q0 = vec4<f16>(
    f16(word & 0xfu),
    f16((word >> 4u) & 0xfu),
    f16((word >> 8u) & 0xfu),
    f16((word >> 12u) & 0xfu),
  );
  let q1 = vec4<f16>(
    f16((word >> 16u) & 0xfu),
    f16((word >> 20u) & 0xfu),
    f16((word >> 24u) & 0xfu),
    f16((word >> 28u) & 0xfu),
  );
  return dot(xv0, q0) + dot(xv1, q1);
}`;

  const kLoopBody =
    precision === "f32"
      ? `      let dot0 = dot_word_q4(w0.x, xv0, xv1) + dot_word_q4(w0.y, xv2, xv3);
      acc0 = acc0 + dot0 * s0 + sum_x * b0;

      if (row1_valid) {
        let w1 = w_data[w_base1 + w2_col];
        let s1 = scales_data[g_base1 + g_col];
        let b1 = select(biases_data[g_base1 + g_col], 0.0, fault);
        let dot1 = dot_word_q4(w1.x, xv0, xv1) + dot_word_q4(w1.y, xv2, xv3);
        acc1 = acc1 + dot1 * s1 + sum_x * b1;
      }`
      : precision === "f16"
        ? `      let h0 = vec4<f16>(xv0);
      let h1 = vec4<f16>(xv1);
      let h2 = vec4<f16>(xv2);
      let h3 = vec4<f16>(xv3);
      let dot0 = f32(dot_word_q4_f16(w0.x, h0, h1) + dot_word_q4_f16(w0.y, h2, h3));
      acc0 = acc0 + dot0 * s0 + sum_x * b0;

      if (row1_valid) {
        let w1 = w_data[w_base1 + w2_col];
        let s1 = scales_data[g_base1 + g_col];
        let b1 = select(biases_data[g_base1 + g_col], 0.0, fault);
        let dot1 = f32(dot_word_q4_f16(w1.x, h0, h1) + dot_word_q4_f16(w1.y, h2, h3));
        acc1 = acc1 + dot1 * s1 + sum_x * b1;
      }`
        : `      let h0 = vec4<f16>(xv0);
      let h1 = vec4<f16>(xv1);
      let h2 = vec4<f16>(xv2);
      let h3 = vec4<f16>(xv3);
      let sum_h = f16(sum_x);
      let dot0 = dot_word_q4_f16(w0.x, h0, h1) + dot_word_q4_f16(w0.y, h2, h3);
      acc0_h = acc0_h + dot0 * f16(s0) + sum_h * f16(b0);

      if (row1_valid) {
        let w1 = w_data[w_base1 + w2_col];
        let s1 = scales_data[g_base1 + g_col];
        let b1 = select(biases_data[g_base1 + g_col], 0.0, fault);
        let dot1 = dot_word_q4_f16(w1.x, h0, h1) + dot_word_q4_f16(w1.y, h2, h3);
        acc1_h = acc1_h + dot1 * f16(s1) + sum_h * f16(b1);
      }`;

  const accDecl =
    precision === "f16_accum"
      ? `  var acc0_h: f16 = 0.0h;
  var acc1_h: f16 = 0.0h;`
      : `  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;`;

  const accFinalize =
    precision === "f16_accum"
      ? `  let acc0 = f32(acc0_h);
  let acc1 = f32(acc1_h);`
      : "";

  const sgBuiltin = useSubgroups ? "\n  @builtin(subgroup_size) sg_size: u32," : "";

  const reductionBlock = useSubgroups
    ? `  if (sg_size == 32u) {
    let total = subgroupAdd(vec2<f32>(acc0, acc1));
    if (lane == 0u) {
      if (row0_valid) { dst_data[row0] = total.x; }
      if (row1_valid) { dst_data[row1] = total.y; }
    }
    return;
  }
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
    if (row0_valid) { dst_data[row0] = total.x; }
    if (row1_valid) { dst_data[row1] = total.y; }
  }`
    : `  s_red[tid] = vec2<f32>(acc0, acc1);
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
    if (row0_valid) { dst_data[row0] = total.x; }
    if (row1_valid) { dst_data[row1] = total.y; }
  }`;

  return `${header}struct Params {
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

const ROWS_PER_WG: u32 = ${rowsPerWg}u;
const ROWS_PER_SLOT: u32 = 2u;

var<workgroup> s_red: array<vec2<f32>, ${wgSize}>;

${dotFn}

@compute @workgroup_size(${wgSize}, 1, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,${sgBuiltin}
) {
  let tid = lid.x;
  let slot = tid >> 5u;
  let lane = tid & 31u;

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
  let fault = params.fault_mode != 0u;

${accDecl}

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

      let w0 = w_data[w_base0 + w2_col];
      let s0 = scales_data[g_base0 + g_col];
      let b0 = select(biases_data[g_base0 + g_col], 0.0, fault);
${kLoopBody}
    }
  }
${accFinalize}
${reductionBlock}
}
`;
}

/**
 * Generates parameterized WGSL for fused `qmm_vec_swiglu` matching `buildQmmVecWgsl`.
 */
export function buildQmmVecSwigluWgsl(opts: QmmVariantOpts): string {
  const { wgSize, precision, useSubgroups } = opts;
  if (wgSize === 128 && precision === "f32" && !useSubgroups) {
    return qmmVecSwigluBaseWgsl;
  }

  const rowsPerWg = qmmRowsPerWg(wgSize);
  const enableLines: string[] = [];
  if (precision !== "f32") enableLines.push("enable f16;");
  if (useSubgroups) enableLines.push("enable subgroups;");
  const header = enableLines.length > 0 ? `${enableLines.join("\n")}\n\n` : "";

  const dotFn =
    precision === "f32"
      ? `fn dot_word_q4(word: u32, xv0: vec4<f32>, xv1: vec4<f32>) -> f32 {
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
}`
      : `fn dot_word_q4_f16(word: u32, xv0: vec4<f16>, xv1: vec4<f16>) -> f16 {
  let q0 = vec4<f16>(
    f16(word & 0xfu),
    f16((word >> 4u) & 0xfu),
    f16((word >> 8u) & 0xfu),
    f16((word >> 12u) & 0xfu),
  );
  let q1 = vec4<f16>(
    f16((word >> 16u) & 0xfu),
    f16((word >> 20u) & 0xfu),
    f16((word >> 24u) & 0xfu),
    f16((word >> 28u) & 0xfu),
  );
  return dot(xv0, q0) + dot(xv1, q1);
}`;

  const accDecl =
    precision === "f16_accum"
      ? `  var acc_g0_h: f16 = 0.0h;
  var acc_u0_h: f16 = 0.0h;
  var acc_g1_h: f16 = 0.0h;
  var acc_u1_h: f16 = 0.0h;`
      : `  var acc_g0: f32 = 0.0;
  var acc_u0: f32 = 0.0;
  var acc_g1: f32 = 0.0;
  var acc_u1: f32 = 0.0;`;

  const kLoopBody =
    precision === "f32"
      ? `      let dot_g0 = dot_word_q4(gw0.x, xv0, xv1) + dot_word_q4(gw0.y, xv2, xv3);
      let dot_u0 = dot_word_q4(uw0.x, xv0, xv1) + dot_word_q4(uw0.y, xv2, xv3);
      let gb0 = select(gate_biases[g_off0], 0.0, fault);
      let ub0 = select(up_biases[g_off0], 0.0, fault);
      acc_g0 = acc_g0 + dot_g0 * gate_scales[g_off0] + sum_x * gb0;
      acc_u0 = acc_u0 + dot_u0 * up_scales[g_off0] + sum_x * ub0;

      if (row1_valid) {
        let w_off1 = w_base1 + w2_col;
        let g_off1 = g_base1 + g_col;
        let gw1 = gate_w[w_off1];
        let uw1 = up_w[w_off1];
        let dot_g1 = dot_word_q4(gw1.x, xv0, xv1) + dot_word_q4(gw1.y, xv2, xv3);
        let dot_u1 = dot_word_q4(uw1.x, xv0, xv1) + dot_word_q4(uw1.y, xv2, xv3);
        let gb1 = select(gate_biases[g_off1], 0.0, fault);
        let ub1 = select(up_biases[g_off1], 0.0, fault);
        acc_g1 = acc_g1 + dot_g1 * gate_scales[g_off1] + sum_x * gb1;
        acc_u1 = acc_u1 + dot_u1 * up_scales[g_off1] + sum_x * ub1;
      }`
      : precision === "f16"
        ? `      let h0 = vec4<f16>(xv0);
      let h1 = vec4<f16>(xv1);
      let h2 = vec4<f16>(xv2);
      let h3 = vec4<f16>(xv3);
      let dot_g0 = f32(dot_word_q4_f16(gw0.x, h0, h1) + dot_word_q4_f16(gw0.y, h2, h3));
      let dot_u0 = f32(dot_word_q4_f16(uw0.x, h0, h1) + dot_word_q4_f16(uw0.y, h2, h3));
      let gb0 = select(gate_biases[g_off0], 0.0, fault);
      let ub0 = select(up_biases[g_off0], 0.0, fault);
      acc_g0 = acc_g0 + dot_g0 * gate_scales[g_off0] + sum_x * gb0;
      acc_u0 = acc_u0 + dot_u0 * up_scales[g_off0] + sum_x * ub0;

      if (row1_valid) {
        let w_off1 = w_base1 + w2_col;
        let g_off1 = g_base1 + g_col;
        let gw1 = gate_w[w_off1];
        let uw1 = up_w[w_off1];
        let dot_g1 = f32(dot_word_q4_f16(gw1.x, h0, h1) + dot_word_q4_f16(gw1.y, h2, h3));
        let dot_u1 = f32(dot_word_q4_f16(uw1.x, h0, h1) + dot_word_q4_f16(uw1.y, h2, h3));
        let gb1 = select(gate_biases[g_off1], 0.0, fault);
        let ub1 = select(up_biases[g_off1], 0.0, fault);
        acc_g1 = acc_g1 + dot_g1 * gate_scales[g_off1] + sum_x * gb1;
        acc_u1 = acc_u1 + dot_u1 * up_scales[g_off1] + sum_x * ub1;
      }`
        : `      let h0 = vec4<f16>(xv0);
      let h1 = vec4<f16>(xv1);
      let h2 = vec4<f16>(xv2);
      let h3 = vec4<f16>(xv3);
      let sum_h = f16(sum_x);
      let dot_g0 = dot_word_q4_f16(gw0.x, h0, h1) + dot_word_q4_f16(gw0.y, h2, h3);
      let dot_u0 = dot_word_q4_f16(uw0.x, h0, h1) + dot_word_q4_f16(uw0.y, h2, h3);
      let gb0 = select(f16(gate_biases[g_off0]), 0.0h, fault);
      let ub0 = select(f16(up_biases[g_off0]), 0.0h, fault);
      acc_g0_h = acc_g0_h + dot_g0 * f16(gate_scales[g_off0]) + sum_h * gb0;
      acc_u0_h = acc_u0_h + dot_u0 * f16(up_scales[g_off0]) + sum_h * ub0;

      if (row1_valid) {
        let w_off1 = w_base1 + w2_col;
        let g_off1 = g_base1 + g_col;
        let gw1 = gate_w[w_off1];
        let uw1 = up_w[w_off1];
        let dot_g1 = dot_word_q4_f16(gw1.x, h0, h1) + dot_word_q4_f16(gw1.y, h2, h3);
        let dot_u1 = dot_word_q4_f16(uw1.x, h0, h1) + dot_word_q4_f16(uw1.y, h2, h3);
        let gb1 = select(f16(gate_biases[g_off1]), 0.0h, fault);
        let ub1 = select(f16(up_biases[g_off1]), 0.0h, fault);
        acc_g1_h = acc_g1_h + dot_g1 * f16(gate_scales[g_off1]) + sum_h * gb1;
        acc_u1_h = acc_u1_h + dot_u1 * f16(up_scales[g_off1]) + sum_h * ub1;
      }`;

  const accFinalize =
    precision === "f16_accum"
      ? `  let acc_g0 = f32(acc_g0_h);
  let acc_u0 = f32(acc_u0_h);
  let acc_g1 = f32(acc_g1_h);
  let acc_u1 = f32(acc_u1_h);`
      : "";

  const sgBuiltin = useSubgroups ? "\n  @builtin(subgroup_size) sg_size: u32," : "";
  const storeCode = `    if (row0_valid) {
      let g = total.x;
      let u = total.y;
      dst_data[row0] = (g / (1.0 + exp(-g))) * u;
    }
    if (row1_valid) {
      let g = total.z;
      let u = total.w;
      dst_data[row1] = (g / (1.0 + exp(-g))) * u;
    }`;

  const reductionBlock = useSubgroups
    ? `  if (sg_size == 32u) {
    let total = subgroupAdd(vec4<f32>(acc_g0, acc_u0, acc_g1, acc_u1));
    if (lane == 0u) {
${storeCode}
    }
    return;
  }
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
    let total = s_red[tid] + s_red[tid + 1u];
${storeCode}
  }`
    : `  s_red[tid] = vec4<f32>(acc_g0, acc_u0, acc_g1, acc_u1);
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
${storeCode}
  }`;

  return `${header}struct Params {
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

const ROWS_PER_WG: u32 = ${rowsPerWg}u;
const ROWS_PER_SLOT: u32 = 2u;

var<workgroup> s_red: array<vec4<f32>, ${wgSize}>;

${dotFn}

@compute @workgroup_size(${wgSize}, 1, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,${sgBuiltin}
) {
  let tid = lid.x;
  let slot = tid >> 5u;
  let lane = tid & 31u;

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
  let fault = params.fault_mode != 0u;

${accDecl}

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
${kLoopBody}
    }
  }
${accFinalize}
${reductionBlock}
}
`;
}

/**
 * Generates PR #997 `sdpa_decode_online` WGSL with optional hardware `subgroupAdd`
 * (eliminating all 7 workgroup barriers inside the K-tile loop on 32-wide SIMDgroups).
 */
export function buildSdpaOnlineWgsl(useSubgroups: boolean): string {
  if (!useSubgroups) return sdpaOnlineBaseWgsl;

  return `enable subgroups;

${sdpaOnlineBaseWgsl
    .replace(
      `@builtin(local_invocation_id) lid: vec3<u32>,\n)`,
      `@builtin(local_invocation_id) lid: vec3<u32>,\n  @builtin(subgroup_size) sg_size: u32,\n)`,
    )
    .replace(
      `    // 32-lane tree reduction within each group (portable subgroupAdd)
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
      let score = s_dot[sg * LANES];`,
      `    var score = 0.0;
    if (sg_size == 32u) {
      score = subgroupAdd(dot);
    } else {
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
      score = s_dot[sg * LANES];
    }

    if (is_active) {`,
    )
    .replace(
      `    // Synchronize before next iteration writes s_dot[t], preventing a WAR race
    // on non-lockstep adapters (e.g. SwiftShader, 16-wide Intel, 64-wide AMD).
    workgroupBarrier();`,
      `    if (sg_size != 32u) {
      workgroupBarrier();
    }`,
    )}`;
}
