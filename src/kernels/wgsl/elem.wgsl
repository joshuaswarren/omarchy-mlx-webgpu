// Ported from joshuaswarren/omarchy-mlx utility/elementwise/argreduce kernels:
//   embed_q4, residual_add, kv_cache_write, and 2-stage argmax (argmax_p1 + argmax_p2)
// SPDX-License-Identifier: MIT
//
// Multi-entry-point portable WebGPU WGSL module:
// - Every module-scope binding uses a unique @binding index so `layout: "auto"`
//   builds a minimal per-entry-point bind group layout (<= 4 storage buffers each).
// - f32 arithmetic, 2 KB workgroup memory (argmax only), no subgroup extensions,
//   no WGSL reserved keywords.

// -----------------------------------------------------------------------------
// 1. embed_q4: MLX affine 4-bit (group_size = 64) token embedding lookup
// -----------------------------------------------------------------------------
struct EmbedParams {
  token_id: u32,
  hidden_size: u32,
  group_size: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> emb_weights: array<u32>;
@group(0) @binding(1) var<storage, read> emb_scales: array<f32>;
@group(0) @binding(2) var<storage, read> emb_biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> emb_dst: array<f32>;
@group(0) @binding(4) var<uniform> emb_params: EmbedParams;

@compute @workgroup_size(256, 1, 1)
fn embed_q4(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  let hidden_size = emb_params.hidden_size;
  let words_per_row = hidden_size >> 3u;
  let groups_per_row = hidden_size / emb_params.group_size;
  let w_row = emb_params.token_id * words_per_row;
  let g_row = emb_params.token_id * groups_per_row;
  let stride = num_wg.x * 256u;

  for (var d = gid.x; d < hidden_size; d = d + stride) {
    let word = emb_weights[w_row + (d >> 3u)];
    let shift = (d & 7u) << 2u;
    let q = (word >> shift) & 0xfu;
    let g_idx = d / emb_params.group_size;
    let scale = emb_scales[g_row + g_idx];
    let bias = emb_biases[g_row + g_idx];
    emb_dst[d] = f32(q) * scale + bias;
  }
}

// -----------------------------------------------------------------------------
// 2. residual_add: elementwise dst[i] = a[i] + b[i]
// -----------------------------------------------------------------------------
struct AddParams {
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(5) var<storage, read> add_a: array<f32>;
@group(0) @binding(6) var<storage, read> add_b: array<f32>;
@group(0) @binding(7) var<storage, read_write> add_dst: array<f32>;
@group(0) @binding(8) var<uniform> add_params: AddParams;

@compute @workgroup_size(256, 1, 1)
fn residual_add(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  let stride = num_wg.x * 256u;
  for (var idx = gid.x; idx < add_params.count; idx = idx + stride) {
    add_dst[idx] = add_a[idx] + add_b[idx];
  }
}

// -----------------------------------------------------------------------------
// 3. kv_cache_write: write single-token K and V into [kv_heads, max_seq, head_dim]
// -----------------------------------------------------------------------------
struct KvWriteParams {
  kv_heads: u32,
  head_dim: u32,
  max_seq: u32,
  pos: u32,
};

@group(0) @binding(9) var<storage, read> kv_k_in: array<f32>;
@group(0) @binding(10) var<storage, read> kv_v_in: array<f32>;
@group(0) @binding(11) var<storage, read_write> kv_k_cache: array<f32>;
@group(0) @binding(12) var<storage, read_write> kv_v_cache: array<f32>;
@group(0) @binding(13) var<uniform> kv_params: KvWriteParams;

@compute @workgroup_size(256, 1, 1)
fn kv_cache_write(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  let head_dim = kv_params.head_dim;
  let total = kv_params.kv_heads * head_dim;
  let stride = num_wg.x * 256u;

  for (var idx = gid.x; idx < total; idx = idx + stride) {
    let h = idx / head_dim;
    let d = idx - h * head_dim;
    let dst_idx = (h * kv_params.max_seq + kv_params.pos) * head_dim + d;
    kv_k_cache[dst_idx] = kv_k_in[idx];
    kv_v_cache[dst_idx] = kv_v_in[idx];
  }
}

// -----------------------------------------------------------------------------
// 4. 2-stage deterministic argmax (handles vocab N = 151936 with zero atomics)
//    Stage 1 (argmax_p1): dispatch (num_blocks, 1, 1), e.g. num_blocks = 256.
//    Stage 2 (argmax_p2): dispatch (1, 1, 1) to reduce num_blocks -> winning id.
//    Tie-breaking: strictly prefers lower index on equal maximum value.
// -----------------------------------------------------------------------------
struct ArgmaxParams {
  count: u32,
  num_blocks: u32,
  _pad0: u32,
  _pad1: u32,
};

const AMAX_NEG_INF: f32 = -3.4028234e+38;
const AMAX_MAX_U32: u32 = 0xffffffffu;

@group(0) @binding(14) var<storage, read> amax_logits: array<f32>;
@group(0) @binding(15) var<storage, read_write> amax_partial_vals: array<f32>;
@group(0) @binding(16) var<storage, read_write> amax_partial_idxs: array<u32>;
@group(0) @binding(17) var<uniform> amax_params: ArgmaxParams;

@group(0) @binding(18) var<storage, read> amax_p2_vals: array<f32>;
@group(0) @binding(19) var<storage, read> amax_p2_idxs: array<u32>;
@group(0) @binding(20) var<storage, read_write> amax_out_idx: array<u32>;
@group(0) @binding(21) var<storage, read_write> amax_out_val: array<f32>;

var<workgroup> s_amax_val: array<f32, 256>;
var<workgroup> s_amax_idx: array<u32, 256>;

fn is_better(cand_val: f32, cand_idx: u32, best_val: f32, best_idx: u32) -> bool {
  return (cand_val > best_val) || (cand_val == best_val && cand_idx < best_idx);
}

@compute @workgroup_size(256, 1, 1)
fn argmax_p1(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  let tid = lid.x;
  let block_id = wg.x;
  let total_threads = num_wg.x * 256u;

  var best_val = AMAX_NEG_INF;
  var best_idx = AMAX_MAX_U32;

  for (var idx = block_id * 256u + tid; idx < amax_params.count; idx = idx + total_threads) {
    let v = amax_logits[idx];
    if (is_better(v, idx, best_val, best_idx)) {
      best_val = v;
      best_idx = idx;
    }
  }

  s_amax_val[tid] = best_val;
  s_amax_idx[tid] = best_idx;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride) {
      let other_val = s_amax_val[tid + stride];
      let other_idx = s_amax_idx[tid + stride];
      if (is_better(other_val, other_idx, s_amax_val[tid], s_amax_idx[tid])) {
        s_amax_val[tid] = other_val;
        s_amax_idx[tid] = other_idx;
      }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (tid == 0u && block_id < amax_params.num_blocks) {
    amax_partial_vals[block_id] = s_amax_val[0];
    amax_partial_idxs[block_id] = s_amax_idx[0];
  }
}

@compute @workgroup_size(256, 1, 1)
fn argmax_p2(
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  var best_val = AMAX_NEG_INF;
  var best_idx = AMAX_MAX_U32;

  for (var b = tid; b < amax_params.num_blocks; b = b + 256u) {
    let v = amax_p2_vals[b];
    let idx = amax_p2_idxs[b];
    if (is_better(v, idx, best_val, best_idx)) {
      best_val = v;
      best_idx = idx;
    }
  }

  s_amax_val[tid] = best_val;
  s_amax_idx[tid] = best_idx;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride) {
      let other_val = s_amax_val[tid + stride];
      let other_idx = s_amax_idx[tid + stride];
      if (is_better(other_val, other_idx, s_amax_val[tid], s_amax_idx[tid])) {
        s_amax_val[tid] = other_val;
        s_amax_idx[tid] = other_idx;
      }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (tid == 0u) {
    amax_out_idx[0] = s_amax_idx[0];
    amax_out_val[0] = s_amax_val[0];
  }
}
