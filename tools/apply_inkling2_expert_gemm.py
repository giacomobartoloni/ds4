#!/usr/bin/env python3
"""Apply the isolated Inkling-2 32-token expert-bucketed IQ4_NL experiment.

Run on the exact PR #2 head (9c54c4b). No network or model files required.
--check checks all anchors; --apply writes the changed files atomically-ish.
"""
from __future__ import annotations
import argparse
from pathlib import Path

FILES = (
    'ds4.c', 'ds4_gpu.h', 'ds4_metal.m', 'ds4_inkling_moe.c',
    'ds4_inkling_gpu_state.h', 'ds4_inkling_gpu_state.c',
    'metal/inkling2.metal', 'tests/test_inkling2_metal.c',
    'tests/test_inkling2_session.c',
)


def replace_once(s: str, before: str, after: str, where: str) -> str:
    n = s.count(before)
    if n != 1:
        raise RuntimeError(f'{where}: expected exactly one anchor, found {n}: {before[:125]!r}')
    return s.replace(before, after, 1)


def patch_c(s):
    return replace_once(s, '    INKLING_GPU_PREFILL_BATCH_TOKENS = 8,',
                        '    INKLING_GPU_PREFILL_BATCH_TOKENS = DS4_INKLING_GPU_MOE_MAX_TOKENS,', 'ds4.c')


def patch_header(s):
    s = replace_once(s, 'enum { DS4_INKLING_GPU_MOE_MAX_TOKENS = 8u };',
                        'enum { DS4_INKLING_GPU_MOE_MAX_TOKENS = 32u };', 'ds4_gpu.h batch')
    addition = '''/* Selected-expert IQ4_NL grouped path. The map is [expert][rank] with a
 * stride of n_tokens*n_used routes and all route IDs stay on the GPU.
 * The high-level Inkling dispatcher alone decides whether to use this path;
 * the low-level routines also allow smaller fixtures for oracle tests. */
int ds4_gpu_inkling2_expert_grouped_available(void);
int ds4_gpu_inkling2_expert_route_map_rows_tensor(
        ds4_gpu_tensor       *route_map,
        ds4_gpu_tensor       *route_counts,
        const ds4_gpu_tensor *expert_ids,
        uint32_t n_tokens, uint32_t n_experts, uint32_t n_used);
int ds4_gpu_inkling2_expert_gemm_id_rows_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        uint32_t              n_experts,
        uint32_t              n_used,
        uint32_t              n_tokens,
        int                   input_per_expert,
        const ds4_gpu_tensor *route_map,
        const ds4_gpu_tensor *route_counts,
        const ds4_gpu_tensor *x);

'''
    return replace_once(s,
        '/* Batched form of selected expert GEMV. IDs and outputs use',
        addition + '/* Batched form of selected expert GEMV. IDs and outputs use', 'ds4_gpu.h API')


def patch_state_header(s):
    return replace_once(s, '    ds4_gpu_tensor *zero_bias;\n} ds4_inkling_gpu_moe_workspace;',
        '    ds4_gpu_tensor *zero_bias;\n    ds4_gpu_tensor *route_map;\n    ds4_gpu_tensor *route_counts;\n} ds4_inkling_gpu_moe_workspace;',
        'ds4_inkling_gpu_state.h')


def patch_state(s):
    s = replace_once(s, '#define DS4_INKLING_GPU_WORK_BYTES (8u * 1024u * 1024u)',
                     '#define DS4_INKLING_GPU_WORK_BYTES (16u * 1024u * 1024u)',
                     'state work capacity')
    s = replace_once(s,
        '    ds4_gpu_tensor_free(workspace->zero_bias);\n    memset(workspace, 0, sizeof(*workspace));',
        '    ds4_gpu_tensor_free(workspace->zero_bias);\n'
        '    ds4_gpu_tensor_free(workspace->route_map);\n'
        '    ds4_gpu_tensor_free(workspace->route_counts);\n'
        '    memset(workspace, 0, sizeof(*workspace));', 'state cleanup')
    s = replace_once(s,
        '    moe->zero_bias = take_float_view(context->scratch, scratch_bytes,\n'
        '        &cursor, DS4_INKLING_GPU_EXPERTS);',
        '    moe->zero_bias = take_float_view(context->scratch, scratch_bytes,\n'
        '        &cursor, DS4_INKLING_GPU_EXPERTS);\n'
        '    /* GPU-only map: per-expert stable route indices plus counts. Worst\n'
        '     * case holds every route for one expert, not just one per token. */\n'
        '    moe->route_map = take_view(context->scratch, scratch_bytes, &cursor,\n'
        '        (uint64_t)DS4_INKLING_GPU_EXPERTS *\n'
        '        (DS4_INKLING_GPU_MOE_MAX_TOKENS * 8u) * sizeof(uint32_t));\n'
        '    moe->route_counts = take_view(context->scratch, scratch_bytes, &cursor,\n'
        '        (uint64_t)DS4_INKLING_GPU_EXPERTS * sizeof(uint32_t));',
        'state allocation')
    s = replace_once(s,
        '        !moe->zero_bias || !batch->current_storage || !batch->next_storage ||',
        '        !moe->zero_bias || !moe->route_map || !moe->route_counts ||\n'
        '        !batch->current_storage || !batch->next_storage ||', 'state required views')
    return s


def patch_moe(s):
    find = '''    if (ok && !ds4_gpu_inkling2_expert_gemv_rows_tensor(
                  workspace->gate_up, model_map, model_size, gate_up_offset,
                  expert_weight_type, hidden_dim, 2u * expert_dim,
                  n_experts, n_used, n_tokens, 0, workspace->ids, input)) {
        ok = 0;
        failed_stage = "gate/up projection";
    }'''
    new = '''    /* The first experiment is deliberately limited to the real Inkling
     * IQ4_NL group-32 shape. Other shapes retain the existing GEMV route. */
    const int grouped = n_tokens == 32u && n_experts == 128u &&
        n_used == 8u && hidden_dim == 2304u && expert_dim == 1152u &&
        expert_weight_type == DS4_INKLING_GGML_TYPE_IQ4_NL &&
        workspace->route_map && workspace->route_counts &&
        ds4_gpu_tensor_bytes(workspace->route_map) >=
            (uint64_t)n_experts * route_count * sizeof(uint32_t) &&
        ds4_gpu_tensor_bytes(workspace->route_counts) >=
            (uint64_t)n_experts * sizeof(uint32_t) &&
        ds4_gpu_inkling2_expert_grouped_available();
    if (ok && grouped && !ds4_gpu_inkling2_expert_route_map_rows_tensor(
            workspace->route_map, workspace->route_counts, workspace->ids,
            n_tokens, n_experts, n_used)) {
        ok = 0;
        failed_stage = "expert route map";
    }
    if (ok && !(grouped ?
          ds4_gpu_inkling2_expert_gemm_id_rows_tensor(
              workspace->gate_up, model_map, model_size, gate_up_offset,
              hidden_dim, 2u * expert_dim, n_experts, n_used, n_tokens, 0,
              workspace->route_map, workspace->route_counts, input) :
          ds4_gpu_inkling2_expert_gemv_rows_tensor(
              workspace->gate_up, model_map, model_size, gate_up_offset,
              expert_weight_type, hidden_dim, 2u * expert_dim,
              n_experts, n_used, n_tokens, 0, workspace->ids, input))) {
        ok = 0;
        failed_stage = "gate/up projection";
    }'''
    s = replace_once(s, find, new, 'moe gate/up')
    find = '''    if (ok && !ds4_gpu_inkling2_expert_gemv_rows_tensor(
                  workspace->expert_outputs, model_map, model_size, down_offset,
                  expert_weight_type, expert_dim, hidden_dim,
                  n_experts, n_used, n_tokens, 1, workspace->ids,
                  workspace->activation)) {
        ok = 0;
        failed_stage = "down projection";
    }'''
    new = '''    if (ok && !(grouped ?
          ds4_gpu_inkling2_expert_gemm_id_rows_tensor(
              workspace->expert_outputs, model_map, model_size, down_offset,
              expert_dim, hidden_dim, n_experts, n_used, n_tokens, 1,
              workspace->route_map, workspace->route_counts,
              workspace->activation) :
          ds4_gpu_inkling2_expert_gemv_rows_tensor(
              workspace->expert_outputs, model_map, model_size, down_offset,
              expert_weight_type, expert_dim, hidden_dim,
              n_experts, n_used, n_tokens, 1, workspace->ids,
              workspace->activation))) {
        ok = 0;
        failed_stage = "down projection";
    }'''
    return replace_once(s, find, new, 'moe down')


def patch_metal(s):
    # Shared IQ4 decoder ensures both old and grouped paths use exactly the
    # same nibble/scale/codebook interpretation.
    helper = '''static inline float inkling2_iq4_nl_weight(
        device const char *row_weights, uint feature) {
    const uint block_index = feature >> 5u;
    const uint within = feature & 31u;
    device const char *block = row_weights + (ulong)block_index * 18u;
    const float scale = float(*(device const half *)block);
    device const uchar *packed = (device const uchar *)(block + 2);
    const uchar byte = packed[within & 15u];
    const uint code = within < 16u ? (uint)(byte & 0x0fu) : (uint)(byte >> 4u);
    return scale * float(ds4_inkling2_iq4_nl_values[code]);
}

'''
    s = replace_once(s,
        '/* One lane owns one token. The fixed 128-way scan is small, deterministic,',
        helper + '/* One lane owns one token. The fixed 128-way scan is small, deterministic,',
        'metal shared decoder')
    exact = '''        const uint block_index = i >> 5u;
        const uint within = i & 31u;
        device const char *block = row_weights + (uint64_t)block_index * 18u;
        const float scale = float(*(device const half *)block);
        device const uchar *packed = (device const uchar *)(block + 2);
        const uchar byte = packed[within & 15u];
        const uint code = within < 16u ? (uint)(byte & 0x0fu) : (uint)(byte >> 4u);
        sum += scale * float(ds4_inkling2_iq4_nl_values[code]) * x[i];'''
    s = replace_once(s, exact,
        '        sum += inkling2_iq4_nl_weight(row_weights, i) * x[i];',
        'metal ordinary IQ4 GEMV')
    exact = '''            device const uchar *packed = (device const uchar *)(block + 2);
            const uchar byte = packed[within & 15u];
            const uint code = within < 16u ? (uint)(byte & 0x0fu) : (uint)(byte >> 4u);
            const ulong x_offset = args.input_per_expert != 0u ?
                (ulong)route * args.in_dim : (ulong)token * args.in_dim;
            sum += scale * float(ds4_inkling2_iq4_nl_values[code]) * x[x_offset + i];'''
    s = replace_once(s, exact, '''            const ulong x_offset = args.input_per_expert != 0u ?
                (ulong)route * args.in_dim : (ulong)token * args.in_dim;
            sum += inkling2_iq4_nl_weight(row_weights, i) * x[x_offset + i];''',
        'metal expert IQ4 GEMV')
    structs = '''/* Expert-bucketed route and matrix arguments have matching C structures in
 * ds4_metal.m. The map stride is n_tokens*n_used (max 256). */
struct ds4_metal_args_inkling2_route_map {
    uint32_t n_tokens;
    uint32_t n_experts;
    uint32_t n_used;
    uint32_t route_stride;
};

struct ds4_metal_args_inkling2_expert_gemm {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_experts;
    uint32_t n_used;
    uint32_t n_tokens;
    uint32_t blocks_per_row;
    uint32_t input_per_expert;
    uint32_t route_stride;
    ulong expert_bytes;
};

'''
    s = replace_once(s, 'struct ds4_metal_args_inkling2_situ {',
        structs + 'struct ds4_metal_args_inkling2_situ {', 'metal structs')
    kernels = '''/* One workgroup per expert. Lane zero makes a stable scan of the small
 * token-major route list; no host sorting or readback is performed. The
 * dense stride covers repeated IDs even in adversarial unit fixtures. */
kernel void kernel_inkling2_expert_route_map(
        constant ds4_metal_args_inkling2_route_map &args [[buffer(0)]],
        device const int32_t *ids [[buffer(1)]],
        device uint *route_map [[buffer(2)]],
        device uint *route_counts [[buffer(3)]],
        uint expert [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    if (expert >= args.n_experts || lane != 0u) return;
    const uint routes = args.n_tokens * args.n_used;
    uint count = 0;
    for (uint route = 0; route < routes; route++) {
        if (ids[route] == (int32_t)expert)
            route_map[(ulong)expert * args.route_stride + count++] = route;
    }
    route_counts[expert] = count;
}

/* Group all routes of one expert into a single output-row matrix operation.
 * The dequantized IQ4 row is staged ONCE per group in threadgroup memory;
 * each route is then multiplied against the shared row and scattered back
 * to its original route-major position. No dequantized expert bank exists. */
kernel void kernel_inkling2_expert_gemm_id_iq4_nl(
        constant ds4_metal_args_inkling2_expert_gemm &args [[buffer(0)]],
        device const char *weights [[buffer(1)]],
        device const uint *route_map [[buffer(2)]],
        device const uint *route_counts [[buffer(3)]],
        device const float *x [[buffer(4)]],
        device float *out [[buffer(5)]],
        threadgroup float *weight_row [[threadgroup(0)]],
        threadgroup float *partial [[threadgroup(1)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]],
        uint lanes [[threads_per_threadgroup]]) {
    const uint row = group.x;
    const uint expert = group.y;
    if (row >= args.out_dim || expert >= args.n_experts) return;
    const uint count = route_counts[expert];
    if (count == 0u) return;
    const ulong row_bytes = (ulong)args.blocks_per_row * 18u;
    device const char *row_weights = weights +
        (ulong)expert * args.expert_bytes + (ulong)row * row_bytes;
    for (uint i = lane; i < args.in_dim; i += lanes)
        weight_row[i] = inkling2_iq4_nl_weight(row_weights, i);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint simd_width = 32u;
    const uint simd_groups = (lanes + simd_width - 1u) / simd_width;
    for (uint rank = 0; rank < count; rank++) {
        const uint route = route_map[(ulong)expert * args.route_stride + rank];
        const ulong input_row = args.input_per_expert != 0u ?
            (ulong)route : (ulong)(route / args.n_used);
        float sum = 0.0f;
        for (uint i = lane; i < args.in_dim; i += lanes)
            sum += weight_row[i] * x[input_row * args.in_dim + i];
        const float group_sum = simd_sum(sum);
        if ((lane % simd_width) == 0u) partial[lane / simd_width] = group_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            float value = 0.0f;
            for (uint simd = 0u; simd < simd_groups; simd++) value += partial[simd];
            out[(ulong)route * args.out_dim + row] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

'''
    return replace_once(s,
        'static inline float inkling2_tanh_saturating(float value) {',
        kernels + 'static inline float inkling2_tanh_saturating(float value) {',
        'metal grouped kernels')


def patch_objc(s):
    # Existing ds4_gpu_inkling2_tensor_fits and pipeline helpers are declared
    # above this insertion point. Keep all wrappers at the existing API layer.
    code = r'''/* Inkling 2 expert-ID matrix experiment. This is intentionally separate
 * from general matmul dispatch: only the high-level MoE path opts in. */
typedef struct {
    uint32_t n_tokens;
    uint32_t n_experts;
    uint32_t n_used;
    uint32_t route_stride;
} ds4_gpu_inkling2_route_map_args;

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_experts;
    uint32_t n_used;
    uint32_t n_tokens;
    uint32_t blocks_per_row;
    uint32_t input_per_expert;
    uint32_t route_stride;
    uint64_t expert_bytes;
} ds4_gpu_inkling2_expert_gemm_args;

int ds4_gpu_inkling2_expert_grouped_available(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    return ds4_gpu_get_pipeline("kernel_inkling2_expert_route_map") &&
        ds4_gpu_get_pipeline("kernel_inkling2_expert_gemm_id_iq4_nl");
}

int ds4_gpu_inkling2_expert_route_map_rows_tensor(
        ds4_gpu_tensor       *route_map,
        ds4_gpu_tensor       *route_counts,
        const ds4_gpu_tensor *expert_ids,
        uint32_t n_tokens, uint32_t n_experts, uint32_t n_used) {
    if ((!g_initialized && !ds4_gpu_init()) || !route_map || !route_counts ||
        !expert_ids || n_tokens == 0u || n_tokens > 32u ||
        n_experts == 0u || n_experts > 128u || n_used == 0u ||
        n_used > 8u || n_used > n_experts) return 0;
    const uint64_t routes = (uint64_t)n_tokens * n_used;
    const uint64_t map_bytes = (uint64_t)n_experts * routes * sizeof(uint32_t);
    const uint64_t count_bytes = (uint64_t)n_experts * sizeof(uint32_t);
    if (!ds4_gpu_inkling2_tensor_fits(expert_ids, routes * sizeof(int32_t),
                                      "expert IDs") ||
        !ds4_gpu_inkling2_tensor_fits(route_map, map_bytes, "expert route map") ||
        !ds4_gpu_inkling2_tensor_fits(route_counts, count_bytes,
                                      "expert route counts")) return 0;

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_inkling2_expert_route_map");
        if (!pipeline) return 0;
        ds4_gpu_inkling2_route_map_args args = {
            .n_tokens = n_tokens, .n_experts = n_experts,
            .n_used = n_used, .route_stride = (uint32_t)routes,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:ds4_gpu_tensor_buffer(expert_ids)
                offset:ds4_gpu_tensor_offset(expert_ids) atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(route_map)
                offset:ds4_gpu_tensor_offset(route_map) atIndex:2];
        [enc setBuffer:ds4_gpu_tensor_buffer(route_counts)
                offset:ds4_gpu_tensor_offset(route_counts) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(n_experts, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned,
                                            "Inkling2 expert route map")) return 0;
    }
    return 1;
}

int ds4_gpu_inkling2_expert_gemm_id_rows_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        uint32_t              n_experts,
        uint32_t              n_used,
        uint32_t              n_tokens,
        int                   input_per_expert,
        const ds4_gpu_tensor *route_map,
        const ds4_gpu_tensor *route_counts,
        const ds4_gpu_tensor *x) {
    if ((!g_initialized && !ds4_gpu_init()) || !out || !model_map ||
        !route_map || !route_counts || !x ||
        in_dim == 0u || in_dim > 2304u || (in_dim % 32u) != 0u ||
        out_dim == 0u || n_experts == 0u || n_experts > 128u ||
        n_tokens == 0u || n_tokens > 32u || n_used == 0u ||
        n_used > 8u || n_used > n_experts ||
        (input_per_expert != 0 && input_per_expert != 1)) return 0;
    const uint64_t routes = (uint64_t)n_tokens * n_used;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 18u;
    const uint64_t expert_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_bytes = (uint64_t)n_experts * expert_bytes;
    const uint64_t x_rows = input_per_expert ? routes : n_tokens;
    const uint64_t output_bytes = routes * out_dim * sizeof(float);
    const uint64_t input_bytes = x_rows * in_dim * sizeof(float);
    const uint64_t map_bytes = (uint64_t)n_experts * routes * sizeof(uint32_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        !ds4_gpu_inkling2_tensor_fits(out, output_bytes, "expert GEMM output") ||
        !ds4_gpu_inkling2_tensor_fits(x, input_bytes, "expert GEMM input") ||
        !ds4_gpu_inkling2_tensor_fits(route_map, map_bytes, "expert GEMM route map") ||
        !ds4_gpu_inkling2_tensor_fits(route_counts,
                (uint64_t)n_experts * sizeof(uint32_t), "expert GEMM counts")) return 0;

    @autoreleasepool {
        uint64_t inner_offset = 0;
        id<MTLBuffer> wbuf = ds4_gpu_wrap_model_range(
            model_map, model_size, weight_offset, weight_bytes, &inner_offset);
        id<MTLComputePipelineState> pipeline =
            ds4_gpu_get_pipeline("kernel_inkling2_expert_gemm_id_iq4_nl");
        if (!wbuf || !pipeline) return 0;
        ds4_gpu_inkling2_expert_gemm_args args = {
            .in_dim = in_dim, .out_dim = out_dim,
            .n_experts = n_experts, .n_used = n_used, .n_tokens = n_tokens,
            .blocks_per_row = in_dim / 32u, .input_per_expert = input_per_expert,
            .route_stride = (uint32_t)routes, .expert_bytes = expert_bytes,
        };
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:wbuf offset:(NSUInteger)inner_offset atIndex:1];
        [enc setBuffer:ds4_gpu_tensor_buffer(route_map)
                offset:ds4_gpu_tensor_offset(route_map) atIndex:2];
        [enc setBuffer:ds4_gpu_tensor_buffer(route_counts)
                offset:ds4_gpu_tensor_offset(route_counts) atIndex:3];
        [enc setBuffer:ds4_gpu_tensor_buffer(x)
                offset:ds4_gpu_tensor_offset(x) atIndex:4];
        [enc setBuffer:ds4_gpu_tensor_buffer(out)
                offset:ds4_gpu_tensor_offset(out) atIndex:5];
        [enc setThreadgroupMemoryLength:(NSUInteger)in_dim * sizeof(float)
                                 atIndex:0];
        [enc setThreadgroupMemoryLength:256u * sizeof(float) atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, n_experts, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        ds4_gpu_end_compute_encoder(cb, enc);
        if (!ds4_gpu_finish_command_buffer(cb, owned,
                                            "Inkling2 expert grouped IQ4_NL")) return 0;
    }
    return 1;
}

'''
    return replace_once(s, 'int ds4_gpu_inkling2_expert_gemv_rows_tensor(\n',
                        code + 'int ds4_gpu_inkling2_expert_gemv_rows_tensor(\n',
                        'objc new wrappers')


def patch_session_test(s):
    # Group-size-specific old checkpoint test must span > 32 tokens, otherwise
    # it would interrupt at the final group where logits already exist.
    old = '''    const int long_prompt_tokens[] = { 42, 17, 9, 12, 3, 5, 7, 11, 13, 19 };
    const ds4_tokens long_prompt = { .v = (int *)long_prompt_tokens,
        .len = 10, .cap = 10 };
    const ds4_tokens first_eight = { .v = (int *)long_prompt_tokens,
        .len = 8, .cap = 8 };'''
    new = '''    int long_prompt_tokens[DS4_INKLING_GPU_MOE_MAX_TOKENS + 2] =
        { 42, 17, 9, 12, 3, 5, 7, 11, 13, 19 };
    for (int i = 10; i < (int)(sizeof(long_prompt_tokens) /
                               sizeof(long_prompt_tokens[0])); i++)
        long_prompt_tokens[i] = (i * 7) % TEST_VOCAB;
    const int prompt_len = (int)(sizeof(long_prompt_tokens) /
                                 sizeof(long_prompt_tokens[0]));
    const ds4_tokens long_prompt = { .v = long_prompt_tokens,
        .len = prompt_len, .cap = prompt_len };
    const ds4_tokens first_group = { .v = long_prompt_tokens,
        .len = DS4_INKLING_GPU_MOE_MAX_TOKENS,
        .cap = DS4_INKLING_GPU_MOE_MAX_TOKENS };'''
    s=replace_once(s,old,new,'session long prompt fixture')
    s=replace_once(s,'&first_eight, error, sizeof(error)',
        '&first_group, error, sizeof(error)', 'session prefix')
    s=replace_once(s,'cancel_probe.last_current == 8',
        'cancel_probe.last_current == DS4_INKLING_GPU_MOE_MAX_TOKENS', 'session cancel group')
    s=replace_once(s,'assert(ds4_session_pos(session) == 8);',
        'assert(ds4_session_pos(session) == DS4_INKLING_GPU_MOE_MAX_TOKENS);',
        'session position')
    return s


def patch_test(s):
    code = r'''/* The grouped IQ4_NL operator uses a stable per-expert route map.
 * This fixture reuses the small synthetic weights from test_situ_moe_metal,
 * while the high-level path is gated to the real 2304/1152 Inkling shape. */
static int test_expert_grouped_iq4(void) {
    enum { TOKENS = 32, EXPERTS = 128, USED = 8,
           HIDDEN = 32, FFN = 32, ROUTES = TOKENS * USED };
    int ok = 1;
    const uint64_t gate_offset = align_up_u64(
        64u + (uint64_t)HIDDEN * EXPERTS * sizeof(float), 64u);
    const uint64_t gate_bytes =
        (uint64_t)EXPERTS * 2u * FFN * (HIDDEN / 32u * 18u);
    const uint64_t down_offset = align_up_u64(gate_offset + gate_bytes, 64u);
    int32_t ids[ROUTES];
    float input[TOKENS * HIDDEN];
    for (uint32_t token = 0; token < TOKENS; token++) {
        for (uint32_t col = 0; col < HIDDEN; col++)
            input[token * HIDDEN + col] =
                (float)((int32_t)((token * 17u + col * 3u) % 31u) - 15) / 64.0f;
        for (uint32_t slot = 0; slot < USED; slot++)
            ids[token * USED + slot] = (int32_t)((token * 7u + slot * 13u) % EXPERTS);
    }
    ds4_gpu_tensor *tid = upload(ids, sizeof(ids));
    ds4_gpu_tensor *tx = upload(input, sizeof(input));
    ds4_gpu_tensor *gate_ref = ds4_gpu_tensor_alloc(
        (uint64_t)ROUTES * 2u * FFN * sizeof(float));
    ds4_gpu_tensor *gate_gemm = ds4_gpu_tensor_alloc(
        (uint64_t)ROUTES * 2u * FFN * sizeof(float));
    ds4_gpu_tensor *act = ds4_gpu_tensor_alloc(
        (uint64_t)ROUTES * FFN * sizeof(float));
    ds4_gpu_tensor *down_ref = ds4_gpu_tensor_alloc(
        (uint64_t)ROUTES * HIDDEN * sizeof(float));
    ds4_gpu_tensor *down_gemm = ds4_gpu_tensor_alloc(
        (uint64_t)ROUTES * HIDDEN * sizeof(float));
    uint32_t counts[EXPERTS];
    uint32_t routes[EXPERTS * ROUTES];
    uint8_t seen[ROUTES];
    CHECK(tid && tx && gate_ref && gate_gemm && act && down_ref && down_gemm);
    CHECK(ds4_gpu_inkling2_expert_grouped_available());

    /* Validate the mapper at 1, 8 and 32 tokens, including stable ordering. */
    const uint32_t tested_sizes[] = {1u, 8u, 32u};
    for (uint32_t j = 0; j < 3u; j++) {
        const uint32_t n = tested_sizes[j];
        const uint32_t nr = n * USED;
        CHECK(ds4_gpu_inkling2_expert_route_map_rows_tensor(
            g_operator_context.moe.route_map,
            g_operator_context.moe.route_counts, tid, n, EXPERTS, USED));
        CHECK(ds4_gpu_tensor_read(g_operator_context.moe.route_counts, 0,
                                  counts, sizeof(counts)));
        CHECK(ds4_gpu_tensor_read(g_operator_context.moe.route_map, 0,
            routes, (uint64_t)EXPERTS * nr * sizeof(uint32_t)));
        memset(seen, 0, sizeof(seen));
        uint32_t sum = 0;
        for (uint32_t expert = 0; expert < EXPERTS; expert++) {
            uint32_t prev = 0;
            for (uint32_t j = 0; j < counts[expert]; j++) {
                const uint32_t route = routes[expert * nr + j];
                CHECK(route < nr);
                CHECK(ids[route] == (int32_t)expert);
                CHECK(j == 0u || route > prev);
                CHECK(!seen[route]);
                seen[route] = 1;
                prev = route;
                sum++;
            }
        }
        CHECK(sum == nr);
        for (uint32_t r = 0; r < nr; r++) CHECK(seen[r] == 1u);
    }
    /* Exercise full-count and unused-expert map behavior, even when the
     * repeated IDs would not arise from the production top-8 selector. */
    for (uint32_t r = 0; r < ROUTES; r++) ids[r] = 17;
    CHECK(ds4_gpu_tensor_write(tid, 0, ids, sizeof(ids)));
    CHECK(ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS, EXPERTS, USED));
    CHECK(ds4_gpu_tensor_read(g_operator_context.moe.route_counts, 0,
                              counts, sizeof(counts)));
    CHECK(counts[17] == ROUTES && counts[0] == 0u && counts[127] == 0u);
    for (uint32_t r = 0; r < ROUTES; r++)
        ids[r] = (int32_t)(((r / USED) * 7u + (r % USED) * 13u) % EXPERTS);
    CHECK(ds4_gpu_tensor_write(tid, 0, ids, sizeof(ids)));
    CHECK(ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS, EXPERTS, USED));
    ds4_gpu_tensor *small = ds4_gpu_tensor_alloc(4u);
    CHECK(small);
    CHECK(!ds4_gpu_inkling2_expert_route_map_rows_tensor(
        small, g_operator_context.moe.route_counts, tid, TOKENS, EXPERTS, USED));
    CHECK(!ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS + 1u, EXPERTS, USED));
    ds4_gpu_tensor_free(small);

    CHECK(ds4_gpu_inkling2_expert_gemv_rows_tensor(
        gate_ref, g_synthetic_moe_map, g_synthetic_moe_size, gate_offset,
        20u, HIDDEN, 2u * FFN, EXPERTS, USED, TOKENS, 0, tid, tx));
    CHECK(ds4_gpu_inkling2_expert_gemm_id_rows_tensor(
        gate_gemm, g_synthetic_moe_map, g_synthetic_moe_size, gate_offset,
        HIDDEN, 2u * FFN, EXPERTS, USED, TOKENS, 0,
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tx));
    float gate_a[ROUTES * 2u * FFN], gate_b[ROUTES * 2u * FFN];
    CHECK(ds4_gpu_tensor_read(gate_ref, 0, gate_a, sizeof(gate_a)));
    CHECK(ds4_gpu_tensor_read(gate_gemm, 0, gate_b, sizeof(gate_b)));
    for (uint32_t i = 0; i < ROUTES * 2u * FFN; i++)
        CHECK(near(gate_a[i], gate_b[i],
                   6e-4f * fmaxf(1.0f, fabsf(gate_a[i]))));

    CHECK(ds4_gpu_inkling2_situ_rows_tensor(
        act, gate_ref, TOKENS, USED, FFN, 1.25f, 0.75f));
    CHECK(ds4_gpu_inkling2_expert_gemv_rows_tensor(
        down_ref, g_synthetic_moe_map, g_synthetic_moe_size, down_offset,
        20u, FFN, HIDDEN, EXPERTS, USED, TOKENS, 1, tid, act));
    CHECK(ds4_gpu_inkling2_expert_gemm_id_rows_tensor(
        down_gemm, g_synthetic_moe_map, g_synthetic_moe_size, down_offset,
        FFN, HIDDEN, EXPERTS, USED, TOKENS, 1,
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, act));
    float down_a[ROUTES * HIDDEN], down_b[ROUTES * HIDDEN];
    CHECK(ds4_gpu_tensor_read(down_ref, 0, down_a, sizeof(down_a)));
    CHECK(ds4_gpu_tensor_read(down_gemm, 0, down_b, sizeof(down_b)));
    for (uint32_t i = 0; i < ROUTES * HIDDEN; i++)
        CHECK(near(down_a[i], down_b[i],
                   6e-4f * fmaxf(1.0f, fabsf(down_a[i]))));

cleanup:
    ds4_gpu_tensor_free(tid);
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(gate_ref);
    ds4_gpu_tensor_free(gate_gemm);
    ds4_gpu_tensor_free(act);
    ds4_gpu_tensor_free(down_ref);
    ds4_gpu_tensor_free(down_gemm);
    return ok;
}

'''
    s=replace_once(s, 'static int test_loop_mix_and_control_metal(void) {',
        code + 'static int test_loop_mix_and_control_metal(void) {', 'metal test function')
    s=replace_once(s,
        '    CHECK(test_situ_moe_metal());',
        '    CHECK(test_situ_moe_metal());\n    CHECK(test_expert_grouped_iq4());',
        'metal test call')
    return s


PATCH = {
    'ds4.c': patch_c,
    'ds4_gpu.h': patch_header,
    'ds4_metal.m': patch_objc,
    'ds4_inkling_moe.c': patch_moe,
    'ds4_inkling_gpu_state.h': patch_state_header,
    'ds4_inkling_gpu_state.c': patch_state,
    'metal/inkling2.metal': patch_metal,
    'tests/test_inkling2_metal.c': patch_test,
    'tests/test_inkling2_session.c': patch_session_test,
}


def run(root: Path, write: bool) -> None:
    changes = {}
    for filename in FILES:
        path = root / filename
        if not path.is_file():
            raise RuntimeError(f'{filename}: source not found')
        before = path.read_text()
        after = PATCH[filename](before)
        if before == after:
            raise RuntimeError(f'{filename}: patch did not modify source')
        changes[path] = after
        print(f'{filename}: {len(before)} -> {len(after)} bytes')
    if write:
        for path, content in changes.items():
            path.write_text(content)
        print('Applied all modifications')
    else:
        print('All source anchors verified; no files modified')


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--check', action='store_true')
    cli.add_argument('--apply', action='store_true')
    cli.add_argument('--root', type=Path, default=Path('.'))
    args = cli.parse_args()
    if args.check == args.apply:
        cli.error('select exactly one of --check or --apply')
    run(args.root, args.apply)
