/* Native controller and router kernels for the Inkling 2 logical-loop model. */

struct ds4_metal_args_inkling2 {
    uint32_t count;
    uint32_t n;
    uint32_t rows;
    uint32_t n_experts;
    uint32_t n_used;
    float scalar;
};

struct ds4_metal_args_inkling2_iq4_nl {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t blocks_per_row;
    uint32_t pad;
};

/* Canonical GGML IQ4_NL codebook from ggml-common.h (`kvalues_iq4nl`). */
constant short ds4_inkling2_iq4_nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

kernel void kernel_inkling2_gate_update(
        constant ds4_metal_args_inkling2 &args [[buffer(0)]],
        device const float *current [[buffer(1)]],
        device const float *message [[buffer(2)]],
        device const float *gain [[buffer(3)]],
        device const float *forget [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.count) return;
    const float keep = 1.0f - forget[index];
    out[index] = keep * current[index] + gain[index] * message[index];
}

kernel void kernel_inkling2_clip_l2_rows(
        constant ds4_metal_args_inkling2 &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device float *out [[buffer(2)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]],
        uint lanes [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;
    device const float *row_x = x + (uint64_t)row * args.n;
    device float *row_out = out + (uint64_t)row * args.n;

    float sum = 0.0f;
    for (uint i = lane; i < args.n; i += lanes) {
        const float v = row_x[i];
        sum += v * v;
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lanes >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float norm = sqrt(partial[0]);
    const float scale = norm > args.scalar ? args.scalar / norm : 1.0f;
    for (uint i = lane; i < args.n; i += lanes) {
        row_out[i] = row_x[i] * scale;
    }
}

kernel void kernel_inkling2_logit_softcap(
        constant ds4_metal_args_inkling2 &args [[buffer(0)]],
        device const float *logits [[buffer(1)]],
        device float *out [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.count) return;
    const float value = logits[index];
    if (isinf(value)) {
        out[index] = value > 0.0f ? args.scalar : -args.scalar;
    } else {
        out[index] = args.scalar * tanh(value / args.scalar);
    }
}

static inline float inkling2_sigmoid(float x) {
    if (isnan(x)) return 0.0f;
    if (x >= 0.0f) {
        const float e = exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = exp(x);
    return e / (1.0f + e);
}

/* One lane owns one token. The fixed 128-way scan is small, deterministic,
 * and keeps tie handling and normalization identical across GPU families. */
kernel void kernel_inkling2_router_topk(
        constant ds4_metal_args_inkling2 &args [[buffer(0)]],
        device const float *logits [[buffer(1)]],
        device const float *correction_bias [[buffer(2)]],
        device int32_t *selected [[buffer(3)]],
        device float *weights [[buffer(4)]],
        uint token [[thread_position_in_grid]]) {
    if (token >= args.count) return;
    const uint n_experts = args.n_experts;
    device const float *token_logits = logits + (uint64_t)token * n_experts;
    uint chosen[4] = { 0u, 0u, 0u, 0u };
    float raw_scores[8];
    float total = 0.0f;

    for (uint rank = 0; rank < args.n_used; rank++) {
        int best_id = -1;
        float best_score = -INFINITY;
        for (uint expert = 0; expert < n_experts; expert++) {
            if (chosen[expert >> 5] & (1u << (expert & 31u))) continue;
            float score = inkling2_sigmoid(token_logits[expert]);
            float adjusted = score + correction_bias[expert];
            if (isnan(adjusted)) adjusted = -INFINITY;
            if (best_id < 0 || adjusted > best_score ||
                (adjusted == best_score && expert < (uint)best_id)) {
                best_id = (int)expert;
                best_score = adjusted;
            }
        }

        chosen[(uint)best_id >> 5] |= 1u << ((uint)best_id & 31u);
        selected[(uint64_t)token * args.n_used + rank] = best_id;
        const float score = inkling2_sigmoid(token_logits[(uint)best_id]);
        raw_scores[rank] = score;
        total += score;
    }

    const float denominator = total + 1e-20f;
    for (uint rank = 0; rank < args.n_used; rank++) {
        weights[(uint64_t)token * args.n_used + rank] =
            args.scalar * raw_scores[rank] / denominator;
    }
}

/* IQ4_NL stores 32 values in 18 bytes: an fp16 block scale followed by 16
 * bytes whose low nibbles encode values 0..15 and high nibbles encode 16..31.
 * Each threadgroup owns one output row and decodes weights in place. */
kernel void kernel_inkling2_gemv_iq4_nl(
        constant ds4_metal_args_inkling2_iq4_nl &args [[buffer(0)]],
        device const char *weights [[buffer(1)]],
        device const float *x [[buffer(2)]],
        device float *out [[buffer(3)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]],
        uint lanes [[threads_per_threadgroup]]) {
    if (row >= args.out_dim) return;
    const uint64_t row_bytes = (uint64_t)args.blocks_per_row * 18u;
    device const char *row_weights = weights + (uint64_t)row * row_bytes;
    float sum = 0.0f;
    for (uint i = lane; i < args.in_dim; i += lanes) {
        const uint block_index = i >> 5u;
        const uint within = i & 31u;
        device const char *block = row_weights + (uint64_t)block_index * 18u;
        const float scale = float(*(device const half *)block);
        device const uchar *packed = (device const uchar *)(block + 2);
        const uchar byte = packed[within & 15u];
        const uint code = within < 16u ? (uint)(byte & 0x0fu) : (uint)(byte >> 4u);
        sum += scale * float(ds4_inkling2_iq4_nl_values[code]) * x[i];
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes >> 1u; stride > 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) out[row] = partial[0];
}

/* Accumulate already-computed, selected expert vectors without expanding the
 * quantized expert matrices into a dense temporary. */
kernel void kernel_inkling2_expert_accumulate(
        constant ds4_metal_args_inkling2 &args [[buffer(0)]],
        device const float *expert_outputs [[buffer(1)]],
        device const float *weights [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint row [[thread_position_in_grid]]) {
    if (row >= args.count) return;
    float sum = 0.0f;
    for (uint expert = 0; expert < args.n_used; expert++) {
        sum += weights[expert] * expert_outputs[(uint64_t)expert * args.count + row];
    }
    out[row] = sum;
}
