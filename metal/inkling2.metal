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

struct ds4_metal_args_inkling2_expert_gemv {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_experts;
    uint32_t n_used;
    uint32_t weight_type;
    uint32_t blocks_per_row;
    uint32_t input_per_expert;
    uint32_t pad0;
    ulong expert_bytes;
};

struct ds4_metal_args_inkling2_situ {
    uint32_t n_used;
    uint32_t expert_dim;
    float beta;
    float linear_beta;
};

struct ds4_metal_args_inkling2_qk {
    uint32_t rows;
    uint32_t width;
    float threshold;
    float log_scale_alpha;
    float log_scale_floor;
    uint32_t apply_log_scale;
    uint32_t pad0;
    uint32_t pad1;
};

struct ds4_metal_args_inkling2_relative_bias {
    uint32_t rows;
    uint32_t heads;
    uint32_t relative_dim;
    uint32_t relative_extent;
    uint32_t keys;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

struct ds4_metal_args_inkling2_attention {
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t capacity;
    uint32_t count;
    uint32_t sliding_window;
    uint32_t relative_extent;
    uint32_t has_sink;
    ulong oldest_position;
    ulong query_position;
};

struct ds4_metal_args_inkling2_shortconv {
    uint32_t n_tokens;
    uint32_t n_channels;
    uint32_t n_heads;
    uint32_t kernel_size;
    uint32_t history_count;
    uint32_t has_dynamic_taps;
    uint32_t pad0;
    uint32_t pad1;
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

/* Selected-expert GEMV keeps routing IDs on device and gathers rows from the
 * shared physical expert tensor. Q8_0 and IQ4_NL use their native block bytes. */
kernel void kernel_inkling2_expert_gemv(
        constant ds4_metal_args_inkling2_expert_gemv &args [[buffer(0)]],
        device const char *weights [[buffer(1)]],
        device const int32_t *expert_ids [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device float *out [[buffer(4)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]],
        uint lanes [[threads_per_threadgroup]]) {
    const uint slot = group / args.out_dim;
    const uint row = group - slot * args.out_dim;
    if (group >= args.out_dim * args.n_used || row >= args.out_dim || slot >= args.n_used) return;
    const int32_t expert = expert_ids[slot];
    if (expert < 0 || (uint32_t)expert >= args.n_experts) {
        if (lane == 0u) out[(ulong)slot * args.out_dim + row] = NAN;
        return;
    }
    const uint block_bytes = args.weight_type == 20u ? 18u : 34u;
    const ulong row_bytes = (ulong)args.blocks_per_row * block_bytes;
    device const char *row_weights = weights + (ulong)(uint32_t)expert * args.expert_bytes +
        (ulong)row * row_bytes;
    float sum = 0.0f;
    for (uint i = lane; i < args.in_dim; i += lanes) {
        const uint block_index = i >> 5u;
        const uint within = i & 31u;
        device const char *block = row_weights + (ulong)block_index * block_bytes;
        const float scale = float(*(device const half *)block);
        if (args.weight_type == 20u) {
            device const uchar *packed = (device const uchar *)(block + 2);
            const uchar byte = packed[within & 15u];
            const uint code = within < 16u ? (uint)(byte & 0x0fu) : (uint)(byte >> 4u);
            const ulong x_offset = args.input_per_expert != 0u ?
                (ulong)slot * args.in_dim : 0u;
            sum += scale * float(ds4_inkling2_iq4_nl_values[code]) * x[x_offset + i];
        } else {
            const int8_t quant = *(device const int8_t *)(block + 2 + within);
            const ulong x_offset = args.input_per_expert != 0u ?
                (ulong)slot * args.in_dim : 0u;
            sum += scale * float(quant) * x[x_offset + i];
        }
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes >> 1u; stride > 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) out[(ulong)slot * args.out_dim + row] = partial[0];
}

static inline float inkling2_tanh_saturating(float value) {
    if (isinf(value)) return value >= 0.0f ? 1.0f : -1.0f;
    if (value >= 10.0f) return 1.0f;
    if (value <= -10.0f) return -1.0f;
    return tanh(value);
}

kernel void kernel_inkling2_situ(
        constant ds4_metal_args_inkling2_situ &args [[buffer(0)]],
        device const float *gate_up [[buffer(1)]],
        device float *out [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    const uint count = args.n_used * args.expert_dim;
    if (index >= count) return;
    const uint slot = index / args.expert_dim;
    const uint row = index - slot * args.expert_dim;
    const ulong base = (ulong)slot * 2u * args.expert_dim;
    const float gate = gate_up[base + row];
    const float up = gate_up[base + args.expert_dim + row];
    const float gate_act = args.beta * inkling2_tanh_saturating(gate / args.beta) *
        inkling2_sigmoid(gate);
    const float up_act = args.linear_beta *
        inkling2_tanh_saturating(up / args.linear_beta);
    out[index] = gate_act * up_act;
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

/* Per-head scaled L2 clipping for Q/K/R states. Full-attention Q and relative
 * states apply log scaling only after the learned scale and clipping stages. */
kernel void kernel_inkling2_qk_prepare(
        constant ds4_metal_args_inkling2_qk &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const float *learned_scale [[buffer(2)]],
        device const ulong *positions [[buffer(3)]],
        device float *out [[buffer(4)]],
        threadgroup float *partial [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]],
        uint lanes [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;
    const uint64_t offset = (uint64_t)row * args.width;
    float sum = 0.0f;
    for (uint col = lane; col < args.width; col += lanes) {
        const float value = x[offset + col] * learned_scale[offset + col];
        sum += value * value;
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes >> 1u; stride > 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float norm = sqrt(partial[0]);
    const float clip = min(args.threshold / (norm + 1e-6f), 1.0f);
    float log_scale = 1.0f;
    if (args.apply_log_scale != 0u) {
        const float ratio = max((float(positions[row]) + 1.0f) /
                                args.log_scale_floor, 1.0f);
        log_scale += args.log_scale_alpha * log(ratio);
    }
    for (uint col = lane; col < args.width; col += lanes) {
        out[offset + col] = x[offset + col] * learned_scale[offset + col] *
                            clip * log_scale;
    }
}

/* Project the per-query relative state against each learned distance profile.
 * Distances outside the learned extent contribute zero, as in the reference. */
kernel void kernel_inkling2_relative_bias(
        constant ds4_metal_args_inkling2_relative_bias &args [[buffer(0)]],
        device const float *relative_states [[buffer(1)]],
        device const float *projection [[buffer(2)]],
        device const long *relative_distances [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    const uint64_t count = (uint64_t)args.rows * args.heads * args.keys;
    if (index >= count) return;
    const uint key = index % args.keys;
    const uint head = (index / args.keys) % args.heads;
    const uint row = index / ((uint64_t)args.keys * args.heads);
    const long distance = relative_distances[(uint64_t)row * args.keys + key];
    float sum = 0.0f;
    if (distance >= 0 && (ulong)distance < args.relative_extent) {
        const uint64_t state_offset =
            ((uint64_t)row * args.heads + head) * args.relative_dim;
        for (uint dim = 0; dim < args.relative_dim; dim++) {
            sum += relative_states[state_offset + dim] *
                   projection[(uint64_t)dim * args.relative_extent + (uint)distance];
        }
    }
    out[index] = sum;
}

/* One threadgroup owns each query head. Lanes cooperate over head dimensions;
 * the online softmax avoids materializing an 8192-entry score row while it
 * includes the learned zero-value sink in its denominator. */
kernel void kernel_inkling2_attention_decode(
        constant ds4_metal_args_inkling2_attention &args [[buffer(0)]],
        device const float *query [[buffer(1)]],
        device const half *key_cache [[buffer(2)]],
        device const half *value_cache [[buffer(3)]],
        device const float *relative_bias [[buffer(4)]],
        device const float *sink_logits [[buffer(5)]],
        device float *out [[buffer(6)]],
        uint head [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]],
        uint lanes [[threads_per_threadgroup]]) {
    if (head >= args.query_heads) return;
    threadgroup float partial[128];
    threadgroup float accumulated_value[128];
    threadgroup float current_logit;
    threadgroup float maximum;
    threadgroup float denominator;
    threadgroup float old_scale;
    threadgroup float new_scale;

    const uint heads_per_kv = args.query_heads / args.kv_heads;
    const uint kv_head = head / heads_per_kv;
    ulong key_start = args.oldest_position;
    ulong key_end = min(args.oldest_position + args.count,
                        args.query_position + 1ul);
    if (args.sliding_window != 0u) {
        const ulong window_start = args.query_position + 1ul > args.sliding_window
            ? args.query_position + 1ul - args.sliding_window : 0ul;
        key_start = max(key_start, window_start);
    }
    key_start = min(key_start, key_end);
    const float scale = rsqrt((float)args.head_dim);
    const float sink = args.has_sink != 0u ? sink_logits[head] : -FLT_MAX;
    const device float *q = query + (uint64_t)head * args.head_dim;
    accumulated_value[lane] = 0.0f;
    if (lane == 0u) {
        maximum = sink;
        denominator = args.has_sink != 0u ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ulong position = key_start; position < key_end; position++) {
        const uint ring = (uint)(position % args.capacity);
        const uint64_t row = ((uint64_t)ring * args.kv_heads + kv_head) * args.head_dim;
        float dot = 0.0f;
        for (uint dim = lane; dim < args.head_dim; dim += lanes) {
            dot += q[dim] * float(key_cache[row + dim]);
        }
        partial[lane] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = lanes >> 1u; stride > 0u; stride >>= 1u) {
            if (lane < stride) partial[lane] += partial[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const ulong distance = args.query_position - position;
        const float bias = distance < args.relative_extent
            ? relative_bias[(uint64_t)head * args.relative_extent + distance] : 0.0f;
        if (lane == 0u) {
            current_logit = partial[0] * scale + bias;
            const float next_maximum = max(maximum, current_logit);
            old_scale = denominator == 0.0f ? 0.0f : exp(maximum - next_maximum);
            new_scale = exp(current_logit - next_maximum);
            denominator = denominator * old_scale + new_scale;
            maximum = next_maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float value = lane < args.head_dim
            ? float(value_cache[row + lane]) : 0.0f;
        accumulated_value[lane] = accumulated_value[lane] * old_scale +
                                  new_scale * value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane < args.head_dim) {
        out[(uint64_t)head * args.head_dim + lane] = denominator > 0.0f
            ? accumulated_value[lane] / denominator : 0.0f;
    }
}

/* The attention gate is a sigmoid projection of the layer input, applied
 * before the output projection. */
kernel void kernel_inkling2_attention_output_gate(
        constant ds4_metal_args_inkling2 &args [[buffer(0)]],
        device const float *attention [[buffer(1)]],
        device const float *gate_logits [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.count) return;
    const float logit = gate_logits[index];
    const float gate = isnan(logit) ? logit : inkling2_sigmoid(logit);
    out[index] = attention[index] * gate;
}

/* Causal depthwise short convolution. The history stays chronological and is
 * updated by the same channel owner after all output tokens have consumed it. */
kernel void kernel_inkling2_shortconv(
        constant ds4_metal_args_inkling2_shortconv &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const float *static_taps [[buffer(2)]],
        device const float *dynamic_taps [[buffer(3)]],
        device float *history [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint channel [[thread_position_in_grid]]) {
    if (channel >= args.n_channels) return;
    const uint width = args.n_channels;
    const uint head_dim = width / args.n_heads;
    const uint head = channel / head_dim;
    float previous[3] = {
        history[channel], history[width + channel], history[2u * width + channel]
    };
    uint history_count = args.history_count;
    for (uint i = 0; i < 3u - history_count; i++) previous[i] = 0.0f;
    const float divisor = sqrt((float)((args.has_dynamic_taps != 0u ? 2u : 1u) *
                                       args.kernel_size + 1u));
    for (uint token = 0; token < args.n_tokens; token++) {
        const float current = x[(uint64_t)token * width + channel];
        float sum = 0.0f;
        for (uint lag = 0; lag < args.kernel_size; lag++) {
            float sample = current;
            if (lag != 0u) {
                sample = lag <= history_count ? previous[3u - lag] : 0.0f;
            }
            float tap = static_taps[(uint64_t)lag * width + channel];
            if (args.has_dynamic_taps != 0u) {
                tap += dynamic_taps[((uint64_t)token * args.n_heads + head) *
                                    args.kernel_size + lag];
            }
            sum += sample * tap;
        }
        out[(uint64_t)token * width + channel] = (current + sum) / divisor;
        previous[0] = previous[1];
        previous[1] = previous[2];
        previous[2] = current;
        if (history_count < 3u) history_count++;
    }
    history[channel] = previous[0];
    history[width + channel] = previous[1];
    history[2u * width + channel] = previous[2];
}
