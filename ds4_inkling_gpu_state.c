#include "ds4_inkling_gpu_state.h"

#include <limits.h>
#include <string.h>

#define DS4_INKLING_GPU_HIDDEN 2304u
#define DS4_INKLING_GPU_VOCAB_PADDED 201024u
#define DS4_INKLING_GPU_QUERY_WIDTH (24u * 128u)
#define DS4_INKLING_GPU_RELATIVE_WIDTH (24u * 16u)
#define DS4_INKLING_GPU_EXPERTS 128u
#define DS4_INKLING_GPU_EXPERT_DIM 1152u
#define DS4_INKLING_GPU_WEIGHT_ALIGNMENT 256u
#define DS4_INKLING_GPU_WORK_BYTES (8u * 1024u * 1024u)

static const uint32_t g_conv_width[DS4_INKLING_CONV_GROUP_COUNT] = {
    DS4_INKLING_KV_WIDTH,
    DS4_INKLING_KV_WIDTH,
    DS4_INKLING_GPU_HIDDEN,
    DS4_INKLING_GPU_HIDDEN,
};

typedef struct {
    uint64_t kv_storage_bytes;
    uint64_t conv_storage_bytes;
    uint64_t residual_bank_bytes;
    uint64_t prelude_bytes;
    uint64_t logits_bytes;
    uint64_t weight_storage_bytes;
    uint64_t total_bytes;
} ds4_inkling_gpu_layout;

static bool checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (b != 0 && a > UINT64_MAX / b)) return false;
    *out = a * b;
    return true;
}

static bool checked_add_bytes(uint64_t *total, uint64_t bytes) {
    return total && checked_add_u64(*total, bytes, total);
}

static bool aligned_parameter_bytes(uint64_t *cursor, uint64_t bytes) {
    if (!cursor || *cursor > UINT64_MAX - (DS4_INKLING_GPU_WEIGHT_ALIGNMENT - 1u))
        return false;
    const uint64_t aligned = (*cursor + DS4_INKLING_GPU_WEIGHT_ALIGNMENT - 1u) &
        ~(uint64_t)(DS4_INKLING_GPU_WEIGHT_ALIGNMENT - 1u);
    return checked_add_u64(aligned, bytes, cursor);
}

static bool add_parameter_bytes(ds4_inkling_gpu_layout *layout,
                                uint64_t elements) {
    uint64_t bytes = 0;
    return checked_mul_u64(elements, sizeof(float), &bytes) &&
        aligned_parameter_bytes(&layout->weight_storage_bytes, bytes);
}

static bool parameter_storage_size(const ds4_inkling_plan *plan,
                                   uint64_t *bytes) {
    if (!plan || !bytes) return false;
    ds4_inkling_gpu_layout layout = {0};
    bool sliding[DS4_INKLING_PHYSICAL_LAYERS] = { false };
    bool found[DS4_INKLING_PHYSICAL_LAYERS] = { false };
    for (uint32_t i = 0; i < plan->visit_count; i++) {
        const ds4_inkling_visit *visit = &plan->visits[i];
        sliding[visit->physical_layer] = visit->is_sliding_window != 0;
        found[visit->physical_layer] = true;
    }
    if (!add_parameter_bytes(&layout, DS4_INKLING_GPU_HIDDEN)) return false;
    for (uint32_t layer = 0; layer < DS4_INKLING_PHYSICAL_LAYERS; layer++) {
        if (!found[layer] ||
            !add_parameter_bytes(&layout, 128u * 24u) ||
            !add_parameter_bytes(&layout, 16u * 1024u) ||
            !add_parameter_bytes(&layout, 16u * 24u) ||
            !add_parameter_bytes(&layout, 4u * DS4_INKLING_KV_WIDTH) ||
            !add_parameter_bytes(&layout, 4u * DS4_INKLING_KV_WIDTH) ||
            !add_parameter_bytes(&layout, 4u * DS4_INKLING_GPU_HIDDEN) ||
            !add_parameter_bytes(&layout, 4u * DS4_INKLING_GPU_HIDDEN) ||
            (layer != 0 && !add_parameter_bytes(&layout,
                DS4_INKLING_GPU_HIDDEN * (sliding[layer] ? 1u : 3u))) ||
            !add_parameter_bytes(&layout, DS4_INKLING_GPU_HIDDEN) ||
            !add_parameter_bytes(&layout, DS4_INKLING_GPU_EXPERTS) ||
            (layer >= DS4_INKLING_LOOP_START && layer < DS4_INKLING_LOOP_END &&
             !add_parameter_bytes(&layout, 4u * DS4_INKLING_GPU_HIDDEN))) {
            return false;
        }
    }
    *bytes = layout.weight_storage_bytes;
    return true;
}

static bool plan_is_valid(const ds4_inkling_plan *plan) {
    if (!plan || plan->physical_layer_count != DS4_INKLING_PHYSICAL_LAYERS ||
        plan->cache_slot_count != DS4_INKLING_CACHE_SLOTS ||
        plan->visit_count != DS4_INKLING_VISITS) return false;
    bool seen[DS4_INKLING_CACHE_SLOTS] = { false };
    for (uint32_t i = 0; i < plan->visit_count; i++) {
        const ds4_inkling_visit *visit = &plan->visits[i];
        if (visit->physical_layer >= DS4_INKLING_PHYSICAL_LAYERS ||
            visit->cache_slot >= DS4_INKLING_CACHE_SLOTS ||
            seen[visit->cache_slot]) return false;
        seen[visit->cache_slot] = true;
    }
    for (uint32_t i = 0; i < DS4_INKLING_CACHE_SLOTS; i++) {
        if (!seen[i]) return false;
    }
    return true;
}

static uint32_t slot_capacity(const ds4_inkling_visit *visit,
                              uint32_t context_capacity) {
    if (visit->is_sliding_window && context_capacity > DS4_INKLING_SLIDING_WINDOW)
        return DS4_INKLING_SLIDING_WINDOW;
    return context_capacity;
}

static bool context_layout(const ds4_inkling_plan *plan,
                           uint32_t context_capacity,
                           ds4_inkling_gpu_layout *layout) {
    if (!layout || !plan_is_valid(plan) || context_capacity == 0 ||
        context_capacity > DS4_INKLING_CONTEXT_MAX) return false;
    memset(layout, 0, sizeof(*layout));

    if (!parameter_storage_size(plan, &layout->weight_storage_bytes)) return false;

    uint64_t total_capacity_rows = 0;
    uint64_t conv_slot_bytes = 0;
    for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
        uint64_t group_bytes = 0;
        if (!checked_mul_u64(DS4_INKLING_CONV_HISTORY, g_conv_width[group],
                             &group_bytes) ||
            !checked_mul_u64(group_bytes, sizeof(float), &group_bytes) ||
            !checked_add_u64(conv_slot_bytes, group_bytes,
                             &conv_slot_bytes)) return false;
    }
    for (uint32_t i = 0; i < plan->visit_count; i++) {
        if (!checked_add_u64(total_capacity_rows,
                             slot_capacity(&plan->visits[i], context_capacity),
                             &total_capacity_rows) ||
            !checked_add_u64(layout->conv_storage_bytes, conv_slot_bytes,
                             &layout->conv_storage_bytes)) return false;
    }

    if (!checked_mul_u64(total_capacity_rows, DS4_INKLING_KV_WIDTH,
                         &layout->kv_storage_bytes) ||
        !checked_mul_u64(layout->kv_storage_bytes, sizeof(uint16_t),
                         &layout->kv_storage_bytes) ||
        !checked_mul_u64(DS4_INKLING_RESIDUAL_BLOCK,
                         DS4_INKLING_GPU_HIDDEN * sizeof(float),
                         &layout->residual_bank_bytes) ||
        !checked_mul_u64(DS4_INKLING_GPU_HIDDEN, sizeof(float),
                         &layout->prelude_bytes) ||
        !checked_mul_u64(DS4_INKLING_GPU_VOCAB_PADDED, sizeof(float),
                         &layout->logits_bytes)) return false;

    return checked_add_bytes(&layout->total_bytes, layout->kv_storage_bytes) &&
        checked_add_bytes(&layout->total_bytes, layout->kv_storage_bytes) &&
        checked_add_bytes(&layout->total_bytes, layout->conv_storage_bytes) &&
        checked_add_bytes(&layout->total_bytes, DS4_INKLING_GPU_WORK_BYTES) &&
        checked_add_bytes(&layout->total_bytes, layout->residual_bank_bytes) &&
        checked_add_bytes(&layout->total_bytes, layout->residual_bank_bytes) &&
        checked_add_bytes(&layout->total_bytes, layout->prelude_bytes) &&
        checked_add_bytes(&layout->total_bytes, sizeof(int32_t)) &&
        checked_add_bytes(&layout->total_bytes, layout->logits_bytes) &&
        checked_add_bytes(&layout->total_bytes, layout->weight_storage_bytes);
}

bool ds4_inkling_gpu_context_required_bytes(
        const ds4_inkling_plan *plan,
        uint32_t context_capacity,
        uint64_t *required_bytes) {
    ds4_inkling_gpu_layout layout;
    if (!required_bytes || !context_layout(plan, context_capacity, &layout))
        return false;
    *required_bytes = layout.total_bytes;
    return true;
}

static void free_slot_views(ds4_inkling_gpu_slot *slot) {
    if (!slot) return;
    ds4_gpu_tensor_free(slot->key);
    ds4_gpu_tensor_free(slot->value);
    for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++)
        ds4_gpu_tensor_free(slot->conv[group]);
    memset(slot, 0, sizeof(*slot));
}

static void free_layer_parameters(ds4_inkling_gpu_layer_parameters *parameters) {
    if (!parameters) return;
    ds4_gpu_tensor_free(parameters->attn_q_clip);
    ds4_gpu_tensor_free(parameters->attn_rel_bias);
    ds4_gpu_tensor_free(parameters->attn_rel_clip);
    ds4_gpu_tensor_free(parameters->attn_k_conv);
    ds4_gpu_tensor_free(parameters->attn_v_conv);
    ds4_gpu_tensor_free(parameters->attn_post_conv);
    ds4_gpu_tensor_free(parameters->ffn_post_conv);
    ds4_gpu_tensor_free(parameters->attn_res_query);
    ds4_gpu_tensor_free(parameters->ffn_res_query);
    ds4_gpu_tensor_free(parameters->router_bias);
    ds4_gpu_tensor_free(parameters->loop_msg_query);
    memset(parameters, 0, sizeof(*parameters));
}

static void free_layer_workspace(ds4_inkling_gpu_layer_workspace *workspace) {
    if (!workspace) return;
    for (uint32_t i = 0; i < 4u; i++) {
        ds4_gpu_tensor_free(workspace->mix_output_rows[i]);
        ds4_gpu_tensor_free(workspace->message_rows[i]);
    }
    for (uint32_t i = 0; i < 3u; i++)
        ds4_gpu_tensor_free(workspace->controlled_input_rows[i]);
    ds4_gpu_tensor_free(workspace->previous_loop_states);
    ds4_gpu_tensor_free(workspace->next);
    ds4_gpu_tensor_free(workspace->mix_states);
    ds4_gpu_tensor_free(workspace->mix_outputs);
    ds4_gpu_tensor_free(workspace->controlled_inputs);
    ds4_gpu_tensor_free(workspace->query);
    ds4_gpu_tensor_free(workspace->prepared_query);
    ds4_gpu_tensor_free(workspace->key);
    ds4_gpu_tensor_free(workspace->prepared_key);
    ds4_gpu_tensor_free(workspace->value);
    ds4_gpu_tensor_free(workspace->prepared_value);
    ds4_gpu_tensor_free(workspace->relative);
    ds4_gpu_tensor_free(workspace->prepared_relative);
    ds4_gpu_tensor_free(workspace->sink_logits);
    ds4_gpu_tensor_free(workspace->relative_bias);
    ds4_gpu_tensor_free(workspace->relative_distances);
    ds4_gpu_tensor_free(workspace->attention_gate);
    ds4_gpu_tensor_free(workspace->attention_context);
    ds4_gpu_tensor_free(workspace->attention_gated);
    ds4_gpu_tensor_free(workspace->attention_output);
    ds4_gpu_tensor_free(workspace->ffn_input);
    ds4_gpu_tensor_free(workspace->ffn_mixed);
    ds4_gpu_tensor_free(workspace->messages);
    ds4_gpu_tensor_free(workspace->dynamic_k);
    ds4_gpu_tensor_free(workspace->dynamic_v);
    memset(workspace, 0, sizeof(*workspace));
}

static void free_control_workspace(ds4_inkling_gpu_control_workspace *workspace) {
    if (!workspace) return;
    ds4_gpu_tensor_free(workspace->normalized);
    ds4_gpu_tensor_free(workspace->bottleneck);
    ds4_gpu_tensor_free(workspace->gain);
    ds4_gpu_tensor_free(workspace->forget);
    memset(workspace, 0, sizeof(*workspace));
}

static void free_moe_workspace(ds4_inkling_gpu_moe_workspace *workspace) {
    if (!workspace) return;
    ds4_gpu_tensor_free(workspace->logits);
    ds4_gpu_tensor_free(workspace->ids);
    ds4_gpu_tensor_free(workspace->weights);
    ds4_gpu_tensor_free(workspace->gate_up);
    ds4_gpu_tensor_free(workspace->activation);
    ds4_gpu_tensor_free(workspace->expert_outputs);
    ds4_gpu_tensor_free(workspace->zero_bias);
    memset(workspace, 0, sizeof(*workspace));
}

static void free_batch_workspace(ds4_inkling_gpu_batch_workspace *workspace) {
    if (!workspace) return;
    for (uint32_t row = 0; row < DS4_INKLING_GPU_MOE_MAX_TOKENS; row++) {
        ds4_gpu_tensor_free(workspace->current[row]);
        ds4_gpu_tensor_free(workspace->next[row]);
        ds4_gpu_tensor_free(workspace->prelude[row]);
        ds4_gpu_tensor_free(workspace->current_blocks[row]);
        ds4_gpu_tensor_free(workspace->previous_blocks[row]);
        ds4_gpu_tensor_free(workspace->ffn_input[row]);
        ds4_gpu_tensor_free(workspace->moe_input[row]);
        ds4_gpu_tensor_free(workspace->moe_output[row]);
        for (uint32_t input = 0; input < 3u; input++)
            ds4_gpu_tensor_free(workspace->attn_norm[input][row]);
        ds4_gpu_tensor_free(workspace->query[row]);
        ds4_gpu_tensor_free(workspace->key[row]);
        ds4_gpu_tensor_free(workspace->value[row]);
        ds4_gpu_tensor_free(workspace->relative[row]);
        ds4_gpu_tensor_free(workspace->dynamic_k[row]);
        ds4_gpu_tensor_free(workspace->dynamic_v[row]);
        ds4_gpu_tensor_free(workspace->sink_logits[row]);
        ds4_gpu_tensor_free(workspace->attention_gate[row]);
        ds4_gpu_tensor_free(workspace->attention_gated[row]);
        ds4_gpu_tensor_free(workspace->attention_output[row]);
    }
    ds4_gpu_tensor_free(workspace->current_storage);
    ds4_gpu_tensor_free(workspace->next_storage);
    ds4_gpu_tensor_free(workspace->prelude_storage);
    ds4_gpu_tensor_free(workspace->current_blocks_storage);
    ds4_gpu_tensor_free(workspace->previous_blocks_storage);
    ds4_gpu_tensor_free(workspace->ffn_input_storage);
    ds4_gpu_tensor_free(workspace->moe_input_storage);
    ds4_gpu_tensor_free(workspace->moe_output_storage);
    for (uint32_t input = 0; input < 3u; input++)
        ds4_gpu_tensor_free(workspace->attn_norm_storage[input]);
    ds4_gpu_tensor_free(workspace->query_storage);
    ds4_gpu_tensor_free(workspace->key_storage);
    ds4_gpu_tensor_free(workspace->value_storage);
    ds4_gpu_tensor_free(workspace->relative_storage);
    ds4_gpu_tensor_free(workspace->dynamic_k_storage);
    ds4_gpu_tensor_free(workspace->dynamic_v_storage);
    ds4_gpu_tensor_free(workspace->sink_logits_storage);
    ds4_gpu_tensor_free(workspace->attention_gate_storage);
    ds4_gpu_tensor_free(workspace->attention_gated_storage);
    ds4_gpu_tensor_free(workspace->attention_output_storage);
    memset(workspace, 0, sizeof(*workspace));
}

static ds4_gpu_tensor *take_view(ds4_gpu_tensor *base, uint64_t base_bytes,
                                 uint64_t *cursor, uint64_t bytes) {
    if (!base || !cursor || bytes == 0 ||
        *cursor > UINT64_MAX - (DS4_INKLING_GPU_WEIGHT_ALIGNMENT - 1u))
        return NULL;
    const uint64_t aligned = (*cursor + DS4_INKLING_GPU_WEIGHT_ALIGNMENT - 1u) &
        ~(uint64_t)(DS4_INKLING_GPU_WEIGHT_ALIGNMENT - 1u);
    if (aligned > base_bytes || bytes > base_bytes - aligned) return NULL;
    ds4_gpu_tensor *view = ds4_gpu_tensor_view(base, aligned, bytes);
    if (!view || !checked_add_u64(aligned, bytes, cursor)) {
        ds4_gpu_tensor_free(view);
        return NULL;
    }
    return view;
}

static ds4_gpu_tensor *take_float_view(ds4_gpu_tensor *base,
                                       uint64_t base_bytes,
                                       uint64_t *cursor,
                                       uint64_t elements) {
    uint64_t bytes = 0;
    if (!checked_mul_u64(elements, sizeof(float), &bytes)) return NULL;
    return take_view(base, base_bytes, cursor, bytes);
}

static ds4_gpu_tensor *take_parameter_view(ds4_inkling_gpu_context *context,
                                           uint64_t *cursor,
                                           uint64_t elements) {
    return take_float_view(context->weight_storage,
                            ds4_gpu_tensor_bytes(context->weight_storage),
                            cursor, elements);
}

static bool allocate_parameter_views(ds4_inkling_gpu_context *context) {
    bool sliding[DS4_INKLING_PHYSICAL_LAYERS] = { false };
    bool found[DS4_INKLING_PHYSICAL_LAYERS] = { false };
    for (uint32_t i = 0; i < context->plan.visit_count; i++) {
        const ds4_inkling_visit *visit = &context->plan.visits[i];
        sliding[visit->physical_layer] = visit->is_sliding_window != 0;
        found[visit->physical_layer] = true;
    }
    uint64_t cursor = 0;
    context->parameters.output_res_query = take_parameter_view(
        context, &cursor, DS4_INKLING_GPU_HIDDEN);
    if (!context->parameters.output_res_query) return false;
    for (uint32_t layer = 0; layer < DS4_INKLING_PHYSICAL_LAYERS; layer++) {
        ds4_inkling_gpu_layer_parameters *p = &context->parameters.layer[layer];
        if (!found[layer]) return false;
        p->attn_q_clip = take_parameter_view(context, &cursor, 128u * 24u);
        p->attn_rel_bias = take_parameter_view(context, &cursor, 16u * 1024u);
        p->attn_rel_clip = take_parameter_view(context, &cursor, 16u * 24u);
        p->attn_k_conv = take_parameter_view(context, &cursor,
            4u * DS4_INKLING_KV_WIDTH);
        p->attn_v_conv = take_parameter_view(context, &cursor,
            4u * DS4_INKLING_KV_WIDTH);
        p->attn_post_conv = take_parameter_view(context, &cursor,
            4u * DS4_INKLING_GPU_HIDDEN);
        p->ffn_post_conv = take_parameter_view(context, &cursor,
            4u * DS4_INKLING_GPU_HIDDEN);
        if (layer != 0) {
            p->attn_res_query = take_parameter_view(context, &cursor,
                DS4_INKLING_GPU_HIDDEN * (sliding[layer] ? 1u : 3u));
        }
        p->ffn_res_query = take_parameter_view(context, &cursor,
            DS4_INKLING_GPU_HIDDEN);
        p->router_bias = take_parameter_view(context, &cursor,
            DS4_INKLING_GPU_EXPERTS);
        if (layer >= DS4_INKLING_LOOP_START &&
            layer < DS4_INKLING_LOOP_END) {
            p->loop_msg_query = take_parameter_view(context, &cursor,
                4u * DS4_INKLING_GPU_HIDDEN);
        }
        if (!p->attn_q_clip || !p->attn_rel_bias || !p->attn_rel_clip ||
            !p->attn_k_conv || !p->attn_v_conv || !p->attn_post_conv ||
            !p->ffn_post_conv || (layer != 0 && !p->attn_res_query) ||
            !p->ffn_res_query || !p->router_bias ||
            ((layer >= DS4_INKLING_LOOP_START &&
              layer < DS4_INKLING_LOOP_END) && !p->loop_msg_query)) return false;
    }
    return cursor <= ds4_gpu_tensor_bytes(context->weight_storage);
}

static bool allocate_compute_workspaces(ds4_inkling_gpu_context *context) {
    const uint64_t scratch_bytes = ds4_gpu_tensor_bytes(context->scratch);
    uint64_t cursor = 0;
    ds4_inkling_gpu_control_workspace *control = &context->control;
    control->normalized = take_float_view(context->scratch, scratch_bytes,
        &cursor, 2u * DS4_INKLING_GPU_HIDDEN);
    control->bottleneck = take_float_view(context->scratch, scratch_bytes,
        &cursor, 64u);
    control->gain = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_HIDDEN);
    control->forget = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_HIDDEN);

    ds4_inkling_gpu_moe_workspace *moe = &context->moe;
    moe->logits = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_EXPERTS);
    moe->ids = take_view(context->scratch, scratch_bytes, &cursor,
        DS4_INKLING_GPU_MOE_MAX_TOKENS * 8u * sizeof(int32_t));
    moe->weights = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * 8u);
    moe->gate_up = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * 8u *
            2u * DS4_INKLING_GPU_EXPERT_DIM);
    moe->activation = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * 8u *
            DS4_INKLING_GPU_EXPERT_DIM);
    moe->expert_outputs = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * 8u *
            DS4_INKLING_GPU_HIDDEN);
    moe->zero_bias = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_EXPERTS);

    ds4_inkling_gpu_batch_workspace *batch = &context->batch;
    batch->current_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_HIDDEN);
    batch->next_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_HIDDEN);
    batch->prelude_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_HIDDEN);
    batch->current_blocks_storage = take_float_view(context->scratch,
        scratch_bytes, &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS *
            4u * DS4_INKLING_GPU_HIDDEN);
    batch->previous_blocks_storage = take_float_view(context->scratch,
        scratch_bytes, &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS *
            2u * DS4_INKLING_GPU_HIDDEN);
    batch->ffn_input_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_HIDDEN);
    batch->moe_input_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_HIDDEN);
    batch->moe_output_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_HIDDEN);
    for (uint32_t input = 0; input < 3u; input++) {
        batch->attn_norm_storage[input] = take_float_view(context->scratch,
            scratch_bytes, &cursor,
            DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_HIDDEN);
    }
    batch->query_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_QUERY_WIDTH);
    batch->key_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_KV_WIDTH);
    batch->value_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_KV_WIDTH);
    batch->relative_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS *
            DS4_INKLING_GPU_RELATIVE_WIDTH);
    batch->dynamic_k_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * 32u);
    batch->dynamic_v_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * 32u);
    batch->sink_logits_storage = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_MOE_MAX_TOKENS * 24u);
    batch->attention_gate_storage = take_float_view(context->scratch,
        scratch_bytes, &cursor,
        DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_QUERY_WIDTH);
    batch->attention_gated_storage = take_float_view(context->scratch,
        scratch_bytes, &cursor,
        DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_QUERY_WIDTH);
    batch->attention_output_storage = take_float_view(context->scratch,
        scratch_bytes, &cursor,
        DS4_INKLING_GPU_MOE_MAX_TOKENS * DS4_INKLING_GPU_HIDDEN);

    ds4_inkling_gpu_layer_workspace *layer = &context->layer;
    layer->current = context->activation;
    layer->next = take_float_view(context->scratch, scratch_bytes, &cursor,
        DS4_INKLING_GPU_HIDDEN);
    layer->mix_states = take_float_view(context->scratch, scratch_bytes,
        &cursor, 6u * DS4_INKLING_GPU_HIDDEN);
    layer->mix_outputs = take_float_view(context->scratch, scratch_bytes,
        &cursor, 4u * DS4_INKLING_GPU_HIDDEN);
    layer->controlled_inputs = take_float_view(context->scratch, scratch_bytes,
        &cursor, 3u * DS4_INKLING_GPU_HIDDEN);
    layer->query = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_QUERY_WIDTH);
    layer->prepared_query = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_QUERY_WIDTH);
    layer->key = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_KV_WIDTH);
    layer->prepared_key = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_KV_WIDTH);
    layer->value = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_KV_WIDTH);
    layer->prepared_value = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_KV_WIDTH);
    layer->relative = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_RELATIVE_WIDTH);
    layer->prepared_relative = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_RELATIVE_WIDTH);
    layer->sink_logits = take_float_view(context->scratch, scratch_bytes,
        &cursor, 24u);
    layer->relative_bias = take_float_view(context->scratch, scratch_bytes,
        &cursor, 24u * 1024u);
    layer->relative_distances = take_view(context->scratch, scratch_bytes,
        &cursor, 1024u * sizeof(int64_t));
    layer->attention_gate = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_QUERY_WIDTH);
    layer->attention_context = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_QUERY_WIDTH);
    layer->attention_gated = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_QUERY_WIDTH);
    layer->attention_output = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_HIDDEN);
    layer->ffn_input = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_HIDDEN);
    layer->ffn_mixed = take_float_view(context->scratch, scratch_bytes,
        &cursor, DS4_INKLING_GPU_HIDDEN);
    layer->messages = take_float_view(context->scratch, scratch_bytes,
        &cursor, 4u * DS4_INKLING_GPU_HIDDEN);
    layer->dynamic_k = take_float_view(context->scratch, scratch_bytes,
        &cursor, 32u);
    layer->dynamic_v = take_float_view(context->scratch, scratch_bytes,
        &cursor, 32u);

    const uint64_t hidden_bytes =
        (uint64_t)DS4_INKLING_GPU_HIDDEN * sizeof(float);
    const uint64_t query_bytes =
        (uint64_t)DS4_INKLING_GPU_QUERY_WIDTH * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)DS4_INKLING_KV_WIDTH * sizeof(float);
    const uint64_t relative_bytes =
        (uint64_t)DS4_INKLING_GPU_RELATIVE_WIDTH * sizeof(float);
    const uint64_t dynamic_bytes = 32u * sizeof(float);
    const uint64_t sink_bytes = 24u * sizeof(float);
    for (uint32_t row = 0; row < DS4_INKLING_GPU_MOE_MAX_TOKENS; row++) {
        const uint64_t row_offset = (uint64_t)row * hidden_bytes;
        const uint64_t block_offset = (uint64_t)row * 4u * hidden_bytes;
        const uint64_t previous_offset = (uint64_t)row * 2u * hidden_bytes;
        batch->current[row] = ds4_gpu_tensor_view(batch->current_storage,
                                                   row_offset, hidden_bytes);
        batch->next[row] = ds4_gpu_tensor_view(batch->next_storage,
                                                row_offset, hidden_bytes);
        batch->prelude[row] = ds4_gpu_tensor_view(batch->prelude_storage,
                                                   row_offset, hidden_bytes);
        batch->current_blocks[row] = ds4_gpu_tensor_view(
            batch->current_blocks_storage, block_offset, 4u * hidden_bytes);
        batch->previous_blocks[row] = ds4_gpu_tensor_view(
            batch->previous_blocks_storage, previous_offset, 2u * hidden_bytes);
        batch->ffn_input[row] = ds4_gpu_tensor_view(batch->ffn_input_storage,
                                                    row_offset, hidden_bytes);
        batch->moe_input[row] = ds4_gpu_tensor_view(batch->moe_input_storage,
                                                    row_offset, hidden_bytes);
        batch->moe_output[row] = ds4_gpu_tensor_view(batch->moe_output_storage,
                                                     row_offset, hidden_bytes);
        for (uint32_t input = 0; input < 3u; input++) {
            batch->attn_norm[input][row] = ds4_gpu_tensor_view(
                batch->attn_norm_storage[input], row_offset, hidden_bytes);
        }
        batch->query[row] = ds4_gpu_tensor_view(batch->query_storage,
                                                (uint64_t)row * query_bytes,
                                                query_bytes);
        batch->key[row] = ds4_gpu_tensor_view(batch->key_storage,
                                              (uint64_t)row * kv_bytes,
                                              kv_bytes);
        batch->value[row] = ds4_gpu_tensor_view(batch->value_storage,
                                                (uint64_t)row * kv_bytes,
                                                kv_bytes);
        batch->relative[row] = ds4_gpu_tensor_view(batch->relative_storage,
            (uint64_t)row * relative_bytes, relative_bytes);
        batch->dynamic_k[row] = ds4_gpu_tensor_view(batch->dynamic_k_storage,
            (uint64_t)row * dynamic_bytes, dynamic_bytes);
        batch->dynamic_v[row] = ds4_gpu_tensor_view(batch->dynamic_v_storage,
            (uint64_t)row * dynamic_bytes, dynamic_bytes);
        batch->sink_logits[row] = ds4_gpu_tensor_view(batch->sink_logits_storage,
            (uint64_t)row * sink_bytes, sink_bytes);
        batch->attention_gate[row] = ds4_gpu_tensor_view(
            batch->attention_gate_storage, (uint64_t)row * query_bytes,
            query_bytes);
        batch->attention_gated[row] = ds4_gpu_tensor_view(
            batch->attention_gated_storage, (uint64_t)row * query_bytes,
            query_bytes);
        batch->attention_output[row] = ds4_gpu_tensor_view(
            batch->attention_output_storage, row_offset, hidden_bytes);
    }
    for (uint32_t i = 0; i < 4u; i++) {
        layer->mix_output_rows[i] = ds4_gpu_tensor_view(
            layer->mix_outputs, (uint64_t)i * hidden_bytes, hidden_bytes);
        layer->message_rows[i] = ds4_gpu_tensor_view(
            layer->messages, (uint64_t)i * hidden_bytes, hidden_bytes);
    }
    for (uint32_t i = 0; i < 3u; i++) {
        layer->controlled_input_rows[i] = ds4_gpu_tensor_view(
            layer->controlled_inputs, (uint64_t)i * hidden_bytes, hidden_bytes);
    }
    layer->previous_loop_states = ds4_gpu_tensor_view(
        context->previous_blocks, 0, 2u * hidden_bytes);

    if (!control->normalized || !control->bottleneck || !control->gain ||
        !control->forget || !moe->logits || !moe->ids || !moe->weights ||
        !moe->gate_up || !moe->activation || !moe->expert_outputs ||
        !moe->zero_bias || !batch->current_storage || !batch->next_storage ||
        !batch->prelude_storage || !batch->current_blocks_storage ||
        !batch->previous_blocks_storage || !batch->ffn_input_storage ||
        !batch->moe_input_storage || !batch->moe_output_storage ||
        !batch->query_storage || !batch->key_storage || !batch->value_storage ||
        !batch->relative_storage || !batch->dynamic_k_storage ||
        !batch->dynamic_v_storage || !batch->sink_logits_storage ||
        !batch->attention_gate_storage || !batch->attention_gated_storage ||
        !batch->attention_output_storage ||
        !layer->next || !layer->mix_states ||
        !layer->mix_outputs || !layer->controlled_inputs || !layer->query ||
        !layer->prepared_query || !layer->key || !layer->prepared_key ||
        !layer->value || !layer->prepared_value || !layer->relative ||
        !layer->prepared_relative ||
        !layer->sink_logits || !layer->relative_bias ||
        !layer->relative_distances || !layer->attention_gate ||
        !layer->attention_context || !layer->attention_gated ||
        !layer->attention_output || !layer->ffn_input || !layer->ffn_mixed ||
        !layer->messages || !layer->dynamic_k || !layer->dynamic_v ||
        !layer->previous_loop_states ||
        cursor > scratch_bytes) return false;
    for (uint32_t i = 0; i < 4u; i++) {
        if (!layer->mix_output_rows[i] || !layer->message_rows[i]) return false;
    }
    for (uint32_t i = 0; i < 3u; i++) {
        if (!layer->controlled_input_rows[i]) return false;
    }
    for (uint32_t row = 0; row < DS4_INKLING_GPU_MOE_MAX_TOKENS; row++) {
        if (!batch->current[row] || !batch->next[row] || !batch->prelude[row] ||
            !batch->current_blocks[row] || !batch->previous_blocks[row] ||
            !batch->ffn_input[row] || !batch->moe_input[row] ||
            !batch->moe_output[row]) return false;
        for (uint32_t input = 0; input < 3u; input++) {
            if (!batch->attn_norm_storage[input] ||
                !batch->attn_norm[input][row]) return false;
        }
        if (!batch->query[row] || !batch->key[row] || !batch->value[row] ||
            !batch->relative[row] || !batch->dynamic_k[row] ||
            !batch->dynamic_v[row] || !batch->sink_logits[row] ||
            !batch->attention_gate[row] || !batch->attention_gated[row] ||
            !batch->attention_output[row]) return false;
    }

    int64_t distances[1024];
    for (uint32_t i = 0; i < 1024u; i++) distances[i] = (int64_t)i;
    return ds4_gpu_tensor_fill_f32(moe->zero_bias, 0.0f,
                                   DS4_INKLING_GPU_EXPERTS) &&
        ds4_gpu_tensor_write(layer->relative_distances, 0, distances,
                             sizeof(distances));
}

void ds4_inkling_gpu_context_free(ds4_inkling_gpu_context *context) {
    if (!context) return;
    for (uint32_t slot = 0; slot < DS4_INKLING_CACHE_SLOTS; slot++)
        free_slot_views(&context->slots[slot]);
    free_control_workspace(&context->control);
    free_moe_workspace(&context->moe);
    free_batch_workspace(&context->batch);
    free_layer_workspace(&context->layer);
    ds4_gpu_tensor_free(context->parameters.output_res_query);
    for (uint32_t layer = 0; layer < DS4_INKLING_PHYSICAL_LAYERS; layer++)
        free_layer_parameters(&context->parameters.layer[layer]);
    ds4_gpu_tensor_free(context->weight_storage);
    ds4_gpu_tensor_free(context->embedding);
    ds4_gpu_tensor_free(context->activation);
    ds4_gpu_tensor_free(context->scratch);
    ds4_gpu_tensor_free(context->key_storage);
    ds4_gpu_tensor_free(context->value_storage);
    ds4_gpu_tensor_free(context->conv_storage);
    ds4_gpu_tensor_free(context->work_storage);
    ds4_gpu_tensor_free(context->current_blocks);
    ds4_gpu_tensor_free(context->previous_blocks);
    ds4_gpu_tensor_free(context->prelude_output);
    ds4_gpu_tensor_free(context->input_token);
    ds4_gpu_tensor_free(context->logits);
    memset(context, 0, sizeof(*context));
}

bool ds4_inkling_gpu_context_init(
        ds4_inkling_gpu_context *context,
        const ds4_inkling_plan *plan,
        uint32_t context_capacity) {
    ds4_inkling_gpu_layout layout;
    if (!context || !context_layout(plan, context_capacity, &layout)) return false;
    memset(context, 0, sizeof(*context));

    const uint64_t kv_storage_bytes = layout.kv_storage_bytes;
    const uint64_t conv_storage_bytes = layout.conv_storage_bytes;
    uint64_t conv_slot_bytes[DS4_INKLING_CONV_GROUP_COUNT];
    for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
        if (!checked_mul_u64(DS4_INKLING_CONV_HISTORY,
                             g_conv_width[group], &conv_slot_bytes[group]) ||
            !checked_mul_u64(conv_slot_bytes[group], sizeof(float),
                             &conv_slot_bytes[group])) goto fail;
    }

    context->plan = *plan;
    context->weight_owner_count = DS4_INKLING_PHYSICAL_LAYERS;
    context->context_capacity = context_capacity;
    context->allocated_bytes = layout.total_bytes;
    for (uint32_t layer = 0; layer < DS4_INKLING_PHYSICAL_LAYERS; layer++)
        context->weight_owner[layer] = layer;

    context->key_storage = ds4_gpu_tensor_alloc(kv_storage_bytes);
    context->value_storage = ds4_gpu_tensor_alloc(kv_storage_bytes);
    context->conv_storage = ds4_gpu_tensor_alloc(conv_storage_bytes);
    context->work_storage = ds4_gpu_tensor_alloc(DS4_INKLING_GPU_WORK_BYTES);
    context->current_blocks = ds4_gpu_tensor_alloc(layout.residual_bank_bytes);
    context->previous_blocks = ds4_gpu_tensor_alloc(layout.residual_bank_bytes);
    context->prelude_output = ds4_gpu_tensor_alloc(layout.prelude_bytes);
    context->input_token = ds4_gpu_tensor_alloc(sizeof(int32_t));
    context->logits = ds4_gpu_tensor_alloc(layout.logits_bytes);
    context->weight_storage = ds4_gpu_tensor_alloc(layout.weight_storage_bytes);
    if (!context->key_storage || !context->value_storage ||
        !context->conv_storage || !context->work_storage ||
        !context->current_blocks || !context->previous_blocks ||
        !context->prelude_output || !context->input_token ||
        !context->logits || !context->weight_storage) goto fail;

    const uint64_t embedding_bytes =
        (uint64_t)DS4_INKLING_GPU_HIDDEN * sizeof(float);
    context->embedding = ds4_gpu_tensor_view(context->work_storage, 0,
                                              embedding_bytes);
    context->activation = ds4_gpu_tensor_view(context->work_storage,
                                               embedding_bytes,
                                               embedding_bytes);
    context->scratch = ds4_gpu_tensor_view(
        context->work_storage, 2u * embedding_bytes,
        DS4_INKLING_GPU_WORK_BYTES - 2u * embedding_bytes);
    if (!context->embedding || !context->activation || !context->scratch ||
        !allocate_parameter_views(context) ||
        !allocate_compute_workspaces(context))
        goto fail;

    uint64_t kv_offset = 0;
    uint64_t conv_offset = 0;
    for (uint32_t i = 0; i < plan->visit_count; i++) {
        const ds4_inkling_visit *visit = &plan->visits[i];
        ds4_inkling_gpu_slot *slot = &context->slots[visit->cache_slot];
        slot->capacity = slot_capacity(visit, context_capacity);
        uint64_t kv_bytes = 0;
        if (!checked_mul_u64(slot->capacity, DS4_INKLING_KV_WIDTH,
                             &kv_bytes) ||
            !checked_mul_u64(kv_bytes, sizeof(uint16_t), &kv_bytes)) goto fail;
        slot->key_offset = kv_offset;
        slot->value_offset = kv_offset;
        slot->key = ds4_gpu_tensor_view(context->key_storage,
                                        slot->key_offset, kv_bytes);
        slot->value = ds4_gpu_tensor_view(context->value_storage,
                                          slot->value_offset, kv_bytes);
        if (!slot->key || !slot->value) goto fail;

        uint64_t group_offset = conv_offset;
        for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
            slot->conv_offset[group] = group_offset;
            slot->conv[group] = ds4_gpu_tensor_view(
                context->conv_storage, group_offset, conv_slot_bytes[group]);
            if (!slot->conv[group]) goto fail;
            group_offset += conv_slot_bytes[group];
        }
        conv_offset = group_offset;
        kv_offset += kv_bytes;
    }
    return true;

fail:
    ds4_inkling_gpu_context_free(context);
    return false;
}

void ds4_inkling_gpu_context_reset(ds4_inkling_gpu_context *context) {
    if (!context) return;
    context->position = 0;
    for (uint32_t i = 0; i < DS4_INKLING_CACHE_SLOTS; i++) {
        ds4_inkling_gpu_slot *slot = &context->slots[i];
        slot->length = 0;
        slot->next_position = 0;
        memset(slot->conv_length, 0, sizeof(slot->conv_length));
    }
}
