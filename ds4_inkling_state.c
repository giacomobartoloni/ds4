#include "ds4_inkling_state.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool checked_bytes(uint64_t count, uint64_t element_size, size_t *bytes) {
    if (element_size != 0 && count > SIZE_MAX / element_size) return false;
    *bytes = (size_t)(count * element_size);
    return true;
}

static bool valid_group(ds4_inkling_conv_group group) {
    return (uint32_t)group < DS4_INKLING_CONV_GROUP_COUNT;
}

static void transaction_free(ds4_inkling_state_slot *slot) {
    free(slot->transaction_indices);
    free(slot->transaction_key_backup);
    free(slot->transaction_value_backup);
    for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
        free(slot->transaction_conv_backup[group]);
        slot->transaction_conv_backup[group] = NULL;
    }
    slot->transaction_indices = NULL;
    slot->transaction_key_backup = NULL;
    slot->transaction_value_backup = NULL;
    slot->transaction_active = false;
    slot->transaction_limit = 0;
    slot->transaction_count = 0;
}

static void slot_free(ds4_inkling_state_slot *slot) {
    if (!slot) return;
    transaction_free(slot);
    free(slot->key);
    free(slot->value);
    for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
        free(slot->conv_history[group]);
    }
    memset(slot, 0, sizeof(*slot));
}

bool ds4_inkling_state_session_init(
    ds4_inkling_state_session *session,
    const ds4_inkling_plan *plan,
    const ds4_inkling_state_config *config) {
    if (!session) return false;
    memset(session, 0, sizeof(*session));
    if (!plan || !config ||
        plan->physical_layer_count != DS4_INKLING_PHYSICAL_LAYERS ||
        plan->visit_count != DS4_INKLING_VISITS ||
        plan->cache_slot_count != DS4_INKLING_CACHE_SLOTS ||
        config->context_capacity == 0 || config->sliding_window == 0 ||
        config->sliding_window > config->context_capacity || config->kv_width == 0) {
        return false;
    }
    for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
        if (config->conv_width[group] == 0) return false;
    }

    uint8_t seen[DS4_INKLING_CACHE_SLOTS] = { 0 };
    uint8_t layer_seen[DS4_INKLING_PHYSICAL_LAYERS] = { 0 };
    uint8_t layer_is_sliding[DS4_INKLING_PHYSICAL_LAYERS] = { 0 };
    uint8_t layer_visit_count[DS4_INKLING_PHYSICAL_LAYERS] = { 0 };
    uint8_t is_sliding[DS4_INKLING_CACHE_SLOTS] = { 0 };
    for (uint32_t visit = 0; visit < plan->visit_count; visit++) {
        const ds4_inkling_visit *entry = &plan->visits[visit];
        if (entry->physical_layer >= DS4_INKLING_PHYSICAL_LAYERS ||
            entry->cache_slot >= DS4_INKLING_CACHE_SLOTS ||
            entry->is_sliding_window > 1 || seen[entry->cache_slot]) {
            return false;
        }
        if (layer_seen[entry->physical_layer] &&
            layer_is_sliding[entry->physical_layer] != entry->is_sliding_window) {
            return false;
        }
        layer_seen[entry->physical_layer] = 1;
        layer_is_sliding[entry->physical_layer] = entry->is_sliding_window;
        layer_visit_count[entry->physical_layer]++;
        seen[entry->cache_slot] = 1;
        is_sliding[entry->cache_slot] = entry->is_sliding_window;
    }
    for (uint32_t layer = 0; layer < DS4_INKLING_PHYSICAL_LAYERS; layer++) {
        const uint8_t expected_visits =
            layer >= DS4_INKLING_LOOP_START && layer < DS4_INKLING_LOOP_END
                ? DS4_INKLING_LOOP_PASSES : 1u;
        if (!layer_seen[layer] || layer_visit_count[layer] != expected_visits) {
            return false;
        }
    }
    for (uint32_t index = 0; index < DS4_INKLING_CACHE_SLOTS; index++) {
        if (!seen[index]) return false;
    }

    session->slot_count = DS4_INKLING_CACHE_SLOTS;
    session->kv_width = config->kv_width;
    for (uint32_t index = 0; index < session->slot_count; index++) {
        ds4_inkling_state_slot *slot = &session->slots[index];
        slot->is_sliding = is_sliding[index] != 0;
        slot->capacity = slot->is_sliding ? config->sliding_window :
                                            config->context_capacity;
        slot->kv_width = config->kv_width;
        const uint64_t kv_elements = (uint64_t)slot->capacity * slot->kv_width;
        size_t kv_bytes;
        if (!checked_bytes(kv_elements, sizeof(uint16_t), &kv_bytes)) goto fail;
        slot->key = (uint16_t *)calloc(1, kv_bytes);
        slot->value = (uint16_t *)calloc(1, kv_bytes);
        if (!slot->key || !slot->value) goto fail;
        for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
            slot->conv_width[group] = config->conv_width[group];
            size_t history_bytes;
            if (!checked_bytes((uint64_t)DS4_INKLING_CONV_HISTORY *
                               config->conv_width[group], sizeof(float),
                               &history_bytes)) goto fail;
            slot->conv_history[group] = (float *)calloc(1, history_bytes);
            if (!slot->conv_history[group]) goto fail;
        }
    }
    return true;

fail:
    ds4_inkling_state_session_free(session);
    return false;
}

void ds4_inkling_state_session_free(ds4_inkling_state_session *session) {
    if (!session) return;
    for (uint32_t index = 0; index < DS4_INKLING_CACHE_SLOTS; index++) {
        slot_free(&session->slots[index]);
    }
    memset(session, 0, sizeof(*session));
}

uint64_t ds4_inkling_state_session_bytes(const ds4_inkling_state_session *session) {
    if (!session) return 0;
    uint64_t total = 0;
    for (uint32_t index = 0; index < session->slot_count; index++) {
        const ds4_inkling_state_slot *slot = &session->slots[index];
        total += (uint64_t)slot->capacity * slot->kv_width * 2u * sizeof(uint16_t);
        for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
            total += (uint64_t)DS4_INKLING_CONV_HISTORY *
                     slot->conv_width[group] * sizeof(float);
        }
    }
    return total;
}

uint64_t ds4_inkling_state_oldest_position(const ds4_inkling_state_slot *slot) {
    if (!slot || slot->count > slot->next_position) return 0;
    return slot->next_position - slot->count;
}

static bool transaction_backup_kv(
    ds4_inkling_state_slot *slot,
    uint32_t ring_index) {
    if (!slot->transaction_active) return true;
    const uint32_t item = slot->transaction_count;
    if (item >= slot->transaction_limit) return false;
    slot->transaction_indices[item] = ring_index;
    const uint64_t offset = (uint64_t)ring_index * slot->kv_width;
    const uint64_t backup_offset = (uint64_t)item * slot->kv_width;
    memcpy(slot->transaction_key_backup + backup_offset,
           slot->key + offset, (size_t)slot->kv_width * sizeof(uint16_t));
    memcpy(slot->transaction_value_backup + backup_offset,
           slot->value + offset, (size_t)slot->kv_width * sizeof(uint16_t));
    slot->transaction_count++;
    return true;
}

bool ds4_inkling_state_append_kv_fp16(
    ds4_inkling_state_slot *slot,
    uint64_t position,
    const uint16_t *key,
    const uint16_t *value) {
    if (!slot || !slot->key || !slot->value || !key || !value ||
        position != slot->next_position || slot->next_position == UINT64_MAX) {
        return false;
    }
    if (slot->transaction_active && slot->transaction_count >= slot->transaction_limit) {
        return false;
    }
    const uint32_t ring_index = (uint32_t)(position % slot->capacity);
    if (!transaction_backup_kv(slot, ring_index)) return false;
    const uint64_t offset = (uint64_t)ring_index * slot->kv_width;
    memcpy(slot->key + offset, key, (size_t)slot->kv_width * sizeof(uint16_t));
    memcpy(slot->value + offset, value, (size_t)slot->kv_width * sizeof(uint16_t));
    slot->next_position++;
    if (slot->count < slot->capacity) slot->count++;
    return true;
}

bool ds4_inkling_state_read_kv_fp16(
    const ds4_inkling_state_slot *slot,
    uint64_t position,
    uint16_t *key,
    uint16_t *value) {
    if (!slot || !key || !value || !slot->capacity || !slot->kv_width ||
        position < ds4_inkling_state_oldest_position(slot) ||
        position >= slot->next_position) return false;
    const uint32_t index = (uint32_t)(position % slot->capacity);
    const uint64_t offset = (uint64_t)index * slot->kv_width;
    memcpy(key, slot->key + offset, (size_t)slot->kv_width * sizeof(uint16_t));
    memcpy(value, slot->value + offset, (size_t)slot->kv_width * sizeof(uint16_t));
    return true;
}

bool ds4_inkling_state_begin(ds4_inkling_state_slot *slot, uint32_t max_tokens) {
    if (!slot || !slot->key || !slot->value || slot->transaction_active || max_tokens == 0) {
        return false;
    }
    size_t index_bytes, kv_backup_bytes;
    if (!checked_bytes(max_tokens, sizeof(uint32_t), &index_bytes) ||
        !checked_bytes((uint64_t)max_tokens * slot->kv_width,
                       sizeof(uint16_t), &kv_backup_bytes)) return false;
    slot->transaction_indices = (uint32_t *)malloc(index_bytes);
    slot->transaction_key_backup = (uint16_t *)malloc(kv_backup_bytes);
    slot->transaction_value_backup = (uint16_t *)malloc(kv_backup_bytes);
    if (!slot->transaction_indices || !slot->transaction_key_backup ||
        !slot->transaction_value_backup) {
        transaction_free(slot);
        return false;
    }
    for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
        size_t bytes;
        if (!checked_bytes((uint64_t)DS4_INKLING_CONV_HISTORY *
                           slot->conv_width[group], sizeof(float), &bytes)) {
            transaction_free(slot);
            return false;
        }
        slot->transaction_conv_backup[group] = (float *)malloc(bytes);
        if (!slot->transaction_conv_backup[group]) {
            transaction_free(slot);
            return false;
        }
        memcpy(slot->transaction_conv_backup[group], slot->conv_history[group], bytes);
        slot->transaction_conv_count[group] = slot->conv_count[group];
        slot->transaction_conv_next[group] = slot->conv_next[group];
    }
    slot->transaction_count_snapshot = slot->count;
    slot->transaction_position_snapshot = slot->next_position;
    slot->transaction_limit = max_tokens;
    slot->transaction_count = 0;
    slot->transaction_active = true;
    return true;
}

bool ds4_inkling_state_commit(ds4_inkling_state_slot *slot) {
    if (!slot || !slot->transaction_active) return false;
    transaction_free(slot);
    return true;
}

bool ds4_inkling_state_rollback(ds4_inkling_state_slot *slot) {
    if (!slot || !slot->transaction_active) return false;
    while (slot->transaction_count > 0) {
        const uint32_t item = --slot->transaction_count;
        const uint32_t index = slot->transaction_indices[item];
        const uint64_t offset = (uint64_t)index * slot->kv_width;
        const uint64_t backup_offset = (uint64_t)item * slot->kv_width;
        memcpy(slot->key + offset, slot->transaction_key_backup + backup_offset,
               (size_t)slot->kv_width * sizeof(uint16_t));
        memcpy(slot->value + offset, slot->transaction_value_backup + backup_offset,
               (size_t)slot->kv_width * sizeof(uint16_t));
    }
    slot->count = slot->transaction_count_snapshot;
    slot->next_position = slot->transaction_position_snapshot;
    for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
        memcpy(slot->conv_history[group], slot->transaction_conv_backup[group],
               (size_t)DS4_INKLING_CONV_HISTORY * slot->conv_width[group] * sizeof(float));
        slot->conv_count[group] = slot->transaction_conv_count[group];
        slot->conv_next[group] = slot->transaction_conv_next[group];
    }
    transaction_free(slot);
    return true;
}

bool ds4_inkling_state_reset(ds4_inkling_state_slot *slot) {
    if (!slot || slot->transaction_active) return false;
    slot->count = 0;
    slot->next_position = 0;
    for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
        memset(slot->conv_history[group], 0,
               (size_t)DS4_INKLING_CONV_HISTORY * slot->conv_width[group] * sizeof(float));
        slot->conv_count[group] = 0;
        slot->conv_next[group] = 0;
    }
    return true;
}

bool ds4_inkling_state_clone(
    ds4_inkling_state_slot *dst,
    const ds4_inkling_state_slot *src) {
    if (!dst || !src || dst == src || !dst->key || !src->key ||
        dst->transaction_active || src->transaction_active ||
        dst->capacity != src->capacity || dst->kv_width != src->kv_width ||
        dst->is_sliding != src->is_sliding) {
        return false;
    }
    for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
        if (dst->conv_width[group] != src->conv_width[group]) return false;
    }
    const uint64_t kv_bytes = (uint64_t)src->capacity * src->kv_width * sizeof(uint16_t);
    memcpy(dst->key, src->key, (size_t)kv_bytes);
    memcpy(dst->value, src->value, (size_t)kv_bytes);
    for (uint32_t group = 0; group < DS4_INKLING_CONV_GROUP_COUNT; group++) {
        memcpy(dst->conv_history[group], src->conv_history[group],
               (size_t)DS4_INKLING_CONV_HISTORY * src->conv_width[group] * sizeof(float));
        dst->conv_count[group] = src->conv_count[group];
        dst->conv_next[group] = src->conv_next[group];
    }
    dst->next_position = src->next_position;
    dst->count = src->count;
    return true;
}

bool ds4_inkling_state_conv_push(
    ds4_inkling_state_slot *slot,
    ds4_inkling_conv_group group,
    const float *input,
    uint32_t width) {
    if (!slot || !valid_group(group) || !input ||
        slot->conv_width[group] != width || width == 0) return false;
    const uint32_t index = slot->conv_next[group];
    memcpy(slot->conv_history[group] + (uint64_t)index * width,
           input, (size_t)width * sizeof(float));
    slot->conv_next[group] = (index + 1u) % DS4_INKLING_CONV_HISTORY;
    if (slot->conv_count[group] < DS4_INKLING_CONV_HISTORY) slot->conv_count[group]++;
    return true;
}

bool ds4_inkling_state_conv_read_lag(
    const ds4_inkling_state_slot *slot,
    ds4_inkling_conv_group group,
    uint32_t lag,
    float *output,
    uint32_t width) {
    if (!slot || !valid_group(group) || !output ||
        slot->conv_width[group] != width || lag == 0 ||
        lag > DS4_INKLING_CONV_HISTORY) return false;
    if (lag > slot->conv_count[group]) return false;
    const uint32_t index = (slot->conv_next[group] +
                            DS4_INKLING_CONV_HISTORY - lag) %
                           DS4_INKLING_CONV_HISTORY;
    memcpy(output, slot->conv_history[group] + (uint64_t)index * width,
           (size_t)width * sizeof(float));
    return true;
}

bool ds4_inkling_state_conv_export_history(
    const ds4_inkling_state_slot *slot,
    ds4_inkling_conv_group group,
    float *chronological_output,
    uint32_t width) {
    if (!slot || !valid_group(group) || !chronological_output ||
        slot->conv_width[group] != width || width == 0) return false;
    memset(chronological_output, 0,
           (size_t)DS4_INKLING_CONV_HISTORY * width * sizeof(float));
    const uint32_t count = slot->conv_count[group];
    const uint32_t leading_zero_rows = DS4_INKLING_CONV_HISTORY - count;
    for (uint32_t lag = count; lag > 0; lag--) {
        const uint32_t row = leading_zero_rows + count - lag;
        const uint32_t index = (slot->conv_next[group] +
                                DS4_INKLING_CONV_HISTORY - lag) %
                               DS4_INKLING_CONV_HISTORY;
        memcpy(chronological_output + (uint64_t)row * width,
               slot->conv_history[group] + (uint64_t)index * width,
               (size_t)width * sizeof(float));
    }
    return true;
}

static float history_value(
    const ds4_inkling_state_slot *slot,
    ds4_inkling_conv_group group,
    uint32_t lag,
    uint32_t channel) {
    if (lag == 0 || lag > slot->conv_count[group]) return 0.0f;
    const uint32_t index = (slot->conv_next[group] +
                            DS4_INKLING_CONV_HISTORY - lag) %
                           DS4_INKLING_CONV_HISTORY;
    return slot->conv_history[group][(uint64_t)index * slot->conv_width[group] + channel];
}

bool ds4_inkling_state_shortconv(
    ds4_inkling_state_slot *slot,
    ds4_inkling_conv_group group,
    const float *input,
    const float *static_taps,
    const float *dynamic_taps,
    float *output,
    uint32_t tokens,
    uint32_t width,
    uint32_t heads,
    uint32_t kernel_size) {
    if (!slot || !valid_group(group) || !input || !static_taps || !output ||
        tokens == 0 || width == 0 || heads == 0 || kernel_size == 0 ||
        kernel_size > DS4_INKLING_CONV_KERNEL ||
        kernel_size - 1u > DS4_INKLING_CONV_HISTORY ||
        slot->conv_width[group] != width || width % heads != 0) return false;
    float *current = (float *)malloc((size_t)width * sizeof(float));
    if (!current) return false;
    const uint32_t head_dim = width / heads;
    const float norm = 1.0f / sqrtf((float)((dynamic_taps ? 2u : 1u) *
                                           kernel_size + 1u));
    for (uint32_t token = 0; token < tokens; token++) {
        const float *token_input = input + (uint64_t)token * width;
        memcpy(current, token_input, (size_t)width * sizeof(float));
        for (uint32_t channel = 0; channel < width; channel++) {
            const uint32_t head = channel / head_dim;
            float sum = 0.0f;
            for (uint32_t lag = 0; lag < kernel_size; lag++) {
                const float sample = lag == 0 ? current[channel] :
                    history_value(slot, group, lag, channel);
                /* GGUF stores static taps oldest-to-current, while the causal
                 * loop below visits samples current-to-oldest. */
                float tap = static_taps[(uint64_t)channel * kernel_size +
                                        (kernel_size - 1u - lag)];
                if (dynamic_taps) {
                    tap += dynamic_taps[((uint64_t)token * heads + head) *
                                        kernel_size + lag];
                }
                sum += sample * tap;
            }
            output[(uint64_t)token * width + channel] =
                (current[channel] + sum) * norm;
        }
        if (!ds4_inkling_state_conv_push(slot, group, current, width)) {
            free(current);
            return false;
        }
    }
    free(current);
    return true;
}

bool ds4_inkling_prepare_qk(
    float *output,
    const float *input,
    const float *learned_scale,
    const uint64_t *positions,
    uint32_t rows,
    uint32_t width,
    float threshold,
    float log_scale_alpha,
    float log_scale_floor,
    int apply_log_scale) {
    if (!output || !input || !positions || rows == 0 || width == 0 ||
        !(threshold > 0.0f) ||
        (apply_log_scale && !(log_scale_floor > 0.0f))) return false;
    for (uint32_t row = 0; row < rows; row++) {
        const uint64_t offset = (uint64_t)row * width;
        float norm_sq = 0.0f;
        for (uint32_t col = 0; col < width; col++) {
            const float scale = learned_scale ? learned_scale[offset + col] : 1.0f;
            const float value = input[offset + col] * scale;
            output[offset + col] = value;
            norm_sq += value * value;
        }
        const float norm = sqrtf(norm_sq);
        float clip = threshold / (norm + 1e-6f);
        if (clip > 1.0f) clip = 1.0f;
        float log_scale = 1.0f;
        if (apply_log_scale) {
            float ratio = ((float)positions[row] + 1.0f) / log_scale_floor;
            if (ratio < 1.0f) ratio = 1.0f;
            log_scale += log_scale_alpha * logf(ratio);
        }
        for (uint32_t col = 0; col < width; col++) {
            output[offset + col] *= clip * log_scale;
        }
    }
    return true;
}

bool ds4_inkling_relative_bias(
    float *output,
    const float *relative_states,
    const float *projection,
    const int64_t *relative_distances,
    uint32_t rows,
    uint32_t heads,
    uint32_t relative_dim,
    uint32_t relative_extent,
    uint32_t keys) {
    if (!output || !relative_states || !projection || !relative_distances ||
        rows == 0 || heads == 0 || relative_dim == 0 ||
        relative_extent == 0 || keys == 0) return false;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            const float *state = relative_states +
                ((uint64_t)row * heads + head) * relative_dim;
            for (uint32_t key = 0; key < keys; key++) {
                const int64_t distance = relative_distances[(uint64_t)row * keys + key];
                float sum = 0.0f;
                if (distance >= 0 && (uint64_t)distance < relative_extent) {
                    for (uint32_t dim = 0; dim < relative_dim; dim++) {
                        /* GGUF tensor shape is {relative_dim, extent}, so
                         * the distance/profile row is contiguous. */
                        sum += state[dim] *
                            projection[(uint64_t)(uint32_t)distance * relative_dim +
                                       dim];
                    }
                }
                output[((uint64_t)row * heads + head) * keys + key] = sum;
            }
        }
    }
    return true;
}

static float fp16_to_float(uint16_t half) {
    const uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t mantissa = half & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            exponent = 127u - 15u + 1u;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

bool ds4_inkling_state_attention(
    const ds4_inkling_state_slot *slot,
    const float *query,
    uint32_t query_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint64_t query_position,
    const float *relative_bias,
    uint32_t relative_extent,
    const float *sink_logits,
    float *output) {
    if (!slot || !query || !output || !slot->key || !slot->value ||
        query_heads == 0 || kv_heads == 0 || head_dim == 0 ||
        query_heads % kv_heads != 0 ||
        (uint64_t)kv_heads * head_dim != slot->kv_width ||
        (relative_bias && relative_extent == 0) || query_position == UINT64_MAX) {
        return false;
    }
    const uint64_t retained_start = ds4_inkling_state_oldest_position(slot);
    uint64_t key_start = retained_start;
    if (slot->is_sliding) {
        const uint64_t window_start = query_position + 1u > slot->capacity
            ? query_position + 1u - slot->capacity : 0;
        if (window_start > key_start) key_start = window_start;
    }
    uint64_t key_end = slot->next_position;
    if (key_end > query_position + 1u) key_end = query_position + 1u;
    if (key_end < key_start) key_end = key_start;

    const float scale = 1.0f / sqrtf((float)head_dim);
    const uint32_t heads_per_kv = query_heads / kv_heads;
    float *head_output = output;
    for (uint32_t head = 0; head < query_heads; head++) {
        float max_logit = sink_logits ? sink_logits[head] : -FLT_MAX;
        const uint32_t kv_head = head / heads_per_kv;
        for (uint64_t position = key_start; position < key_end; position++) {
            const uint32_t index = (uint32_t)(position % slot->capacity);
            const uint64_t offset = ((uint64_t)index * kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            const float *head_query = query + (uint64_t)head * head_dim;
            for (uint32_t dim = 0; dim < head_dim; dim++) {
                dot += head_query[dim] * fp16_to_float(slot->key[offset + dim]);
            }
            float bias = 0.0f;
            const uint64_t distance = query_position - position;
            if (relative_bias && distance < relative_extent) {
                bias = relative_bias[(uint64_t)head * relative_extent + distance];
            }
            const float logit = dot * scale + bias;
            if (logit > max_logit) max_logit = logit;
        }
        float denominator = sink_logits ? expf(sink_logits[head] - max_logit) : 0.0f;
        for (uint32_t dim = 0; dim < head_dim; dim++) head_output[(uint64_t)head * head_dim + dim] = 0.0f;
        for (uint64_t position = key_start; position < key_end; position++) {
            const uint32_t index = (uint32_t)(position % slot->capacity);
            const uint64_t offset = ((uint64_t)index * kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            const float *head_query = query + (uint64_t)head * head_dim;
            for (uint32_t dim = 0; dim < head_dim; dim++) {
                dot += head_query[dim] * fp16_to_float(slot->key[offset + dim]);
            }
            const uint64_t distance = query_position - position;
            const float bias = relative_bias && distance < relative_extent
                ? relative_bias[(uint64_t)head * relative_extent + distance] : 0.0f;
            const float weight = expf(dot * scale + bias - max_logit);
            denominator += weight;
            for (uint32_t dim = 0; dim < head_dim; dim++) {
                head_output[(uint64_t)head * head_dim + dim] +=
                    weight * fp16_to_float(slot->value[offset + dim]);
            }
        }
        if (denominator > 0.0f) {
            const float inverse = 1.0f / denominator;
            for (uint32_t dim = 0; dim < head_dim; dim++) {
                head_output[(uint64_t)head * head_dim + dim] *= inverse;
            }
        }
    }
    return true;
}
