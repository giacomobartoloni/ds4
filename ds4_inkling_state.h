#ifndef DS4_INKLING_STATE_H
#define DS4_INKLING_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4_inkling.h"

#define DS4_INKLING_CONTEXT 8192u
#define DS4_INKLING_SLIDING_WINDOW 1024u
#define DS4_INKLING_KV_HEADS 8u
#define DS4_INKLING_KV_HEAD_DIM 128u
#define DS4_INKLING_KV_WIDTH (DS4_INKLING_KV_HEADS * DS4_INKLING_KV_HEAD_DIM)
#define DS4_INKLING_CONV_GROUP_COUNT 4u
#define DS4_INKLING_CONV_HISTORY 3u
#define DS4_INKLING_CONV_KERNEL 4u

typedef enum {
    DS4_INKLING_CONV_KEY = 0,
    DS4_INKLING_CONV_VALUE = 1,
    DS4_INKLING_CONV_ATTENTION_OUT = 2,
    DS4_INKLING_CONV_FFN_OUT = 3,
} ds4_inkling_conv_group;

typedef struct {
    uint32_t context_capacity;
    uint32_t sliding_window;
    uint32_t kv_width;
    uint32_t conv_width[DS4_INKLING_CONV_GROUP_COUNT];
} ds4_inkling_state_config;

typedef struct {
    uint32_t capacity;
    uint32_t kv_width;
    uint32_t count;
    uint64_t next_position;
    bool is_sliding;
    uint16_t *key;
    uint16_t *value;
    uint32_t conv_width[DS4_INKLING_CONV_GROUP_COUNT];
    uint32_t conv_count[DS4_INKLING_CONV_GROUP_COUNT];
    uint32_t conv_next[DS4_INKLING_CONV_GROUP_COUNT];
    float *conv_history[DS4_INKLING_CONV_GROUP_COUNT];

    bool transaction_active;
    uint32_t transaction_limit;
    uint32_t transaction_count;
    uint32_t transaction_count_snapshot;
    uint64_t transaction_position_snapshot;
    uint32_t *transaction_indices;
    uint16_t *transaction_key_backup;
    uint16_t *transaction_value_backup;
    float *transaction_conv_backup[DS4_INKLING_CONV_GROUP_COUNT];
    uint32_t transaction_conv_count[DS4_INKLING_CONV_GROUP_COUNT];
    uint32_t transaction_conv_next[DS4_INKLING_CONV_GROUP_COUNT];
} ds4_inkling_state_slot;

typedef struct {
    uint32_t slot_count;
    uint32_t kv_width;
    ds4_inkling_state_slot slots[DS4_INKLING_CACHE_SLOTS];
} ds4_inkling_state_session;

/* One state allocation per logical visit; only the plan's weight owner repeats. */
bool ds4_inkling_state_session_init(
    ds4_inkling_state_session *session,
    const ds4_inkling_plan *plan,
    const ds4_inkling_state_config *config);
void ds4_inkling_state_session_free(ds4_inkling_state_session *session);
uint64_t ds4_inkling_state_session_bytes(const ds4_inkling_state_session *session);

uint64_t ds4_inkling_state_oldest_position(const ds4_inkling_state_slot *slot);
bool ds4_inkling_state_append_kv_fp16(
    ds4_inkling_state_slot *slot,
    uint64_t position,
    const uint16_t *key,
    const uint16_t *value);
bool ds4_inkling_state_read_kv_fp16(
    const ds4_inkling_state_slot *slot,
    uint64_t position,
    uint16_t *key,
    uint16_t *value);
bool ds4_inkling_state_reset(ds4_inkling_state_slot *slot);
bool ds4_inkling_state_clone(
    ds4_inkling_state_slot *dst,
    const ds4_inkling_state_slot *src);

/* Transactions preserve overwritten ring rows and all four conv histories. */
bool ds4_inkling_state_begin(ds4_inkling_state_slot *slot, uint32_t max_tokens);
bool ds4_inkling_state_commit(ds4_inkling_state_slot *slot);
bool ds4_inkling_state_rollback(ds4_inkling_state_slot *slot);

bool ds4_inkling_state_conv_push(
    ds4_inkling_state_slot *slot,
    ds4_inkling_conv_group group,
    const float *input,
    uint32_t width);
bool ds4_inkling_state_conv_read_lag(
    const ds4_inkling_state_slot *slot,
    ds4_inkling_conv_group group,
    uint32_t lag,
    float *output,
    uint32_t width);
bool ds4_inkling_state_conv_export_history(
    const ds4_inkling_state_slot *slot,
    ds4_inkling_conv_group group,
    float *chronological_output,
    uint32_t width);

/* Static and dynamic taps use backward-distance order: tap zero is current. */
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
    uint32_t kernel_size);

/* Apply learned per-channel scaling, L2 clip, then optional full-layer log scale. */
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
    int apply_log_scale);

bool ds4_inkling_relative_bias(
    float *output,
    const float *relative_states,
    const float *projection,
    const int64_t *relative_distances,
    uint32_t rows,
    uint32_t heads,
    uint32_t relative_dim,
    uint32_t relative_extent,
    uint32_t keys);

/* Decode attention uses FP16 cache values, GQA, relative bias and a zero-value sink. */
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
    float *output);

#endif /* DS4_INKLING_STATE_H */
