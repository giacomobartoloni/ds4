#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../ds4_inkling.h"
#include "../ds4_inkling_state.h"

static void make_plan(ds4_inkling_plan *plan) {
    uint8_t pattern[DS4_INKLING_PHYSICAL_LAYERS];
    memset(pattern, 1, sizeof(pattern));
    pattern[5] = pattern[11] = pattern[17] = pattern[23] = 0;
    const ds4_inkling_config config = {
        .physical_layers = DS4_INKLING_PHYSICAL_LAYERS,
        .loop_start = DS4_INKLING_LOOP_START,
        .loop_end = DS4_INKLING_LOOP_END,
        .loop_passes = DS4_INKLING_LOOP_PASSES,
        .residual_block_size = DS4_INKLING_RESIDUAL_BLOCK,
        .sliding_window_pattern = pattern,
        .pattern_count = DS4_INKLING_PHYSICAL_LAYERS,
    };
    assert(ds4_inkling_build_plan(&config, plan) == DS4_INKLING_PLAN_OK);
}

static ds4_inkling_state_config small_state_config(void) {
    return (ds4_inkling_state_config){
        .context_capacity = 9,
        .sliding_window = 4,
        .kv_width = 2,
        .conv_width = { 2, 2, 3, 2 },
    };
}

static void append_pattern(ds4_inkling_state_slot *slot, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const uint64_t pos = slot->next_position;
        const uint16_t key[2] = { (uint16_t)(0x3c00u + pos), (uint16_t)(0xbc00u + pos) };
        const uint16_t value[2] = { (uint16_t)(0x4000u + pos), (uint16_t)(0xc000u + pos) };
        assert(ds4_inkling_state_append_kv_fp16(slot, pos, key, value));
    }
}

static int near(float a, float b, float epsilon) {
    return fabsf(a - b) <= epsilon;
}

static void test_independent_ring_slots(void) {
    ds4_inkling_plan plan;
    make_plan(&plan);
    const ds4_inkling_state_config config = small_state_config();
    ds4_inkling_state_session session;
    assert(ds4_inkling_state_session_init(&session, &plan, &config));
    ds4_inkling_state_slot *sliding = &session.slots[0];
    ds4_inkling_state_slot *full = &session.slots[5];
    assert(sliding->is_sliding && sliding->capacity == 4);
    assert(!full->is_sliding && full->capacity == 9);
    assert(sliding->key != full->key && sliding->value != full->value);
    assert(sliding->conv_history[0] != session.slots[1].conv_history[0]);
    ds4_inkling_state_slot *core0 = &session.slots[6];
    ds4_inkling_state_slot *core1 = &session.slots[24];
    ds4_inkling_state_slot *core2 = &session.slots[36];
    assert(plan.visits[6].physical_layer == plan.visits[18].physical_layer);
    assert(plan.visits[18].physical_layer == plan.visits[30].physical_layer);
    assert(core0->key != core1->key && core1->key != core2->key &&
           core0->key != core2->key);
    append_pattern(core0, 1);
    assert(core0->count == 1 && core1->count == 0 && core2->count == 0);

    append_pattern(sliding, 6);
    assert(sliding->next_position == 6 && sliding->count == 4);
    assert(ds4_inkling_state_oldest_position(sliding) == 2);
    uint16_t key[2], value[2];
    assert(!ds4_inkling_state_read_kv_fp16(sliding, 1, key, value));
    assert(ds4_inkling_state_read_kv_fp16(sliding, 2, key, value));
    assert(key[0] == 0x3c02u && value[0] == 0x4002u);
    assert(ds4_inkling_state_read_kv_fp16(sliding, 5, key, value));
    assert(key[0] == 0x3c05u && value[0] == 0x4005u);

    append_pattern(full, 10);
    assert(full->next_position == 10 && full->count == 9);
    assert(ds4_inkling_state_oldest_position(full) == 1);
    assert(!ds4_inkling_state_read_kv_fp16(full, 0, key, value));
    assert(ds4_inkling_state_read_kv_fp16(full, 9, key, value));
    assert(key[0] == 0x3c09u && value[0] == 0x4009u);
    ds4_inkling_state_session_free(&session);
}

static void test_rejects_malformed_state_plan(void) {
    ds4_inkling_plan plan;
    make_plan(&plan);
    const ds4_inkling_state_config config = small_state_config();
    ds4_inkling_state_session session;
    ds4_inkling_plan invalid = plan;
    invalid.visits[18].is_sliding_window =
        (uint8_t)!invalid.visits[18].is_sliding_window;
    assert(!ds4_inkling_state_session_init(&session, &invalid, &config));

    invalid = plan;
    invalid.visits[18].physical_layer = DS4_INKLING_PHYSICAL_LAYERS;
    assert(!ds4_inkling_state_session_init(&session, &invalid, &config));

    invalid = plan;
    invalid.visits[18].physical_layer = 0;
    assert(!ds4_inkling_state_session_init(&session, &invalid, &config));
}

static void test_transaction_rollback_reset_and_clone(void) {
    ds4_inkling_plan plan;
    make_plan(&plan);
    const ds4_inkling_state_config config = small_state_config();
    ds4_inkling_state_session session;
    assert(ds4_inkling_state_session_init(&session, &plan, &config));
    ds4_inkling_state_slot *source = &session.slots[0];
    append_pattern(source, 3);
    const float conv0[2] = { 0.25f, -0.5f };
    assert(ds4_inkling_state_conv_push(source, DS4_INKLING_CONV_KEY,
                                       conv0, 2));
    assert(ds4_inkling_state_begin(source, 6));
    append_pattern(source, 6);
    const float conv1[2] = { 2.0f, 3.0f };
    assert(ds4_inkling_state_conv_push(source, DS4_INKLING_CONV_KEY,
                                       conv1, 2));
    assert(!ds4_inkling_state_begin(source, 1));
    assert(ds4_inkling_state_rollback(source));
    assert(source->next_position == 3 && source->count == 3);
    uint16_t key[2], value[2];
    assert(ds4_inkling_state_read_kv_fp16(source, 0, key, value));
    assert(key[0] == 0x3c00u && value[0] == 0x4000u);
    float history[2];
    assert(ds4_inkling_state_conv_read_lag(source, DS4_INKLING_CONV_KEY,
                                            1, history, 2));
    assert(history[0] == conv0[0] && history[1] == conv0[1]);

    assert(ds4_inkling_state_begin(source, 1));
    append_pattern(source, 1);
    assert(ds4_inkling_state_commit(source));
    assert(source->next_position == 4 && source->count == 4);

    assert(ds4_inkling_state_clone(&session.slots[1], source));
    assert(session.slots[1].key != source->key);
    assert(ds4_inkling_state_read_kv_fp16(&session.slots[1], 3, key, value));
    assert(key[0] == 0x3c03u);
    append_pattern(&session.slots[1], 1);
    assert(source->next_position == 4 && session.slots[1].next_position == 5);
    assert(ds4_inkling_state_reset(source));
    assert(source->next_position == 0 && source->count == 0);
    assert(!ds4_inkling_state_conv_read_lag(source, DS4_INKLING_CONV_KEY,
                                             1, history, 2));
    ds4_inkling_state_session_free(&session);
}

static void run_conv_partitioned(ds4_inkling_state_slot *slot,
                                 const float *input, const float *weights,
                                 const float *dynamic, float *output,
                                 uint32_t total, uint32_t width,
                                 uint32_t heads, const uint32_t *chunks,
                                 uint32_t n_chunks) {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < n_chunks; i++) {
        const uint32_t count = chunks[i];
        assert(ds4_inkling_state_shortconv(
            slot, DS4_INKLING_CONV_KEY, input + (uint64_t)offset * width,
            weights, dynamic ? dynamic + (uint64_t)offset * heads * 4 : NULL,
            output + (uint64_t)offset * width, count, width, heads, 4));
        offset += count;
    }
    assert(offset == total);
}

static void test_shortconv_prefill_partition_invariance(void) {
    ds4_inkling_plan plan;
    make_plan(&plan);
    ds4_inkling_state_config config = small_state_config();
    config.conv_width[DS4_INKLING_CONV_KEY] = 4;
    ds4_inkling_state_session a, b;
    assert(ds4_inkling_state_session_init(&a, &plan, &config));
    assert(ds4_inkling_state_session_init(&b, &plan, &config));
    enum { TOKENS = 7, WIDTH = 4, HEADS = 2, KERNEL = 4 };
    float input[TOKENS * WIDTH], weights[KERNEL * WIDTH];
    float dynamic[TOKENS * HEADS * KERNEL];
    float whole[TOKENS * WIDTH], chunked[TOKENS * WIDTH];
    for (uint32_t i = 0; i < TOKENS * WIDTH; i++) input[i] = (float)((int)i - 9) / 7.0f;
    for (uint32_t i = 0; i < KERNEL * WIDTH; i++) weights[i] = (float)((int)(i % 7) - 3) / 9.0f;
    for (uint32_t i = 0; i < TOKENS * HEADS * KERNEL; i++) dynamic[i] = (float)((int)(i % 5) - 2) / 11.0f;
    const uint32_t monolithic[] = { TOKENS };
    const uint32_t split[] = { 2, 1, 4 };
    run_conv_partitioned(&a.slots[0], input, weights, NULL, whole, TOKENS,
                         WIDTH, 1, monolithic, 1);
    run_conv_partitioned(&b.slots[0], input, weights, NULL, chunked, TOKENS,
                         WIDTH, 1, split, 3);
    for (uint32_t i = 0; i < TOKENS * WIDTH; i++) assert(near(whole[i], chunked[i], 1e-7f));
    float exported[DS4_INKLING_CONV_HISTORY * WIDTH];
    assert(ds4_inkling_state_conv_export_history(
        &a.slots[0], DS4_INKLING_CONV_KEY, exported, WIDTH));
    for (uint32_t lag = 0; lag < DS4_INKLING_CONV_HISTORY; lag++) {
        for (uint32_t channel = 0; channel < WIDTH; channel++) {
            assert(exported[(uint64_t)lag * WIDTH + channel] ==
                   input[(uint64_t)(TOKENS - DS4_INKLING_CONV_HISTORY + lag) * WIDTH + channel]);
        }
    }

    assert(ds4_inkling_state_reset(&a.slots[0]));
    assert(ds4_inkling_state_reset(&b.slots[0]));
    run_conv_partitioned(&a.slots[0], input, weights, dynamic, whole, TOKENS,
                         WIDTH, HEADS, monolithic, 1);
    run_conv_partitioned(&b.slots[0], input, weights, dynamic, chunked, TOKENS,
                         WIDTH, HEADS, split, 3);
    for (uint32_t i = 0; i < TOKENS * WIDTH; i++) assert(near(whole[i], chunked[i], 1e-7f));
    ds4_inkling_state_session_free(&a);
    ds4_inkling_state_session_free(&b);
}

static void test_prepare_qk_and_relative_bias(void) {
    const float states[] = { 3.0f, 4.0f, 0.0f, -2.0f };
    const float learned_scale[] = { 2.0f, 1.0f, 1.0f, 0.5f };
    const uint64_t positions[] = { 7, 31 };
    float prepared[4];
    assert(ds4_inkling_prepare_qk(prepared, states, learned_scale, positions,
        2, 2, 3.0f, 0.5f, 8.0f, 1));
    const float n0 = sqrtf(52.0f);
    const float factor0 = 3.0f / (n0 + 1e-6f);
    const float tau0 = 1.0f + 0.5f * logf(1.0f);
    assert(near(prepared[0], 6.0f * factor0 * tau0, 1e-6f));
    assert(near(prepared[1], 4.0f * factor0 * tau0, 1e-6f));
    const float n1 = 1.0f;
    const float factor1 = fminf(1.0f, 3.0f / (n1 + 1e-6f));
    const float tau1 = 1.0f + 0.5f * logf(4.0f);
    assert(near(prepared[2], 0.0f, 1e-6f));
    assert(near(prepared[3], -1.0f * factor1 * tau1, 1e-6f));

    const float projected[] = { 1.0f, 2.0f, -1.0f, 0.5f };
    /* GGUF shape {relative_dim=2, extent=3}: one contiguous row per distance. */
    const float table[] = { 1.0f, 2.0f, 3.0f, -2.0f, 1.0f, 0.0f };
    const int64_t distances[] = { 0, 2 };
    float bias[4];
    assert(ds4_inkling_relative_bias(bias, projected, table, distances,
                                    1, 2, 2, 3, 2));
    assert(bias[0] == 5.0f && bias[1] == 1.0f);
    assert(bias[2] == 0.0f && bias[3] == -1.0f);

    const int64_t out_of_range_distances[] = { 3, -1 };
    float out_of_range_bias[2];
    assert(ds4_inkling_relative_bias(out_of_range_bias, projected, table,
                                    out_of_range_distances, 1, 1, 2, 3, 2));
    assert(out_of_range_bias[0] == 0.0f && out_of_range_bias[1] == 0.0f);
}

static void test_sliding_and_global_attention(void) {
    ds4_inkling_plan plan;
    make_plan(&plan);
    ds4_inkling_state_config config = small_state_config();
    config.context_capacity = 1030;
    config.sliding_window = 4;
    ds4_inkling_state_session session;
    assert(ds4_inkling_state_session_init(&session, &plan, &config));
    ds4_inkling_state_slot *sliding = &session.slots[0];
    for (uint32_t i = 0; i < 6; i++) {
        const uint16_t k[2] = { 0x0000u, 0x0000u };
        const uint16_t v[2] = { 0x3c00u, 0xbc00u };
        assert(ds4_inkling_state_append_kv_fp16(sliding, i, k, v));
    }
    const float q[4] = { 0, 0, 0, 0 }, sink[2] = { 0, 0 };
    float out[4];
    assert(ds4_inkling_state_attention(sliding, q, 2, 1, 2, 5,
                                       NULL, 0, sink, out));
    for (uint32_t i = 0; i < 4; i++) assert(near(out[i], i % 2 == 0 ? 0.8f : -0.8f, 2e-6f));

    ds4_inkling_state_slot *full = &session.slots[5];
    for (uint32_t i = 0; i < 1025; i++) {
        const uint16_t k[2] = { 0, 0 };
        const uint16_t v[2] = { 0x3c00u, 0x3c00u };
        assert(ds4_inkling_state_append_kv_fp16(full, i, k, v));
    }
    assert(full->capacity == 1030 && full->count == 1025);
    const float q_full[4] = { 0, 0, 0, 0 };
    assert(ds4_inkling_state_attention(full, q_full, 2, 1, 2, 1024,
                                       NULL, 0, sink, out));
    const float all_positions = 1025.0f / 1026.0f;
    for (uint32_t i = 0; i < 4; i++) assert(near(out[i], all_positions, 2e-6f));
    ds4_inkling_state_session_free(&session);
}

int main(void) {
    test_independent_ring_slots();
    test_rejects_malformed_state_plan();
    test_transaction_rollback_reset_and_clone();
    test_shortconv_prefill_partition_invariance();
    test_prepare_qk_and_relative_bias();
    test_sliding_and_global_attention();
    puts("Inkling2 state: independent FP16 KV slots, rollback, reset, prefix clone, chunked conv, attention: ok");
    return 0;
}
