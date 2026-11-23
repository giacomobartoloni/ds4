#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../ds4_inkling.h"
#include "../ds4_gguf.h"

static void make_pattern(uint8_t pattern[DS4_INKLING_PHYSICAL_LAYERS]) {
    memset(pattern, 1, DS4_INKLING_PHYSICAL_LAYERS);
    pattern[5] = 0;
    pattern[11] = 0;
    pattern[17] = 0;
    pattern[23] = 0;
}

static ds4_inkling_config make_config(
    const uint8_t pattern[DS4_INKLING_PHYSICAL_LAYERS]) {
    ds4_inkling_config config = {
        .physical_layers = DS4_INKLING_PHYSICAL_LAYERS,
        .loop_start = DS4_INKLING_LOOP_START,
        .loop_end = DS4_INKLING_LOOP_END,
        .loop_passes = DS4_INKLING_LOOP_PASSES,
        .residual_block_size = DS4_INKLING_RESIDUAL_BLOCK,
        .sliding_window_pattern = pattern,
        .pattern_count = DS4_INKLING_PHYSICAL_LAYERS,
    };
    return config;
}

static void test_visit_mapping(void) {
    uint8_t pattern[DS4_INKLING_PHYSICAL_LAYERS];
    make_pattern(pattern);
    ds4_inkling_config config = make_config(pattern);
    ds4_inkling_plan plan;
    assert(ds4_inkling_build_plan(&config, &plan) == DS4_INKLING_PLAN_OK);
    assert(plan.physical_layer_count == 24);
    assert(plan.cache_slot_count == 48);
    assert(plan.visit_count == 48);
    assert(plan.full_attention_visit_count == 8);
    assert(plan.sliding_attention_visit_count == 40);

    uint8_t cache_seen[DS4_INKLING_CACHE_SLOTS] = {0};
    uint8_t physical_seen[DS4_INKLING_PHYSICAL_LAYERS] = {0};
    for (uint32_t layer = 0; layer < 6; layer++) {
        const ds4_inkling_visit *visit = &plan.visits[layer];
        assert(visit->phase == DS4_INKLING_PHASE_PRELUDE);
        assert(visit->pass == DS4_INKLING_NO_PASS);
        assert(visit->physical_layer == layer);
        assert(visit->cache_slot == layer);
        assert(visit->is_sliding_window == pattern[layer]);
        assert(!cache_seen[visit->cache_slot]++);
        physical_seen[visit->physical_layer] = 1;
    }

    for (uint32_t pass = 0; pass < 3; pass++) {
        for (uint32_t layer = 6; layer < 18; layer++) {
            uint32_t visit_index = 6 + pass * 12 + layer - 6;
            uint32_t expected_slot = pass == 0
                ? layer
                : 24 + (pass - 1) * 12 + layer - 6;
            const ds4_inkling_visit *visit = &plan.visits[visit_index];
            assert(visit->phase == DS4_INKLING_PHASE_CORE);
            assert(visit->pass == pass);
            assert(visit->physical_layer == layer);
            assert(visit->cache_slot == expected_slot);
            assert(visit->is_sliding_window == pattern[layer]);
            assert(!cache_seen[visit->cache_slot]++);
            physical_seen[visit->physical_layer] = 1;
        }
    }

    for (uint32_t layer = 18; layer < 24; layer++) {
        uint32_t visit_index = 42 + layer - 18;
        const ds4_inkling_visit *visit = &plan.visits[visit_index];
        assert(visit->phase == DS4_INKLING_PHASE_EPILOGUE);
        assert(visit->pass == DS4_INKLING_NO_PASS);
        assert(visit->physical_layer == layer);
        assert(visit->cache_slot == layer);
        assert(visit->is_sliding_window == pattern[layer]);
        assert(!cache_seen[visit->cache_slot]++);
        physical_seen[visit->physical_layer] = 1;
    }

    for (uint32_t slot = 0; slot < DS4_INKLING_CACHE_SLOTS; slot++) {
        assert(cache_seen[slot] == 1);
    }
    for (uint32_t layer = 0; layer < DS4_INKLING_PHYSICAL_LAYERS; layer++) {
        assert(physical_seen[layer] == 1);
    }

    /* Repeated visits refer to the same physical weight owner. */
    for (uint32_t layer = 6; layer < 18; layer++) {
        const ds4_inkling_visit *pass0 = &plan.visits[layer];
        const ds4_inkling_visit *pass1 = &plan.visits[18 + layer - 6];
        const ds4_inkling_visit *pass2 = &plan.visits[30 + layer - 6];
        assert(pass0->physical_layer == pass1->physical_layer);
        assert(pass1->physical_layer == pass2->physical_layer);
        assert(pass0->cache_slot != pass1->cache_slot);
        assert(pass1->cache_slot != pass2->cache_slot);
        assert(pass0->cache_slot != pass2->cache_slot);
    }
}

static void test_rejects_unsupported_configuration(void) {
    uint8_t pattern[DS4_INKLING_PHYSICAL_LAYERS];
    make_pattern(pattern);
    ds4_inkling_config config = make_config(pattern);
    ds4_inkling_plan plan;
    assert(ds4_inkling_build_plan(NULL, &plan) ==
           DS4_INKLING_PLAN_INVALID_ARGUMENT);
    assert(ds4_inkling_build_plan(&config, NULL) ==
           DS4_INKLING_PLAN_INVALID_ARGUMENT);

    config.physical_layers = 23;
    assert(ds4_inkling_build_plan(&config, &plan) ==
           DS4_INKLING_PLAN_UNSUPPORTED_CONFIG);
    config = make_config(pattern);
    config.loop_start = 5;
    assert(ds4_inkling_build_plan(&config, &plan) ==
           DS4_INKLING_PLAN_UNSUPPORTED_CONFIG);
    config = make_config(pattern);
    config.loop_end = 17;
    assert(ds4_inkling_build_plan(&config, &plan) ==
           DS4_INKLING_PLAN_UNSUPPORTED_CONFIG);
    config = make_config(pattern);
    config.loop_passes = 2;
    assert(ds4_inkling_build_plan(&config, &plan) ==
           DS4_INKLING_PLAN_UNSUPPORTED_CONFIG);
    config = make_config(pattern);
    config.residual_block_size = 3;
    assert(ds4_inkling_build_plan(&config, &plan) ==
           DS4_INKLING_PLAN_UNSUPPORTED_CONFIG);
    config = make_config(pattern);
    config.pattern_count = 23;
    assert(ds4_inkling_build_plan(&config, &plan) ==
           DS4_INKLING_PLAN_INVALID_PATTERN);

    config = make_config(pattern);
    pattern[0] = 2;
    assert(ds4_inkling_build_plan(&config, &plan) ==
           DS4_INKLING_PLAN_INVALID_PATTERN);
    make_pattern(pattern);
    pattern[0] = 0;
    assert(ds4_inkling_build_plan(&config, &plan) ==
           DS4_INKLING_PLAN_INVALID_PATTERN);
}

static void test_loader_quant_block_sizes(void) {
    assert(DS4_TENSOR_IQ4_NL == 20);
    assert(strcmp(gguf_types[19].name, "iq1_s") == 0);
    assert(gguf_types[19].block_elems == 256);
    assert(gguf_types[19].block_bytes == 50);
    assert(strcmp(gguf_types[DS4_TENSOR_IQ4_NL].name, "iq4_nl") == 0);
    assert(gguf_types[DS4_TENSOR_IQ4_NL].block_elems == 32);
    assert(gguf_types[DS4_TENSOR_IQ4_NL].block_bytes == 18);
}

int main(void) {
    test_visit_mapping();
    test_rejects_unsupported_configuration();
    test_loader_quant_block_sizes();
    puts("inkling2 mapping: 48 visits, 48 isolated caches, 24 shared weight owners");
    puts("GGUF loader: IQ1_S 256/50, IQ4_NL 32/18");
    return 0;
}
