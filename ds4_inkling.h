#ifndef DS4_INKLING_H
#define DS4_INKLING_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define DS4_INKLING_PHYSICAL_LAYERS 24u
#define DS4_INKLING_LOOP_START 6u
#define DS4_INKLING_LOOP_END 18u
#define DS4_INKLING_LOOP_PASSES 3u
#define DS4_INKLING_RESIDUAL_BLOCK 6u
#define DS4_INKLING_CORE_LAYERS \
    (DS4_INKLING_LOOP_END - DS4_INKLING_LOOP_START)
#define DS4_INKLING_CACHE_SLOTS 48u
#define DS4_INKLING_VISITS 48u
#define DS4_INKLING_NO_PASS UINT8_MAX

/* A physical layer identifies shared weights; each visit owns one cache slot. */
typedef struct {
    uint32_t physical_layer;
    uint32_t cache_slot;
    uint8_t pass;
    uint8_t phase;
    uint8_t is_sliding_window;
} ds4_inkling_visit;

typedef enum {
    DS4_INKLING_PHASE_PRELUDE = 0,
    DS4_INKLING_PHASE_CORE = 1,
    DS4_INKLING_PHASE_EPILOGUE = 2,
} ds4_inkling_phase;

typedef struct {
    uint32_t physical_layers;
    uint32_t loop_start;
    uint32_t loop_end;
    uint32_t loop_passes;
    uint32_t residual_block_size;
    const uint8_t *sliding_window_pattern;
    size_t pattern_count;
} ds4_inkling_config;

typedef struct {
    uint32_t physical_layer_count;
    uint32_t cache_slot_count;
    uint32_t visit_count;
    uint32_t full_attention_visit_count;
    uint32_t sliding_attention_visit_count;
    ds4_inkling_visit visits[DS4_INKLING_VISITS];
} ds4_inkling_plan;

typedef enum {
    DS4_INKLING_PLAN_OK = 0,
    DS4_INKLING_PLAN_INVALID_ARGUMENT,
    DS4_INKLING_PLAN_UNSUPPORTED_CONFIG,
    DS4_INKLING_PLAN_INVALID_PATTERN,
} ds4_inkling_plan_status;

static inline void ds4_inkling_append_visit(
    ds4_inkling_plan *plan,
    const ds4_inkling_config *config,
    uint32_t physical_layer,
    uint32_t cache_slot,
    uint8_t phase,
    uint8_t pass) {
    ds4_inkling_visit *visit = &plan->visits[plan->visit_count++];
    visit->physical_layer = physical_layer;
    visit->cache_slot = cache_slot;
    visit->pass = pass;
    visit->phase = phase;
    visit->is_sliding_window = config->sliding_window_pattern[physical_layer];
    if (visit->is_sliding_window) {
        plan->sliding_attention_visit_count++;
    } else {
        plan->full_attention_visit_count++;
    }
}

/* Validate the supported GGUF schedule and build its execution/cache map. */
static inline ds4_inkling_plan_status ds4_inkling_build_plan(
    const ds4_inkling_config *config,
    ds4_inkling_plan *plan) {
    if (plan != NULL) {
        memset(plan, 0, sizeof(*plan));
    }
    if (config == NULL || plan == NULL) {
        return DS4_INKLING_PLAN_INVALID_ARGUMENT;
    }
    if (config->physical_layers != DS4_INKLING_PHYSICAL_LAYERS ||
        config->loop_start != DS4_INKLING_LOOP_START ||
        config->loop_end != DS4_INKLING_LOOP_END ||
        config->loop_passes != DS4_INKLING_LOOP_PASSES ||
        config->residual_block_size != DS4_INKLING_RESIDUAL_BLOCK) {
        return DS4_INKLING_PLAN_UNSUPPORTED_CONFIG;
    }
    if (config->sliding_window_pattern == NULL ||
        config->pattern_count != DS4_INKLING_PHYSICAL_LAYERS) {
        return DS4_INKLING_PLAN_INVALID_PATTERN;
    }
    for (uint32_t layer = 0; layer < DS4_INKLING_PHYSICAL_LAYERS; layer++) {
        uint8_t expected_sliding = (uint8_t)(layer % DS4_INKLING_RESIDUAL_BLOCK !=
                                             DS4_INKLING_RESIDUAL_BLOCK - 1);
        if (config->sliding_window_pattern[layer] > 1 ||
            config->sliding_window_pattern[layer] != expected_sliding) {
            return DS4_INKLING_PLAN_INVALID_PATTERN;
        }
    }

    plan->physical_layer_count = DS4_INKLING_PHYSICAL_LAYERS;
    plan->cache_slot_count = DS4_INKLING_CACHE_SLOTS;
    for (uint32_t layer = 0; layer < DS4_INKLING_LOOP_START; layer++) {
        ds4_inkling_append_visit(plan, config, layer, layer,
                                 DS4_INKLING_PHASE_PRELUDE,
                                 DS4_INKLING_NO_PASS);
    }
    for (uint32_t pass = 0; pass < DS4_INKLING_LOOP_PASSES; pass++) {
        for (uint32_t layer = DS4_INKLING_LOOP_START;
             layer < DS4_INKLING_LOOP_END; layer++) {
            uint32_t cache_slot = pass == 0
                ? layer
                : DS4_INKLING_PHYSICAL_LAYERS +
                    (pass - 1) * DS4_INKLING_CORE_LAYERS +
                    layer - DS4_INKLING_LOOP_START;
            ds4_inkling_append_visit(plan, config, layer, cache_slot,
                                     DS4_INKLING_PHASE_CORE,
                                     (uint8_t)pass);
        }
    }
    for (uint32_t layer = DS4_INKLING_LOOP_END;
         layer < DS4_INKLING_PHYSICAL_LAYERS; layer++) {
        ds4_inkling_append_visit(plan, config, layer, layer,
                                 DS4_INKLING_PHASE_EPILOGUE,
                                 DS4_INKLING_NO_PASS);
    }
    if (plan->visit_count != DS4_INKLING_VISITS) {
        memset(plan, 0, sizeof(*plan));
        return DS4_INKLING_PLAN_UNSUPPORTED_CONFIG;
    }
    return DS4_INKLING_PLAN_OK;
}

#endif /* DS4_INKLING_H */
