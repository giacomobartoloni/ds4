#ifndef DS4_INKLING_H
#define DS4_INKLING_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define DS4_INKLING_CONTEXT_MIN 8192u
#define DS4_INKLING_CONTEXT_MAX 1048576u
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

/* Validated metadata for the initial text-only IQ4_NL Inkling 2 Mini GGUF. */
typedef struct {
    uint32_t physical_layers;
    uint32_t embedding_dim;
    uint32_t vocab_padded;
    uint32_t vocab_unpadded;
    uint32_t head_dim;
    uint32_t expert_count;
    uint32_t experts_used;
    uint32_t expert_ffn_dim;
    uint64_t context_length;
    uint32_t loop_start;
    uint32_t loop_end;
    uint32_t loop_passes;
    uint32_t gate_rank;
    uint32_t residual_block_size;
    uint32_t sliding_window;
    uint32_t shortconv_cache;
    uint32_t relative_dim;
    uint32_t relative_extent;
    uint32_t log_scale_floor;
    uint32_t query_heads[DS4_INKLING_PHYSICAL_LAYERS];
    uint32_t kv_heads[DS4_INKLING_PHYSICAL_LAYERS];
    uint8_t sliding_window_pattern[DS4_INKLING_PHYSICAL_LAYERS];
    float rms_epsilon;
    float qk_clip;
    float relative_clip;
    float log_scale_alpha;
    float expert_weight_scale;
    float situ_beta;
    float situ_linear_beta;
    float final_logit_softcap;
    bool loop_varying;
} ds4_inkling_model_config;

typedef enum {
    DS4_INKLING_MODEL_OK = 0,
    DS4_INKLING_MODEL_INVALID_ARGUMENT,
    DS4_INKLING_MODEL_UNSUPPORTED_CONFIG,
    DS4_INKLING_MODEL_INVALID_PATTERN,
    DS4_INKLING_MODEL_INVALID_NUMERIC,
} ds4_inkling_model_status;

static inline ds4_inkling_model_status ds4_inkling_model_fail(
        ds4_inkling_model_status status,
        char *error,
        size_t error_capacity,
        const char *message) {
    if (error && error_capacity) snprintf(error, error_capacity, "%s", message);
    return status;
}

static inline int ds4_inkling_model_near(float actual, float expected) {
    return isfinite(actual) && fabsf(actual - expected) <=
        1.0e-5f * fmaxf(1.0f, fabsf(expected));
}

static inline ds4_inkling_model_status ds4_inkling_model_validate(
        const ds4_inkling_model_config *config,
        char *error,
        size_t error_capacity) {
    if (error && error_capacity) error[0] = '\0';
    if (!config) {
        return ds4_inkling_model_fail(DS4_INKLING_MODEL_INVALID_ARGUMENT,
                                      error, error_capacity,
                                      "model configuration is null");
    }
    if (config->physical_layers != 24u || config->embedding_dim != 2304u ||
        config->vocab_padded != 201024u || config->vocab_unpadded != 200058u ||
        config->head_dim != 128u || config->expert_count != 128u ||
        config->experts_used != 8u || config->expert_ffn_dim != 1152u ||
        config->context_length < DS4_INKLING_CONTEXT_MIN ||
        config->context_length > DS4_INKLING_CONTEXT_MAX ||
        config->loop_start != 6u || config->loop_end != 18u ||
        config->loop_passes != 3u || config->gate_rank != 64u ||
        config->residual_block_size != 6u || config->sliding_window != 1024u ||
        config->shortconv_cache != 4u || config->relative_dim != 16u ||
        config->relative_extent != 1024u || config->log_scale_floor != 8192u ||
        config->loop_varying) {
        return ds4_inkling_model_fail(DS4_INKLING_MODEL_UNSUPPORTED_CONFIG,
                                      error, error_capacity,
                                      "metadata does not match supported Inkling 2 Mini dimensions");
    }
    for (uint32_t layer = 0; layer < DS4_INKLING_PHYSICAL_LAYERS; layer++) {
        if (config->query_heads[layer] != 24u || config->kv_heads[layer] != 8u) {
            return ds4_inkling_model_fail(DS4_INKLING_MODEL_UNSUPPORTED_CONFIG,
                                          error, error_capacity,
                                          "per-layer attention head counts are unsupported");
        }
        const uint8_t expected_sliding = (uint8_t)(layer % 6u != 5u);
        if (config->sliding_window_pattern[layer] > 1u ||
            config->sliding_window_pattern[layer] != expected_sliding) {
            return ds4_inkling_model_fail(DS4_INKLING_MODEL_INVALID_PATTERN,
                                          error, error_capacity,
                                          "attention schedule must be sliding except layers 5, 11, 17, and 23");
        }
    }
    if (!ds4_inkling_model_near(config->rms_epsilon, 1.0e-6f) ||
        !ds4_inkling_model_near(config->qk_clip, 22.627416610717773f) ||
        !ds4_inkling_model_near(config->relative_clip, 8.0f) ||
        !ds4_inkling_model_near(config->log_scale_alpha, 0.1f) ||
        !ds4_inkling_model_near(config->expert_weight_scale, 8.0f) ||
        !ds4_inkling_model_near(config->situ_beta, 4.0f) ||
        !ds4_inkling_model_near(config->situ_linear_beta, 25.0f) ||
        !ds4_inkling_model_near(config->final_logit_softcap, 30.0f)) {
        return ds4_inkling_model_fail(DS4_INKLING_MODEL_INVALID_NUMERIC,
                                      error, error_capacity,
                                      "normalization, clipping, routing, or activation metadata is unsupported");
    }
    return DS4_INKLING_MODEL_OK;
}

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
