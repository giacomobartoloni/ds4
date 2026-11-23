#include "ds4_inkling.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static ds4_inkling_model_config valid_config(void) {
    ds4_inkling_model_config config = {
        .physical_layers = 24,
        .embedding_dim = 2304,
        .vocab_padded = 201024,
        .vocab_unpadded = 200058,
        .head_dim = 128,
        .expert_count = 128,
        .experts_used = 8,
        .expert_ffn_dim = 1152,
        .context_length = 1048576,
        .loop_start = 6,
        .loop_end = 18,
        .loop_passes = 3,
        .gate_rank = 64,
        .residual_block_size = 6,
        .sliding_window = 1024,
        .shortconv_cache = 4,
        .relative_dim = 16,
        .relative_extent = 1024,
        .log_scale_floor = 8192,
        .rms_epsilon = 1.0e-6f,
        .qk_clip = 22.627416610717773f,
        .relative_clip = 8.0f,
        .log_scale_alpha = 0.1f,
        .expert_weight_scale = 8.0f,
        .situ_beta = 4.0f,
        .situ_linear_beta = 25.0f,
        .final_logit_softcap = 30.0f,
        .loop_varying = false,
    };
    for (uint32_t i = 0; i < DS4_INKLING_PHYSICAL_LAYERS; i++) {
        config.query_heads[i] = 24;
        config.kv_heads[i] = 8;
        config.sliding_window_pattern[i] = (uint8_t)(i % 6u != 5u);
    }
    return config;
}

static void expect_rejected(ds4_inkling_model_config config) {
    char error[160] = {0};
    assert(ds4_inkling_model_validate(&config, error, sizeof(error)) !=
           DS4_INKLING_MODEL_OK);
    assert(error[0] != '\0');
}

int main(void) {
    ds4_inkling_model_config config = valid_config();
    char error[160] = {0};
    assert(ds4_inkling_model_validate(&config, error, sizeof(error)) ==
           DS4_INKLING_MODEL_OK);
    assert(error[0] == '\0');

    config = valid_config();
    config.loop_end = 17;
    expect_rejected(config);

    config = valid_config();
    config.gate_rank = 32;
    expect_rejected(config);

    config = valid_config();
    config.kv_heads[11] = 4;
    expect_rejected(config);

    config = valid_config();
    config.sliding_window_pattern[11] = 1;
    expect_rejected(config);

    config = valid_config();
    config.vocab_unpadded = config.vocab_padded;
    expect_rejected(config);

    config = valid_config();
    config.context_length = 4096;
    expect_rejected(config);

    config = valid_config();
    config.rms_epsilon = NAN;
    expect_rejected(config);

    puts("Inkling2 model configuration: ok");
    return 0;
}
