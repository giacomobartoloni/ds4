/* Exercise the ordinary session API through a complete synthetic Metal
 * decoder. The anonymous model map contains zero-valued Q8_0/IQ4_NL blocks;
 * this tests wiring, loop order, state ownership and deterministic logits. */
#include "../ds4.c"

#include <assert.h>
#include <math.h>
#include <sys/mman.h>

#define TEST_MODEL_BYTES (800ull * 1024ull * 1024ull)
#define TEST_MAX_TENSORS 1024u
#define TEST_HIDDEN 2304u
#define TEST_VOCAB_PADDED 201024u
#define TEST_VOCAB 200058u

static ds4_tensor test_tensors[TEST_MAX_TENSORS];
static uint32_t test_tensor_count;

static ds4_tensor *test_tensor(uint32_t type, uint32_t ndim,
                               uint64_t d0, uint64_t d1, uint64_t d2,
                               uint64_t bytes) {
    assert(test_tensor_count < TEST_MAX_TENSORS);
    ds4_tensor *tensor = &test_tensors[test_tensor_count++];
    memset(tensor, 0, sizeof(*tensor));
    tensor->ndim = ndim;
    tensor->dim[0] = d0;
    tensor->dim[1] = d1;
    tensor->dim[2] = d2;
    tensor->type = type;
    tensor->bytes = bytes;
    tensor->elements = d0 * (ndim > 1u ? d1 : 1u) *
                       (ndim > 2u ? d2 : 1u);
    return tensor;
}

static uint64_t test_quant_bytes(uint32_t type, uint64_t in_dim,
                                 uint64_t out_dim, uint64_t experts) {
    const uint64_t block_bytes = type == DS4_TENSOR_Q8_0 ? 34u : 18u;
    assert(in_dim % 32u == 0);
    return experts * out_dim * (in_dim / 32u) * block_bytes;
}

static ds4_tensor *test_f32(uint32_t ndim, uint64_t d0,
                            uint64_t d1, uint64_t d2) {
    return test_tensor(DS4_TENSOR_F32, ndim, d0, d1, d2,
        d0 * (ndim > 1u ? d1 : 1u) *
        (ndim > 2u ? d2 : 1u) * sizeof(float));
}

static ds4_tensor *test_q8(uint64_t in_dim, uint64_t out_dim) {
    return test_tensor(DS4_TENSOR_Q8_0, 2, in_dim, out_dim, 0,
        test_quant_bytes(DS4_TENSOR_Q8_0, in_dim, out_dim, 1u));
}

static void make_test_plan(ds4_inkling_plan *plan) {
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

static void make_test_config(void) {
    memset(&g_ds4_inkling_model_config, 0,
           sizeof(g_ds4_inkling_model_config));
    ds4_inkling_model_config *config = &g_ds4_inkling_model_config;
    config->physical_layers = DS4_INKLING_PHYSICAL_LAYERS;
    config->embedding_dim = TEST_HIDDEN;
    config->vocab_padded = TEST_VOCAB_PADDED;
    config->vocab_unpadded = TEST_VOCAB;
    config->head_dim = 128u;
    config->expert_count = 128u;
    config->experts_used = 8u;
    config->expert_ffn_dim = 1152u;
    config->context_length = DS4_INKLING_CONTEXT_MAX;
    config->loop_start = DS4_INKLING_LOOP_START;
    config->loop_end = DS4_INKLING_LOOP_END;
    config->loop_passes = DS4_INKLING_LOOP_PASSES;
    config->gate_rank = 64u;
    config->residual_block_size = DS4_INKLING_RESIDUAL_BLOCK;
    config->sliding_window = DS4_INKLING_SLIDING_WINDOW;
    config->shortconv_cache = DS4_INKLING_CONV_KERNEL;
    config->relative_dim = 16u;
    config->relative_extent = 1024u;
    config->log_scale_floor = 8192u;
    config->rms_epsilon = 1.0e-6f;
    config->qk_clip = 22.627416610717773f;
    config->relative_clip = 8.0f;
    config->log_scale_alpha = 0.1f;
    config->expert_weight_scale = 8.0f;
    config->situ_beta = 4.0f;
    config->situ_linear_beta = 25.0f;
    config->final_logit_softcap = 30.0f;
    for (uint32_t i = 0; i < DS4_INKLING_PHYSICAL_LAYERS; i++) {
        config->query_heads[i] = 24u;
        config->kv_heads[i] = 8u;
        config->sliding_window_pattern[i] = (uint8_t)(i % 6u != 5u);
    }
}

static void make_test_weights(ds4_engine *engine) {
    ds4_inkling_weights *weights = &engine->inkling_weights;
    memset(weights, 0, sizeof(*weights));
    weights->token_embd = test_q8(TEST_HIDDEN, TEST_VOCAB_PADDED);
    weights->token_embd_norm = test_f32(1, TEST_HIDDEN, 0, 0);
    weights->output = test_q8(TEST_HIDDEN, TEST_VOCAB_PADDED);
    weights->output_norm = test_f32(1, TEST_HIDDEN, 0, 0);
    weights->output_res_query = test_f32(1, TEST_HIDDEN, 0, 0);

    for (uint32_t il = 0; il < DS4_INKLING_PHYSICAL_LAYERS; il++) {
        ds4_inkling_layer_weights *layer = &weights->layer[il];
        const bool sliding = il % 6u != 5u;
        const uint32_t expert_type = il < 20u ?
            DS4_TENSOR_IQ4_NL : DS4_TENSOR_Q8_0;
        layer->attn_q = test_q8(TEST_HIDDEN, 24u * 128u);
        layer->attn_k = test_q8(TEST_HIDDEN, DS4_INKLING_KV_WIDTH);
        layer->attn_v = test_q8(TEST_HIDDEN, DS4_INKLING_KV_WIDTH);
        layer->attn_output = test_q8(24u * 128u, TEST_HIDDEN);
        layer->attn_gate = test_q8(TEST_HIDDEN, 24u * 128u);
        layer->attn_q_clip = test_f32(2, 128u, 24u, 0);
        layer->attn_sink_proj = test_q8(TEST_HIDDEN, 24u);
        layer->attn_rel = test_q8(TEST_HIDDEN, 24u * 16u);
        layer->attn_rel_bias = test_f32(2, 16u, 1024u, 0);
        layer->attn_rel_clip = test_f32(2, 16u, 24u, 0);
        layer->attn_k_conv = test_f32(2, 4u, DS4_INKLING_KV_WIDTH, 0);
        layer->attn_v_conv = test_f32(2, 4u, DS4_INKLING_KV_WIDTH, 0);
        layer->attn_post_conv = test_f32(2, 4u, TEST_HIDDEN, 0);
        layer->ffn_post_conv = test_f32(2, 4u, TEST_HIDDEN, 0);
        if (sliding) {
            layer->attn_norm = test_f32(1, TEST_HIDDEN, 0, 0);
        } else {
            layer->attn_q_inp_norm = test_f32(1, TEST_HIDDEN, 0, 0);
            layer->attn_k_inp_norm = test_f32(1, TEST_HIDDEN, 0, 0);
            layer->attn_v_inp_norm = test_f32(1, TEST_HIDDEN, 0, 0);
            layer->attn_k_conv_proj = test_q8(TEST_HIDDEN, 32u);
            layer->attn_v_conv_proj = test_q8(TEST_HIDDEN, 32u);
        }
        if (il != 0u)
            layer->attn_res_query = test_f32(2, TEST_HIDDEN,
                                             sliding ? 1u : 3u, 0);
        layer->ffn_res_query = test_f32(1, TEST_HIDDEN, 0, 0);
        layer->ffn_norm = test_f32(1, TEST_HIDDEN, 0, 0);
        layer->ffn_gate_inp = test_f32(2, TEST_HIDDEN, 128u, 0);
        layer->ffn_exp_probs_b = test_f32(1, 128u, 0, 0);
        layer->ffn_gate_up_exps = test_tensor(expert_type, 3,
            TEST_HIDDEN, TEST_HIDDEN, 128u,
            test_quant_bytes(expert_type, TEST_HIDDEN, TEST_HIDDEN, 128u));
        layer->ffn_down_exps = test_tensor(expert_type, 3,
            1152u, TEST_HIDDEN, 128u,
            test_quant_bytes(expert_type, 1152u, TEST_HIDDEN, 128u));
        if (il >= DS4_INKLING_LOOP_START && il < DS4_INKLING_LOOP_END) {
            layer->loop_msg_query = test_f32(2, TEST_HIDDEN, 4u, 0);
            for (uint32_t ctrl = 0; ctrl < DS4_INKLING_CTRL_COUNT; ctrl++) {
                layer->loop_ctrl_down[ctrl] = test_f32(2, 4608u, 64u, 0);
                layer->loop_ctrl_gain[ctrl] = test_f32(2, 64u, TEST_HIDDEN, 0);
                layer->loop_ctrl_forget[ctrl] = test_f32(2, 64u, TEST_HIDDEN, 0);
            }
        }
    }
}

static void verify_residual_query_order(ds4_session *session) {
    const uint64_t row_bytes = (uint64_t)TEST_HIDDEN * sizeof(float);
    float *banks = malloc(2u * (size_t)row_bytes);
    float *current = malloc((size_t)row_bytes);
    float *query = malloc((size_t)row_bytes);
    float *expected = malloc((size_t)row_bytes);
    float *actual = malloc((size_t)row_bytes);
    float states[3u * TEST_HIDDEN];
    assert(banks && current && query && expected && actual);
    for (uint32_t i = 0; i < TEST_HIDDEN; i++) {
        banks[i] = sinf((float)i * 0.013f);
        banks[TEST_HIDDEN + i] = cosf((float)i * 0.019f);
        current[i] = sinf((float)i * 0.007f + 0.4f);
        query[i] = cosf((float)i * 0.003f - 0.2f);
        states[i] = banks[TEST_HIDDEN + i];
        states[TEST_HIDDEN + i] = banks[i];
        states[2u * TEST_HIDDEN + i] = current[i];
    }
    ds4_inkling_gpu_context *context = &session->inkling_gpu;
    assert(ds4_gpu_tensor_write(context->current_blocks, 0, banks,
                                2u * row_bytes));
    assert(ds4_gpu_tensor_write(context->prelude_output, 0, current, row_bytes));
    assert(ds4_gpu_tensor_write(context->parameters.output_res_query, 0,
                                query, row_bytes));
    assert(ds4_inkling_mix_ref(expected, states, query, 3u,
                               TEST_HIDDEN, 1u, 1.0e-6f));
    assert(ds4_gpu_begin_commands());
    assert(inkling_gpu_mix(context, context->prelude_output, 2u,
            context->parameters.output_res_query, 1u,
            context->layer.mix_output_rows[0]));
    assert(ds4_gpu_end_commands());
    assert(ds4_gpu_tensor_read(context->layer.mix_output_rows[0], 0,
                               actual, row_bytes));
    float max_error = 0.0f;
    for (uint32_t i = 0; i < TEST_HIDDEN; i++) {
        const float error = fabsf(expected[i] - actual[i]);
        if (error > max_error) max_error = error;
    }
    assert(max_error <= 2.0e-4f);
    printf("Inkling residual query order max error: %.8g\n", max_error);
    free(banks);
    free(current);
    free(query);
    free(expected);
    free(actual);
}

static void verify_previous_loop_message_states(ds4_session *session) {
    ds4_inkling_gpu_context *context = &session->inkling_gpu;
    const uint64_t row_bytes = (uint64_t)TEST_HIDDEN * sizeof(float);
    const uint64_t messages_bytes = 4u * row_bytes;
    float *current_block = malloc((size_t)row_bytes);
    float *previous_states = malloc((size_t)(2u * row_bytes));
    float *queries = calloc(4u * TEST_HIDDEN, sizeof(float));
    float *expected = malloc((size_t)messages_bytes);
    float *actual = malloc((size_t)messages_bytes);
    assert(current_block && previous_states && queries && expected && actual);
    for (uint32_t i = 0; i < TEST_HIDDEN; i++) {
        current_block[i] = sinf((float)i * 0.017f + 0.6f);
        previous_states[i] = cosf((float)i * 0.011f - 0.2f);
        previous_states[TEST_HIDDEN + i] = sinf((float)i * 0.023f - 0.8f);
    }
    assert(ds4_gpu_tensor_write(context->current_blocks, 0,
                                current_block, row_bytes));
    assert(ds4_gpu_tensor_write(context->layer.previous_loop_states, 0,
                                previous_states, 2u * row_bytes));
    assert(ds4_gpu_tensor_write(context->parameters.layer[6].loop_msg_query, 0,
                                queries, messages_bytes));
    assert(ds4_inkling_mix_ref(expected, previous_states, queries, 2u,
                               TEST_HIDDEN, 4u, 1.0e-6f));
    const ds4_inkling_visit *visit = NULL;
    for (uint32_t i = 0; i < DS4_INKLING_VISITS; i++) {
        const ds4_inkling_visit *candidate = &context->plan.visits[i];
        if (candidate->phase == DS4_INKLING_PHASE_CORE &&
            candidate->pass == 1u && candidate->physical_layer == 6u) {
            visit = candidate;
            break;
        }
    }
    assert(visit);
    assert(ds4_gpu_begin_commands());
    ds4_gpu_tensor *out = NULL;
    const char *failed_stage = NULL;
    const bool evaluated = inkling_gpu_decoder_layer(
        session, visit, 6u, 1u, context->prelude_output,
        &out, &failed_stage);
    assert(ds4_gpu_end_commands());
    if (!evaluated)
        fprintf(stderr, "loop message integration failed at %s\n",
                failed_stage ? failed_stage : "unknown stage");
    assert(evaluated);
    assert(ds4_gpu_tensor_read(context->layer.messages, 0,
                               actual, messages_bytes));
    float max_error = 0.0f;
    for (uint32_t i = 0; i < 4u * TEST_HIDDEN; i++) {
        const float error = fabsf(expected[i] - actual[i]);
        if (error > max_error) max_error = error;
    }
    printf("Inkling previous-pass message max error: %.8g\n", max_error);
    assert(max_error <= 2.0e-4f);
    ds4_inkling_gpu_context_reset(context);
    free(current_block);
    free(previous_states);
    free(queries);
    free(expected);
    free(actual);
}

static int logits_argmax(const float *logits, uint32_t count) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < count; i++)
        if (logits[i] > logits[best]) best = i;
    return (int)best;
}

int main(void) {
    assert(ds4_gpu_init());
    g_ds4_shape = DS4_SHAPE_INKLING2;
    make_test_config();

    uint8_t *model_map = mmap(NULL, (size_t)TEST_MODEL_BYTES,
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    assert(model_map != MAP_FAILED);

    ds4_engine engine = {0};
    engine.backend = DS4_BACKEND_METAL;
    engine.metal_ready = true;
    engine.model.map = model_map;
    engine.model.size = TEST_MODEL_BYTES;
    engine.model.file_size = TEST_MODEL_BYTES;
    engine.model.max_tensor_bytes = TEST_MODEL_BYTES;
    make_test_plan(&engine.inkling_plan);
    make_test_weights(&engine);
    assert(ds4_gpu_set_model_map_range(model_map, TEST_MODEL_BYTES, 0,
            TEST_MODEL_BYTES, TEST_MODEL_BYTES));

    ds4_session *session = NULL;
    assert(ds4_session_create(&session, &engine, 8193) == 0);
    assert(session && session->inkling_gpu_ready);
    assert(session->ctx_size == 8193);
    assert(session->inkling_gpu.weight_owner_count ==
           DS4_INKLING_PHYSICAL_LAYERS);
    verify_residual_query_order(session);
    verify_previous_loop_message_states(session);

    const int short_prompt[] = { 42, 17 };
    const int full_prompt[] = { 42, 17, 9 };
    const ds4_tokens first = { .v = (int *)short_prompt, .len = 2, .cap = 2 };
    const ds4_tokens full = { .v = (int *)full_prompt, .len = 3, .cap = 3 };
    char error[256] = {0};
    assert(ds4_session_sync(session, &first, error, sizeof(error)) == 0);
    assert(ds4_session_pos(session) == 2);
    assert(ds4_session_checkpoint_valid(session));
    assert(logits_argmax(session->logits, TEST_VOCAB) == 0);
    assert(session->logits[TEST_VOCAB] == DS4_NEG_INF);
    assert(ds4_session_argmax(session) == 0);
    float *public_logits = malloc((size_t)TEST_VOCAB * sizeof(float));
    assert(public_logits);
    assert(ds4_session_copy_logits(session, public_logits, TEST_VOCAB) ==
           TEST_VOCAB);
    free(public_logits);
    for (uint32_t i = 0; i < TEST_VOCAB; i++)
        assert(isfinite(session->logits[i]) && session->logits[i] == 0.0f);

    assert(ds4_session_sync(session, &full, error, sizeof(error)) == 0);
    assert(ds4_session_pos(session) == 3);
    float *continued_logits = malloc((size_t)TEST_VOCAB * sizeof(float));
    assert(continued_logits);
    memcpy(continued_logits, session->logits,
           (size_t)TEST_VOCAB * sizeof(float));
    assert(ds4_session_layer_slice_reset(session, error, sizeof(error)) == 0);
    assert(ds4_session_pos(session) == 0);
    assert(!ds4_session_checkpoint_valid(session));
    assert(ds4_session_sync(session, &full, error, sizeof(error)) == 0);
    assert(ds4_session_pos(session) == 3);
    assert(memcmp(continued_logits, session->logits,
                  (size_t)TEST_VOCAB * sizeof(float)) == 0);
    free(continued_logits);

    assert(ds4_session_eval(session, 0, error, sizeof(error)) == 0);
    assert(ds4_session_pos(session) == 4);
    assert(session->logits[TEST_VOCAB] == DS4_NEG_INF);
    assert(logits_argmax(session->logits, TEST_VOCAB) == 0);
    assert(ds4_session_eval_argmax(session, 0, error, sizeof(error)) == 0);
    assert(ds4_session_pos(session) == 5);
    assert(session->logits[TEST_VOCAB] == DS4_NEG_INF);
    assert(logits_argmax(session->logits, TEST_VOCAB) == 0);

    ds4_session_free(session);
    ds4_gpu_cleanup();
    assert(munmap(model_map, (size_t)TEST_MODEL_BYTES) == 0);
    puts("Inkling2 native Metal session decode/prefill/reset: ok");
    return 0;
}
