/* A sparse GGUF exercises DS4's real metadata and tensor binders without
 * allocating or reading the multi-gigabyte tensor payload. */
#include "../ds4.c"
#include <assert.h>
#include <sys/wait.h>

enum {
    TEST_ALIGNMENT = 32,
    TEST_TENSOR_COUNT = 704,
    TEST_METADATA_COUNT = 46,
    TEST_HIDDEN = 2304,
    TEST_VOCAB_PADDED = 201024,
    TEST_EXPERTS = 128,
};

static uint32_t descriptor_count;
static uint64_t payload_end;

static void put_u32(FILE *file, uint32_t value) {
    assert(fwrite(&value, sizeof(value), 1, file) == 1);
}

static void put_u64(FILE *file, uint64_t value) {
    assert(fwrite(&value, sizeof(value), 1, file) == 1);
}

static void put_f32(FILE *file, float value) {
    assert(fwrite(&value, sizeof(value), 1, file) == 1);
}

static void put_string(FILE *file, const char *value) {
    const size_t length = strlen(value);
    put_u64(file, (uint64_t)length);
    assert(fwrite(value, 1, length, file) == length);
}

static void put_metadata_key(FILE *file, const char *key, uint32_t type) {
    put_string(file, key);
    put_u32(file, type);
}

static void metadata_string(FILE *file, const char *key, const char *value) {
    put_metadata_key(file, key, GGUF_VALUE_STRING);
    put_string(file, value);
}

static void metadata_u32(FILE *file, const char *key, uint32_t value) {
    put_metadata_key(file, key, GGUF_VALUE_UINT32);
    put_u32(file, value);
}

static void metadata_u64(FILE *file, const char *key, uint64_t value) {
    put_metadata_key(file, key, GGUF_VALUE_UINT64);
    put_u64(file, value);
}

static void metadata_f32(FILE *file, const char *key, float value) {
    put_metadata_key(file, key, GGUF_VALUE_FLOAT32);
    put_f32(file, value);
}

static void metadata_bool(FILE *file, const char *key, bool value) {
    const uint8_t encoded = value ? 1u : 0u;
    put_metadata_key(file, key, GGUF_VALUE_BOOL);
    assert(fwrite(&encoded, sizeof(encoded), 1, file) == 1);
}

static void metadata_i32_array(FILE *file, const char *key,
                               const int32_t *values, uint64_t count) {
    put_metadata_key(file, key, GGUF_VALUE_ARRAY);
    put_u32(file, GGUF_VALUE_INT32);
    put_u64(file, count);
    for (uint64_t i = 0; i < count; i++) {
        assert(fwrite(&values[i], sizeof(values[i]), 1, file) == 1);
    }
}

static void metadata_bool_array(FILE *file, const char *key,
                                const uint8_t *values, uint64_t count) {
    put_metadata_key(file, key, GGUF_VALUE_ARRAY);
    put_u32(file, GGUF_VALUE_BOOL);
    put_u64(file, count);
    assert(fwrite(values, sizeof(values[0]), (size_t)count, file) == count);
}

static void metadata_string_array(FILE *file, const char *key,
                                  uint64_t count, const char *value) {
    put_metadata_key(file, key, GGUF_VALUE_ARRAY);
    put_u32(file, GGUF_VALUE_STRING);
    put_u64(file, count);
    for (uint64_t i = 0; i < count; i++) put_string(file, value);
}

static void metadata_inkling_tokens(FILE *file) {
    static const struct {
        uint32_t id;
        const char *text;
    } specials[] = {
        {199999u, "<|endoftext|>"},
        {200000u, "<|message_user|>"},
        {200001u, "<|message_model|>"},
        {200002u, "<|message_system|>"},
        {200003u, "<|message_tool|>"},
        {200004u, "<|content_text|>"},
        {200006u, "<|content_model_end_sampling|>"},
        {200008u, "<|content_thinking|>"},
        {200010u, "<|end_message|>"},
    };
    put_metadata_key(file, "tokenizer.ggml.tokens", GGUF_VALUE_ARRAY);
    put_u32(file, GGUF_VALUE_STRING);
    put_u64(file, 200058u);
    size_t next = 0;
    for (uint32_t id = 0; id < 200058u; id++) {
        const char *text = "x";
        if (next < sizeof(specials) / sizeof(specials[0]) &&
            id == specials[next].id) {
            text = specials[next].text;
            next++;
        }
        put_string(file, text);
    }
    assert(next == sizeof(specials) / sizeof(specials[0]));
}

static uint64_t tensor_bytes(uint32_t type, const uint64_t *dimensions,
                             uint32_t ndim) {
    uint64_t elements = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        assert(dimensions[i] != 0 && elements <= UINT64_MAX / dimensions[i]);
        elements *= dimensions[i];
    }
    uint64_t bytes = 0;
    assert(tensor_nbytes(type, elements, &bytes));
    return bytes;
}

static void tensor_descriptor(FILE *file, const char *name, uint32_t type,
                              uint32_t ndim, uint64_t d0, uint64_t d1,
                              uint64_t d2) {
    const uint64_t dimensions[3] = {d0, d1, d2};
    const uint64_t bytes = tensor_bytes(type, dimensions, ndim);
    payload_end = align_up(payload_end, TEST_ALIGNMENT);
    put_string(file, name);
    put_u32(file, ndim);
    for (uint32_t i = 0; i < ndim; i++) put_u64(file, dimensions[i]);
    put_u32(file, type);
    put_u64(file, payload_end);
    assert(payload_end <= UINT64_MAX - bytes);
    payload_end += bytes;
    descriptor_count++;
}

static void layer_tensor(FILE *file, uint32_t layer, const char *suffix,
                         uint32_t type, uint32_t ndim,
                         uint64_t d0, uint64_t d1, uint64_t d2) {
    char name[128];
    assert(snprintf(name, sizeof(name), "blk.%u.%s", layer, suffix) > 0);
    tensor_descriptor(file, name, type, ndim, d0, d1, d2);
}

static void write_model_metadata(FILE *file) {
    static const char *const p = "inkling2_loop.";
    char key[128];
#define META_U32(suffix, value) do { \
    snprintf(key, sizeof(key), "%s%s", p, suffix); \
    metadata_u32(file, key, value); \
} while (0)
#define META_F32(suffix, value) do { \
    snprintf(key, sizeof(key), "%s%s", p, suffix); \
    metadata_f32(file, key, value); \
} while (0)
    metadata_string(file, "general.architecture", "inkling2_loop");
    metadata_u32(file, "general.alignment", TEST_ALIGNMENT);
    META_U32("block_count", 24);
    META_U32("embedding_length", TEST_HIDDEN);
    META_U32("vocab_size", TEST_VOCAB_PADDED);
    META_U32("vocab_size_unpadded", 200058);
    snprintf(key, sizeof(key), "%scontext_length", p);
    metadata_u64(file, key, 1048576);
    META_U32("expert_count", TEST_EXPERTS);
    META_U32("expert_used_count", 8);
    META_U32("expert_feed_forward_length", 1152);
    META_U32("loop.start", 6);
    META_U32("loop.end", 18);
    META_U32("loop.count", 3);
    META_U32("loop.gate_rank", 64);
    snprintf(key, sizeof(key), "%sloop.varying", p);
    metadata_bool(file, key, false);
    META_U32("attn_res.block_size", 6);
    META_U32("attention.sliding_window", 1024);
    META_U32("shortconv.l_cache", 4);
    META_U32("attention.relative_length", 16);
    META_U32("attention.relative_extent", 1024);
    META_U32("attention.log_scale_floor", 8192);
    META_F32("attention.layer_norm_rms_epsilon", 1.0e-6f);
    META_F32("attention.qk_clip", 22.627416610717773f);
    META_F32("attention.relative_clip", 8.0f);
    META_F32("attention.log_scale_alpha", 0.1f);
    META_F32("expert_weights_scale", 8.0f);
    META_F32("activation.situ_beta", 4.0f);
    META_F32("activation.situ_linear_beta", 25.0f);
    META_F32("final_logit_softcapping", 30.0f);
    META_U32("attention.key_length", 128);
    META_U32("attention.value_length", 128);
    META_U32("attention.key_length_swa", 128);
    META_U32("attention.value_length_swa", 128);
    META_U32("feed_forward_length", 1152);
    META_U32("rope.dimension_count", 0);
    int32_t heads[24], kv_heads[24];
    uint8_t sliding[24];
    for (uint32_t i = 0; i < 24; i++) {
        heads[i] = 24;
        kv_heads[i] = 8;
        sliding[i] = (uint8_t)(i % 6u != 5u);
    }
    snprintf(key, sizeof(key), "%sattention.head_count", p);
    metadata_i32_array(file, key, heads, 24);
    snprintf(key, sizeof(key), "%sattention.head_count_kv", p);
    metadata_i32_array(file, key, kv_heads, 24);
    snprintf(key, sizeof(key), "%sattention.sliding_window_pattern", p);
    metadata_bool_array(file, key, sliding, 24);
    metadata_string(file, "tokenizer.ggml.pre", "inkling2");
    metadata_string(file, "tokenizer.ggml.model", "gpt2");
    metadata_inkling_tokens(file);
    metadata_string_array(file, "tokenizer.ggml.merges", 424413, "");
    metadata_bool(file, "tokenizer.ggml.add_bos_token", false);
    metadata_bool(file, "tokenizer.ggml.add_eos_token", false);
    metadata_u32(file, "tokenizer.ggml.eos_token_id", 200006);
    metadata_u32(file, "tokenizer.ggml.padding_token_id", 199999);
#undef META_U32
#undef META_F32
}

static void write_tensor_directory(FILE *file) {
    static const char *const ctrl_names[4] = {"q", "k", "v", "ffn"};
    char name[128];
    tensor_descriptor(file, "token_embd.weight", DS4_TENSOR_Q8_0, 2,
                      TEST_HIDDEN, TEST_VOCAB_PADDED, 0);
    tensor_descriptor(file, "token_embd_norm.weight", DS4_TENSOR_F32, 1,
                      TEST_HIDDEN, 0, 0);
    tensor_descriptor(file, "output.weight", DS4_TENSOR_Q8_0, 2,
                      TEST_HIDDEN, TEST_VOCAB_PADDED, 0);
    tensor_descriptor(file, "output_norm.weight", DS4_TENSOR_F32, 1,
                      TEST_HIDDEN, 0, 0);
    tensor_descriptor(file, "output_res_query.weight", DS4_TENSOR_F32, 1,
                      TEST_HIDDEN, 0, 0);

    for (uint32_t il = 0; il < 24; il++) {
        const bool sliding = il % 6u != 5u;
        const uint32_t expert_type = il < 20 ? DS4_TENSOR_IQ4_NL : DS4_TENSOR_Q8_0;
        layer_tensor(file, il, "attn_q.weight", DS4_TENSOR_Q8_0, 2, 2304, 3072, 0);
        layer_tensor(file, il, "attn_k.weight", DS4_TENSOR_Q8_0, 2, 2304, 1024, 0);
        layer_tensor(file, il, "attn_v.weight", DS4_TENSOR_Q8_0, 2, 2304, 1024, 0);
        layer_tensor(file, il, "attn_output.weight", DS4_TENSOR_Q8_0, 2, 3072, 2304, 0);
        layer_tensor(file, il, "attn_gate.weight", DS4_TENSOR_Q8_0, 2, 2304, 3072, 0);
        layer_tensor(file, il, "attn_q_clip.weight", DS4_TENSOR_F32, 2, 128, 24, 0);
        layer_tensor(file, il, "attn_sink_proj.weight", DS4_TENSOR_Q8_0, 2, 2304, 24, 0);
        layer_tensor(file, il, "attn_rel.weight", DS4_TENSOR_Q8_0, 2, 2304, 384, 0);
        layer_tensor(file, il, "attn_rel_bias.weight", DS4_TENSOR_F32, 2, 16, 1024, 0);
        layer_tensor(file, il, "attn_rel_clip.weight", DS4_TENSOR_F32, 2, 16, 24, 0);
        layer_tensor(file, il, "attn_k_conv.weight", DS4_TENSOR_F32, 2, 4, 1024, 0);
        layer_tensor(file, il, "attn_v_conv.weight", DS4_TENSOR_F32, 2, 4, 1024, 0);
        layer_tensor(file, il, "attn_post_conv.weight", DS4_TENSOR_F32, 2, 4, 2304, 0);
        layer_tensor(file, il, "ffn_post_conv.weight", DS4_TENSOR_F32, 2, 4, 2304, 0);
        if (sliding) {
            layer_tensor(file, il, "attn_norm.weight", DS4_TENSOR_F32, 1, 2304, 0, 0);
        } else {
            layer_tensor(file, il, "attn_q_inp_norm.weight", DS4_TENSOR_F32, 1, 2304, 0, 0);
            layer_tensor(file, il, "attn_k_inp_norm.weight", DS4_TENSOR_F32, 1, 2304, 0, 0);
            layer_tensor(file, il, "attn_v_inp_norm.weight", DS4_TENSOR_F32, 1, 2304, 0, 0);
            layer_tensor(file, il, "attn_k_conv_proj.weight", DS4_TENSOR_Q8_0, 2, 2304, 32, 0);
            layer_tensor(file, il, "attn_v_conv_proj.weight", DS4_TENSOR_Q8_0, 2, 2304, 32, 0);
        }
        if (il != 0)
            layer_tensor(file, il, "attn_res_query.weight", DS4_TENSOR_F32,
                         sliding ? 1u : 2u,
                         2304, sliding ? 0u : 3u, 0);
        layer_tensor(file, il, "ffn_res_query.weight", DS4_TENSOR_F32, 1, 2304, 0, 0);
        layer_tensor(file, il, "ffn_norm.weight", DS4_TENSOR_F32, 1, 2304, 0, 0);
        layer_tensor(file, il, "ffn_gate_inp.weight", DS4_TENSOR_F32, 2, 2304, 128, 0);
        layer_tensor(file, il, "exp_probs_b.bias", DS4_TENSOR_F32, 1, 128, 0, 0);
        layer_tensor(file, il, "ffn_gate_up_exps.weight", expert_type, 3, 2304, 2304, 128);
        layer_tensor(file, il, "ffn_down_exps.weight", expert_type, 3, 1152, 2304, 128);
        if (il >= 6 && il < 18) {
            layer_tensor(file, il, "loop_msg_query.weight", DS4_TENSOR_F32, 2, 2304, 4, 0);
            for (uint32_t ctrl = 0; ctrl < 4; ctrl++) {
                snprintf(name, sizeof(name), "loop_%s_ctrl_down.weight", ctrl_names[ctrl]);
                layer_tensor(file, il, name, DS4_TENSOR_F32, 2, 4608, 64, 0);
                snprintf(name, sizeof(name), "loop_%s_ctrl_gain.weight", ctrl_names[ctrl]);
                layer_tensor(file, il, name, DS4_TENSOR_F32, 2, 64, 2304, 0);
                snprintf(name, sizeof(name), "loop_%s_ctrl_forget.weight", ctrl_names[ctrl]);
                layer_tensor(file, il, name, DS4_TENSOR_F32, 2, 64, 2304, 0);
            }
        }
    }
}

static void write_sparse_gguf(const char *path) {
    descriptor_count = 0;
    payload_end = 0;
    int fd = mkstemp((char *)path);
    assert(fd >= 0);
    FILE *file = fdopen(fd, "w+b");
    assert(file);
    put_u32(file, DS4_GGUF_MAGIC);
    put_u32(file, 3);
    put_u64(file, TEST_TENSOR_COUNT);
    put_u64(file, TEST_METADATA_COUNT);
    write_model_metadata(file);
    write_tensor_directory(file);
    assert(descriptor_count == TEST_TENSOR_COUNT);
    assert(fflush(file) == 0);
    const uint64_t data_offset = align_up((uint64_t)ftello(file), TEST_ALIGNMENT);
    assert(data_offset <= UINT64_MAX - payload_end);
    const uint64_t file_size = data_offset + payload_end;
    assert(ftruncate(fd, (off_t)file_size) == 0);
    assert(fclose(file) == 0);
}

static void expect_bind_failure(const ds4_model *source, bool bad_count,
                                const char *bad_tensor, bool bad_shape) {
    const pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        ds4_model model = *source;
        if (bad_count) model.n_tensors--;
        if (bad_tensor) {
            ds4_tensor *tensor = model_find_tensor(&model, bad_tensor);
            if (!tensor) _exit(2);
            if (bad_shape) tensor->dim[0]++;
            else tensor->type = DS4_TENSOR_F32;
        }
        ds4_inkling_weights weights;
        inkling_weights_bind(&weights, &model);
        _exit(0);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) != 0);
}

int main(void) {
    char path[] = "/tmp/ds4-inkling2-gguf.XXXXXX";
    write_sparse_gguf(path);

    ds4_model model;
    model_open(&model, path, false, false);
    config_validate_model(&model);
    assert(ds4_model_is_inkling2());
    assert(g_ds4_inkling_model_config.context_length == 1048576);
    assert(model.n_tensors == TEST_TENSOR_COUNT);

    ds4_inkling_weights weights;
    inkling_weights_bind(&weights, &model);
    assert(weights.token_embd && weights.output && weights.output_res_query);
    assert(weights.layer[0].attn_res_query == NULL);
    assert(weights.layer[0].attn_norm && !weights.layer[0].attn_q_inp_norm);
    assert(weights.layer[5].attn_q_inp_norm && !weights.layer[5].attn_norm);
    assert(weights.layer[0].attn_k_conv_proj == NULL);
    assert(weights.layer[5].attn_k_conv_proj && weights.layer[5].attn_v_conv_proj);
    assert(weights.layer[19].ffn_gate_up_exps->type == DS4_TENSOR_IQ4_NL);
    assert(weights.layer[20].ffn_gate_up_exps->type == DS4_TENSOR_Q8_0);
    for (uint32_t i = 0; i < 24; i++) {
        const bool loop_core = i >= 6 && i < 18;
        for (uint32_t ctrl = 0; ctrl < DS4_INKLING_CTRL_COUNT; ctrl++) {
            assert((weights.layer[i].loop_ctrl_down[ctrl] != NULL) == loop_core);
            assert((weights.layer[i].loop_ctrl_gain[ctrl] != NULL) == loop_core);
            assert((weights.layer[i].loop_ctrl_forget[ctrl] != NULL) == loop_core);
        }
    }

    expect_bind_failure(&model, true, NULL, false);
    expect_bind_failure(&model, false, "blk.0.attn_q.weight", false);
    expect_bind_failure(&model, false, "blk.5.attn_k_conv_proj.weight", true);
    model_close(&model);

    ds4_engine_options options = {0};
    options.model_path = path;
    options.backend = DS4_BACKEND_CPU;
    options.inspect_only = true;
    ds4_engine *engine = NULL;
    assert(ds4_engine_open(&engine, &options) == 0 && engine);
    assert(engine->inkling_weights.layer[17].loop_msg_query);
    assert(engine->inkling_weights.layer[18].loop_msg_query == NULL);
    assert(engine->inkling_plan.visit_count == DS4_INKLING_VISITS);
    assert(engine->inkling_plan.physical_layer_count == 24);
    assert(engine->inkling_plan.cache_slot_count == 48);
    assert(engine->inkling_plan.visits[18].physical_layer == 6);
    assert(engine->inkling_plan.visits[18].cache_slot == 24);
    ds4_tokens rendered = {0};
    ds4_tokenize_rendered_chat(
        engine,
        "<|message_user|><|content_text|><|end_message|>",
        &rendered);
    assert(rendered.len == 3);
    assert(rendered.v[0] == 200000);
    assert(rendered.v[1] == 200004);
    assert(rendered.v[2] == 200010);
    ds4_tokens_free(&rendered);
    bool seen_cache[DS4_INKLING_CACHE_SLOTS] = {0};
    for (uint32_t i = 0; i < engine->inkling_plan.visit_count; i++) {
        const ds4_inkling_visit *visit = &engine->inkling_plan.visits[i];
        assert(visit->physical_layer < DS4_INKLING_PHYSICAL_LAYERS);
        assert(visit->cache_slot < DS4_INKLING_CACHE_SLOTS);
        assert(!seen_cache[visit->cache_slot]);
        seen_cache[visit->cache_slot] = true;
    }
    for (uint32_t i = 0; i < DS4_INKLING_CORE_LAYERS; i++) {
        const ds4_inkling_visit *pass0 = &engine->inkling_plan.visits[6 + i];
        const ds4_inkling_visit *pass1 = &engine->inkling_plan.visits[18 + i];
        const ds4_inkling_visit *pass2 = &engine->inkling_plan.visits[30 + i];
        assert(pass0->physical_layer == pass1->physical_layer);
        assert(pass1->physical_layer == pass2->physical_layer);
        assert(pass0->cache_slot != pass1->cache_slot);
        assert(pass1->cache_slot != pass2->cache_slot);
        assert(pass0->cache_slot != pass2->cache_slot);
    }
    ds4_engine_close(engine);

    options.inspect_only = false;
    engine = NULL;
    assert(ds4_engine_open(&engine, &options) == 1 && engine == NULL);
    assert(unlink(path) == 0);
    puts("Inkling2 GGUF metadata and 704-tensor binding: ok");
    return 0;
}
