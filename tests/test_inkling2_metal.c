#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../ds4_inkling_moe.h"
#include "../ds4_inkling_loop.h"
#include "../ds4_inkling_state.h"
#include "../ds4_inkling_gpu_state.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void *g_synthetic_model_map;
static uint64_t g_synthetic_model_size;
static void *g_synthetic_moe_map;
static uint64_t g_synthetic_moe_size;
static void *g_full_shape_iq4_map;
static void *g_synthetic_loop_map;
static ds4_inkling_gpu_context g_operator_context;
static float g_m4_qk_relative_max_error;
static float g_m4_attention_max_error;
static float g_m4_full_attention_max_error;
static float g_m4_gate_max_error;
static float g_m4_conv_max_error;
static float g_m4_conv_chunk_max_error;
static float g_m4_static_conv_max_error;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "Inkling2 Metal check failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #cond); \
        ok = 0; \
        goto cleanup; \
    } \
} while (0)

static ds4_gpu_tensor *upload(const void *src, uint64_t bytes) {
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
    if (!tensor) return NULL;
    if (src && !ds4_gpu_tensor_write(tensor, 0, src, bytes)) {
        ds4_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

static int near(float actual, float expected, float tolerance) {
    return isfinite(actual) && isfinite(expected) &&
           fabsf(actual - expected) <= tolerance;
}

static int near_record(float actual, float expected, float tolerance,
                       float *max_abs_error) {
    if (!isfinite(actual) || !isfinite(expected)) return 0;
    const float error = fabsf(actual - expected);
    if (error > *max_abs_error) *max_abs_error = error;
    return error <= tolerance;
}

static float sigmoid_ref(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

static int test_rms_norm(void) {
    enum { N = 8, ROWS = 3 };
    const float x[N * ROWS] = {
        1.0f, -2.0f, 3.0f, -4.0f, 0.5f, 0.25f, -0.75f, 1.25f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        100.0f, -100.0f, 50.0f, -50.0f, 25.0f, -25.0f, 10.0f, -10.0f,
    };
    const float eps = 1e-5f;
    float actual[N * ROWS];
    ds4_gpu_tensor *tx = upload(x, sizeof(x));
    ds4_gpu_tensor *ty = ds4_gpu_tensor_alloc(sizeof(actual));
    if (!tx || !ty) {
        ds4_gpu_tensor_free(tx);
        ds4_gpu_tensor_free(ty);
        return 0;
    }
    int ok = ds4_gpu_rms_norm_plain_rows_tensor(ty, tx, N, ROWS, eps) &&
             ds4_gpu_tensor_read(ty, 0, actual, sizeof(actual));
    for (uint32_t row = 0; ok && row < ROWS; row++) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < N; col++) {
            const float v = x[row * N + col];
            sum += v * v;
        }
        const float scale = 1.0f / sqrtf(sum / (float)N + eps);
        for (uint32_t col = 0; col < N; col++) {
            if (!near(actual[row * N + col], x[row * N + col] * scale,
                      2e-5f)) {
                fprintf(stderr, "Inkling2 RMSNorm mismatch row=%u col=%u\n",
                        row, col);
                ok = 0;
                break;
            }
        }
    }
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(ty);
    return ok;
}

static int test_gate_update(void) {
    enum { N = 17 };
    float current[N], message[N], gain[N], forget[N], expected[N], actual[N];
    for (uint32_t i = 0; i < N; i++) {
        current[i] = (float)((int32_t)i - 8) / 4.0f;
        message[i] = (float)((int32_t)(i * 3u % 11u) - 5) / 3.0f;
        gain[i] = (float)(i % 5u) / 4.0f;
        forget[i] = (float)(i % 7u) / 8.0f;
        expected[i] = (1.0f - forget[i]) * current[i] + gain[i] * message[i];
    }

    ds4_gpu_tensor *tc = upload(current, sizeof(current));
    ds4_gpu_tensor *tm = upload(message, sizeof(message));
    ds4_gpu_tensor *tg = upload(gain, sizeof(gain));
    ds4_gpu_tensor *tf = upload(forget, sizeof(forget));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(actual));
    int ok = tc && tm && tg && tf && to &&
             ds4_gpu_inkling2_gate_update_tensor(to, tc, tm, tg, tf, N) &&
             ds4_gpu_tensor_read(to, 0, actual, sizeof(actual));
    for (uint32_t i = 0; ok && i < N; i++) {
        if (!near(actual[i], expected[i], 1e-6f)) {
            fprintf(stderr, "Inkling2 gate update mismatch i=%u\n", i);
            ok = 0;
        }
    }
    ds4_gpu_tensor_free(tc);
    ds4_gpu_tensor_free(tm);
    ds4_gpu_tensor_free(tg);
    ds4_gpu_tensor_free(tf);
    ds4_gpu_tensor_free(to);
    return ok;
}

static int test_l2_clip(void) {
    enum { N = 7, ROWS = 4 };
    const float x[N * ROWS] = {
        0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f,
        3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        -1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f,
    };
    const float max_norm = 2.5f;
    float expected[N * ROWS], actual[N * ROWS];
    for (uint32_t row = 0; row < ROWS; row++) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < N; col++) {
            const float v = x[row * N + col];
            sum += v * v;
        }
        const float norm = sqrtf(sum);
        const float scale = norm > max_norm ? max_norm / norm : 1.0f;
        for (uint32_t col = 0; col < N; col++) {
            expected[row * N + col] = x[row * N + col] * scale;
        }
    }

    ds4_gpu_tensor *tx = upload(x, sizeof(x));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(actual));
    int ok = tx && to &&
             ds4_gpu_inkling2_clip_l2_rows_tensor(to, tx, N, ROWS, max_norm) &&
             ds4_gpu_tensor_read(to, 0, actual, sizeof(actual));
    for (uint32_t i = 0; ok && i < N * ROWS; i++) {
        if (!near(actual[i], expected[i], 2e-5f)) {
            fprintf(stderr, "Inkling2 L2 clip mismatch i=%u\n", i);
            ok = 0;
        }
    }
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(to);
    return ok;
}

static int test_logit_softcap(void) {
    enum { N = 10 };
    const float cap = 3.25f;
    const float x[N] = {
        0.0f, 0.5f, -0.5f, cap, -cap, 2.0f * cap,
        -2.0f * cap, INFINITY, -INFINITY, NAN,
    };
    float actual[N];
    ds4_gpu_tensor *tx = upload(x, sizeof(x));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(actual));
    int ok = tx && to &&
             ds4_gpu_inkling2_logit_softcap_tensor(to, tx, N, cap) &&
             ds4_gpu_tensor_read(to, 0, actual, sizeof(actual));
    for (uint32_t i = 0; ok && i < N; i++) {
        const float expected = cap * tanhf(x[i] / cap);
        if ((isnan(expected) && !isnan(actual[i])) ||
            (!isnan(expected) && !near(actual[i], expected, 2e-5f))) {
            fprintf(stderr,
                    "Inkling2 logit softcap mismatch i=%u input=%g got=%g expected=%g\n",
                    i, x[i], actual[i], expected);
            ok = 0;
        }
    }
    if (ok && ds4_gpu_inkling2_logit_softcap_tensor(to, tx, N, 0.0f)) {
        fprintf(stderr, "Inkling2 logit softcap accepted zero cap\n");
        ok = 0;
    }
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(to);
    return ok;
}

static void router_reference(const float *logits, const float *bias,
                             uint32_t tokens, uint32_t experts,
                             uint32_t used, float routed_scale,
                             int32_t *selected, float *weights) {
    for (uint32_t token = 0; token < tokens; token++) {
        uint8_t chosen[128] = { 0 };
        float raw[8];
        float total = 0.0f;
        for (uint32_t rank = 0; rank < used; rank++) {
            int32_t best_id = -1;
            float best = -INFINITY;
            for (uint32_t expert = 0; expert < experts; expert++) {
                if (chosen[expert]) continue;
                const float score = sigmoid_ref(logits[token * experts + expert]);
                const float adjusted = score + bias[expert];
                if (best_id < 0 || adjusted > best ||
                    (adjusted == best && expert < (uint32_t)best_id)) {
                    best = adjusted;
                    best_id = (int32_t)expert;
                }
            }
            selected[token * used + rank] = best_id;
            chosen[(uint32_t)best_id] = 1;
            raw[rank] = sigmoid_ref(logits[token * experts + (uint32_t)best_id]);
            total += raw[rank];
        }
        for (uint32_t rank = 0; rank < used; rank++) {
            weights[token * used + rank] = routed_scale * raw[rank] /
                (total + 1e-20f);
        }
    }
}

static int test_router_topk(void) {
    enum { TOKENS = 2, EXPERTS = 128, USED = 8 };
    float logits[TOKENS * EXPERTS];
    float bias[EXPERTS];
    int32_t expected_selected[TOKENS * USED];
    float expected_weights[TOKENS * USED];
    int32_t actual_selected[TOKENS * USED];
    float actual_weights[TOKENS * USED];
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
        logits[expert] = 0.0f; /* all ties: lower expert IDs win */
        logits[EXPERTS + expert] = ((float)expert - 64.0f) / 16.0f;
        bias[expert] = 0.0f;
    }
    bias[70] = 2.0f;
    bias[4] = 0.5f;
    bias[100] = -3.0f;
    const float routed_scale = 1.25f;
    router_reference(logits, bias, TOKENS, EXPERTS, USED, routed_scale,
                     expected_selected, expected_weights);

    ds4_gpu_tensor *tl = upload(logits, sizeof(logits));
    ds4_gpu_tensor *tb = upload(bias, sizeof(bias));
    ds4_gpu_tensor *ti = ds4_gpu_tensor_alloc(sizeof(actual_selected));
    ds4_gpu_tensor *tw = ds4_gpu_tensor_alloc(sizeof(actual_weights));
    int ok = tl && tb && ti && tw &&
             ds4_gpu_inkling2_router_topk_tensor(
                 ti, tw, tl, tb, TOKENS, EXPERTS, USED, routed_scale) &&
             ds4_gpu_tensor_read(ti, 0, actual_selected, sizeof(actual_selected)) &&
             ds4_gpu_tensor_read(tw, 0, actual_weights, sizeof(actual_weights));
    for (uint32_t i = 0; ok && i < TOKENS * USED; i++) {
        if (actual_selected[i] != expected_selected[i]) {
            fprintf(stderr,
                    "Inkling2 router selection mismatch i=%u got=%d expected=%d\n",
                    i, actual_selected[i], expected_selected[i]);
            ok = 0;
            break;
        }
        if (!near(actual_weights[i], expected_weights[i], 2e-6f)) {
            fprintf(stderr,
                    "Inkling2 router weight mismatch i=%u got=%g expected=%g\n",
                    i, actual_weights[i], expected_weights[i]);
            ok = 0;
            break;
        }
    }
    ds4_gpu_tensor_free(tl);
    ds4_gpu_tensor_free(tb);
    ds4_gpu_tensor_free(ti);
    ds4_gpu_tensor_free(tw);
    return ok;
}

static uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static float half_bits_to_float(uint16_t bits) {
    const uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x03ffu;
    uint32_t value;
    if (exponent == 0) {
        if (mantissa == 0) value = sign;
        else {
            exponent = 127u - 15u + 1u;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03ffu;
            value = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31u) {
        value = sign | 0x7f800000u | (mantissa << 13);
    } else {
        value = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static int init_synthetic_quantized_model(uint64_t iq4_offset,
                                           uint64_t q8_offset,
                                           uint64_t model_size) {
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0 || model_size > SIZE_MAX) return 0;
    const uint64_t allocation = align_up_u64(model_size, (uint64_t)page);
    if (posix_memalign(&g_synthetic_model_map, (size_t)page,
                       (size_t)allocation) != 0) return 0;
    g_synthetic_model_size = model_size;
    memset(g_synthetic_model_map, 0, (size_t)allocation);

    enum { IN = 96, OUT = 5, BLOCKS = IN / 32 };
    const uint32_t iq4_row_bytes = BLOCKS * 18u;
    const uint32_t q8_row_bytes = BLOCKS * 34u;
    uint8_t *map = g_synthetic_model_map;
    for (uint32_t row = 0; row < OUT; row++) {
        for (uint32_t block = 0; block < BLOCKS; block++) {
            uint8_t *iq4 = map + iq4_offset + (uint64_t)row * iq4_row_bytes +
                           (uint64_t)block * 18u;
            const uint16_t iq4_scale = (uint16_t[]){ 0x3800u, 0x3c00u, 0x4000u }
                [(row + block) % 3u];
            memcpy(iq4, &iq4_scale, sizeof(iq4_scale));
            for (uint32_t j = 0; j < 16; j++) {
                const uint8_t lo = (uint8_t)((row * 3u + block * 5u + j * 7u) & 15u);
                const uint8_t hi = (uint8_t)((row * 11u + block * 3u + j * 9u + 7u) & 15u);
                iq4[2u + j] = (uint8_t)(lo | (hi << 4u));
            }

            uint8_t *q8 = map + q8_offset + (uint64_t)row * q8_row_bytes +
                          (uint64_t)block * 34u;
            const uint16_t q8_scale = (uint16_t[]){ 0x3800u, 0x3c00u, 0x4000u }
                [(row + 2u * block) % 3u];
            memcpy(q8, &q8_scale, sizeof(q8_scale));
            for (uint32_t j = 0; j < 32; j++) {
                q8[2u + j] = (uint8_t)(int8_t)
                    ((int32_t)((row * 17u + block * 13u + j * 19u) % 255u) - 127);
            }
        }
    }
    return ds4_gpu_set_model_map(g_synthetic_model_map, g_synthetic_model_size);
}

static int test_quantized_matvec(void) {
    enum { IN = 96, OUT = 5, BLOCKS = IN / 32 };
    const uint64_t iq4_row_bytes = BLOCKS * 18u;
    const uint64_t q8_row_bytes = BLOCKS * 34u;
    const uint64_t iq4_offset = 64u;
    const uint64_t q8_offset = align_up_u64(iq4_offset + OUT * iq4_row_bytes, 64u);
    const uint64_t q8_bytes = OUT * q8_row_bytes;
    const uint64_t model_size = q8_offset + q8_bytes + 64u;
    static const int8_t iq4_values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113,
    };
    float x[IN], iq4_actual[OUT], q8_actual[OUT];
    float iq4_expected[OUT] = { 0 };
    float q8_expected[OUT] = { 0 };
    for (uint32_t i = 0; i < IN; i++) {
        x[i] = (float)((int32_t)((i * 23u) % 41u) - 20) / 32.0f;
    }
    if (!init_synthetic_quantized_model(iq4_offset, q8_offset, model_size)) {
        fprintf(stderr, "Inkling2 synthetic quantized model setup failed\n");
        return 0;
    }
    const uint8_t *map = g_synthetic_model_map;
    for (uint32_t row = 0; row < OUT; row++) {
        for (uint32_t i = 0; i < IN; i++) {
            const uint32_t block = i / 32u;
            const uint32_t within = i % 32u;
            const uint8_t *iq4 = map + iq4_offset +
                (uint64_t)row * iq4_row_bytes + (uint64_t)block * 18u;
            uint16_t d_bits;
            memcpy(&d_bits, iq4, sizeof(d_bits));
            const uint8_t packed = iq4[2u + (within & 15u)];
            const uint8_t code = within < 16u ? packed & 15u : packed >> 4u;
            iq4_expected[row] += half_bits_to_float(d_bits) *
                (float)iq4_values[code] * x[i];

            const uint8_t *q8 = map + q8_offset +
                (uint64_t)row * q8_row_bytes + (uint64_t)block * 34u;
            memcpy(&d_bits, q8, sizeof(d_bits));
            const int8_t quant = (int8_t)q8[2u + within];
            q8_expected[row] += half_bits_to_float(d_bits) *
                (float)quant * x[i];
        }
    }

    ds4_gpu_tensor *tx = upload(x, sizeof(x));
    ds4_gpu_tensor *tiq4 = ds4_gpu_tensor_alloc(sizeof(iq4_actual));
    ds4_gpu_tensor *tq8 = ds4_gpu_tensor_alloc(sizeof(q8_actual));
    int ok = tx && tiq4 && tq8 &&
        ds4_gpu_inkling2_gemv_iq4_nl_tensor(
            tiq4, g_synthetic_model_map, g_synthetic_model_size,
            iq4_offset, IN, OUT, tx) &&
        ds4_gpu_matmul_q8_0_tensor(
            tq8, g_synthetic_model_map, g_synthetic_model_size,
            q8_offset, IN, OUT, tx, 1u) &&
        ds4_gpu_tensor_read(tiq4, 0, iq4_actual, sizeof(iq4_actual)) &&
        ds4_gpu_tensor_read(tq8, 0, q8_actual, sizeof(q8_actual));
    for (uint32_t row = 0; ok && row < OUT; row++) {
        const float iq4_tol = 3e-5f * fmaxf(1.0f, fabsf(iq4_expected[row]));
        const float q8_tol = 3e-5f * fmaxf(1.0f, fabsf(q8_expected[row]));
        if (!near(iq4_actual[row], iq4_expected[row], iq4_tol)) {
            fprintf(stderr,
                    "Inkling2 IQ4_NL GEMV mismatch row=%u got=%g expected=%g\n",
                    row, iq4_actual[row], iq4_expected[row]);
            ok = 0;
        }
        if (!near(q8_actual[row], q8_expected[row], q8_tol)) {
            fprintf(stderr,
                    "Inkling2 Q8_0 GEMV mismatch row=%u got=%g expected=%g\n",
                    row, q8_actual[row], q8_expected[row]);
            ok = 0;
        }
    }
    if (ok && ds4_gpu_inkling2_gemv_iq4_nl_tensor(
            tiq4, g_synthetic_model_map, g_synthetic_model_size,
            iq4_offset, IN - 1u, OUT, tx)) {
        fprintf(stderr, "Inkling2 IQ4_NL GEMV accepted a partial block\n");
        ok = 0;
    }
    if (ok && ds4_gpu_matmul_q8_0_tensor(
            tq8, g_synthetic_model_map, g_synthetic_model_size,
            q8_offset, IN - 1u, OUT, tx, 1u)) {
        fprintf(stderr, "Inkling2 Q8_0 GEMV accepted a partial block\n");
        ok = 0;
    }
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tiq4);
    ds4_gpu_tensor_free(tq8);
    return ok;
}

static int test_expert_accumulation(void) {
    enum { USED = 8, OUT = 13 };
    float expert_outputs[USED * OUT];
    float weights[USED];
    float expected[OUT] = { 0 };
    float actual[OUT];
    float total = 0.0f;
    for (uint32_t expert = 0; expert < USED; expert++) {
        weights[expert] = (float)(expert + 1u) / 36.0f;
        total += weights[expert];
        for (uint32_t row = 0; row < OUT; row++) {
            expert_outputs[expert * OUT + row] =
                (float)((int32_t)((expert * 7u + row * 11u) % 31u) - 15) / 9.0f;
        }
    }
    for (uint32_t expert = 0; expert < USED; expert++) {
        weights[expert] /= total;
        for (uint32_t row = 0; row < OUT; row++) {
            expected[row] += weights[expert] * expert_outputs[expert * OUT + row];
        }
    }
    ds4_gpu_tensor *te = upload(expert_outputs, sizeof(expert_outputs));
    ds4_gpu_tensor *tw = upload(weights, sizeof(weights));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(actual));
    int ok = te && tw && to &&
        ds4_gpu_inkling2_expert_accumulate_tensor(to, te, tw, OUT, USED) &&
        ds4_gpu_tensor_read(to, 0, actual, sizeof(actual));
    for (uint32_t row = 0; ok && row < OUT; row++) {
        if (!near(actual[row], expected[row], 2e-6f)) {
            fprintf(stderr,
                    "Inkling2 expert accumulation mismatch row=%u\n", row);
            ok = 0;
        }
    }
    ds4_gpu_tensor_free(te);
    ds4_gpu_tensor_free(tw);
    ds4_gpu_tensor_free(to);
    return ok;
}

static void pack_synthetic_iq4_experts(uint8_t *dst, uint32_t n_experts,
                                      uint32_t rows, uint32_t in_dim,
                                      uint32_t seed) {
    static const uint16_t scales[] = { 0x3800u, 0x3c00u, 0x4000u };
    const uint32_t blocks_per_row = in_dim / 32u;
    const uint32_t row_bytes = blocks_per_row * 18u;
    const uint64_t expert_bytes = (uint64_t)rows * row_bytes;
    for (uint32_t expert = 0; expert < n_experts; expert++) {
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                uint8_t *packed = dst + (uint64_t)expert * expert_bytes +
                    (uint64_t)row * row_bytes + (uint64_t)block * 18u;
                const uint16_t scale = scales[(expert + row + seed) % 3u];
                memcpy(packed, &scale, sizeof(scale));
                for (uint32_t i = 0; i < 16; i++) {
                    const uint8_t lo = (uint8_t)((expert * 3u + row * 5u +
                                                   block * 7u + i * 11u + seed) & 15u);
                    const uint8_t hi = (uint8_t)((expert * 13u + row * 7u +
                                                   block * 3u + i * 5u + seed + 1u) & 15u);
                    packed[2u + i] = (uint8_t)(lo | (hi << 4u));
                }
            }
        }
    }
}

static void pack_synthetic_q8_experts(uint8_t *dst, uint32_t n_experts,
                                      uint32_t rows, uint32_t in_dim,
                                      uint32_t seed) {
    static const uint16_t scales[] = { 0x3800u, 0x3c00u, 0x4000u };
    const uint32_t blocks_per_row = in_dim / 32u;
    const uint32_t row_bytes = blocks_per_row * 34u;
    const uint64_t expert_bytes = (uint64_t)rows * row_bytes;
    for (uint32_t expert = 0; expert < n_experts; expert++) {
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                uint8_t *packed = dst + (uint64_t)expert * expert_bytes +
                    (uint64_t)row * row_bytes + (uint64_t)block * 34u;
                const uint16_t scale = scales[(expert + row + seed) % 3u];
                memcpy(packed, &scale, sizeof(scale));
                for (uint32_t i = 0; i < 32; i++) {
                    packed[2u + i] = (uint8_t)(int8_t)
                        ((int32_t)((expert * 7u + row * 11u + block * 3u + i * 5u + seed) % 63u) - 31);
                }
            }
        }
    }
}

static float synthetic_iq4_expert_dot(const uint8_t *map, uint64_t tensor_offset,
                                      uint32_t expert, uint32_t row,
                                      uint32_t rows, uint32_t in_dim,
                                      const float *x) {
    static const int8_t values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    const uint32_t blocks_per_row = in_dim / 32u;
    const uint32_t row_bytes = blocks_per_row * 18u;
    const uint64_t expert_bytes = (uint64_t)rows * row_bytes;
    const uint8_t *weight = map + tensor_offset + (uint64_t)expert * expert_bytes +
        (uint64_t)row * row_bytes;
    float sum = 0.0f;
    for (uint32_t i = 0; i < in_dim; i++) {
        const uint32_t block = i / 32u;
        const uint32_t within = i % 32u;
        const uint8_t *q = weight + (uint64_t)block * 18u;
        uint16_t scale_bits;
        memcpy(&scale_bits, q, sizeof(scale_bits));
        const uint8_t packed = q[2u + (within & 15u)];
        const uint32_t code = within < 16u ? packed & 15u : packed >> 4u;
        sum += half_bits_to_float(scale_bits) * (float)values[code] * x[i];
    }
    return sum;
}

static float synthetic_q8_expert_dot(const uint8_t *map, uint64_t tensor_offset,
                                     uint32_t expert, uint32_t row,
                                     uint32_t rows, uint32_t in_dim,
                                     const float *x) {
    const uint32_t blocks_per_row = in_dim / 32u;
    const uint32_t row_bytes = blocks_per_row * 34u;
    const uint64_t expert_bytes = (uint64_t)rows * row_bytes;
    const uint8_t *weight = map + tensor_offset + (uint64_t)expert * expert_bytes +
        (uint64_t)row * row_bytes;
    float sum = 0.0f;
    for (uint32_t i = 0; i < in_dim; i++) {
        const uint8_t *q = weight + (uint64_t)(i / 32u) * 34u;
        uint16_t scale_bits;
        memcpy(&scale_bits, q, sizeof(scale_bits));
        const int8_t value = (int8_t)q[2u + i % 32u];
        sum += half_bits_to_float(scale_bits) * (float)value * x[i];
    }
    return sum;
}

static int test_situ_moe_metal(void) {
    enum { HIDDEN = 32, EXPERT_DIM = 32, EXPERTS = 128, USED = 8 };
    const uint64_t router_offset = 64u;
    const uint64_t router_bytes = (uint64_t)HIDDEN * EXPERTS * sizeof(float);
    const uint64_t gate_up_offset = align_up_u64(router_offset + router_bytes, 64u);
    const uint32_t iq4_row_bytes = HIDDEN / 32u * 18u;
    const uint64_t gate_up_bytes = (uint64_t)EXPERTS * 2u * EXPERT_DIM * iq4_row_bytes;
    const uint64_t down_offset = align_up_u64(gate_up_offset + gate_up_bytes, 64u);
    const uint64_t down_bytes = (uint64_t)EXPERTS * HIDDEN * iq4_row_bytes;
    const uint64_t q8_gate_up_offset = align_up_u64(down_offset + down_bytes, 64u);
    const uint32_t q8_row_bytes = HIDDEN / 32u * 34u;
    const uint64_t q8_gate_up_bytes = (uint64_t)EXPERTS * 2u * EXPERT_DIM * q8_row_bytes;
    const uint64_t q8_down_offset = align_up_u64(q8_gate_up_offset + q8_gate_up_bytes, 64u);
    const uint64_t q8_down_bytes = (uint64_t)EXPERTS * HIDDEN * q8_row_bytes;
    const uint64_t model_size = q8_down_offset + q8_down_bytes + 64u;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0 || model_size > SIZE_MAX) {
        fprintf(stderr, "Inkling2 SiTU MoE invalid synthetic map size\n");
        return 0;
    }
    const uint64_t allocation = align_up_u64(model_size, (uint64_t)page);
    if (posix_memalign(&g_synthetic_moe_map, (size_t)page,
                       (size_t)allocation) != 0) {
        fprintf(stderr, "Inkling2 SiTU MoE synthetic map allocation failed\n");
        return 0;
    }
    g_synthetic_moe_size = model_size;
    memset(g_synthetic_moe_map, 0, (size_t)allocation);

    uint8_t *map = g_synthetic_moe_map;
    float *router = (float *)(void *)(map + router_offset);
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
        router[(uint64_t)expert * HIDDEN] = ((int32_t)expert - 64) * 0.03f;
    }
    pack_synthetic_iq4_experts(map + gate_up_offset, EXPERTS,
                               2u * EXPERT_DIM, HIDDEN, 1u);
    pack_synthetic_iq4_experts(map + down_offset, EXPERTS,
                               HIDDEN, EXPERT_DIM, 4u);
    pack_synthetic_q8_experts(map + q8_gate_up_offset, EXPERTS,
                              2u * EXPERT_DIM, HIDDEN, 2u);
    pack_synthetic_q8_experts(map + q8_down_offset, EXPERTS,
                              HIDDEN, EXPERT_DIM, 5u);
    if (!ds4_gpu_set_model_map(g_synthetic_moe_map, g_synthetic_moe_size)) {
        fprintf(stderr, "Inkling2 SiTU MoE model map registration failed\n");
        return 0;
    }

    float input[HIDDEN], bias[EXPERTS], logits[EXPERTS];
    for (uint32_t i = 0; i < HIDDEN; i++)
        input[i] = i == 0 ? 1.0f : (float)((int32_t)(i % 9u) - 4) / 16.0f;
    memset(bias, 0, sizeof(bias));
    bias[0] = 4.0f;
    for (uint32_t expert = 0; expert < EXPERTS; expert++)
        logits[expert] = router[(uint64_t)expert * HIDDEN] * input[0];

    int32_t selected[USED];
    float route_weights[USED];
    router_reference(logits, bias, 1u, EXPERTS, USED, 8.0f,
                     selected, route_weights);
    ds4_gpu_tensor *debug_logits = ds4_gpu_tensor_alloc(sizeof(logits));
    ds4_gpu_tensor *debug_ids = ds4_gpu_tensor_alloc(sizeof(selected));
    ds4_gpu_tensor *debug_weights = ds4_gpu_tensor_alloc(sizeof(route_weights));
    ds4_gpu_tensor *debug_input = upload(input, sizeof(input));
    ds4_gpu_tensor *debug_bias = upload(bias, sizeof(bias));
    int route_ok = debug_logits && debug_ids && debug_weights && debug_input && debug_bias &&
        ds4_gpu_matmul_f32_tensor(debug_logits, map, model_size, router_offset,
                                  HIDDEN, EXPERTS, debug_input, 1u) &&
        ds4_gpu_inkling2_router_topk_tensor(debug_ids, debug_weights, debug_logits,
                                            debug_bias, 1u, EXPERTS, USED, 8.0f);
    int32_t debug_selected[USED];
    float debug_route_weights[USED];
    if (route_ok) route_ok =
        ds4_gpu_tensor_read(debug_ids, 0, debug_selected, sizeof(debug_selected)) &&
        ds4_gpu_tensor_read(debug_weights, 0, debug_route_weights,
                            sizeof(debug_route_weights));
    for (uint32_t i = 0; route_ok && i < USED; i++) {
        if (debug_selected[i] != selected[i] ||
            !near(debug_route_weights[i], route_weights[i], 2e-6f)) {
            fprintf(stderr, "Inkling2 SiTU MoE route mismatch i=%u id=%d expected=%d\n",
                    i, debug_selected[i], selected[i]);
            route_ok = 0;
        }
    }
    if (!route_ok) fprintf(stderr, "Inkling2 SiTU MoE router fixture failed\n");
    ds4_gpu_tensor_free(debug_logits);
    ds4_gpu_tensor_free(debug_weights);
    ds4_gpu_tensor_free(debug_bias);
    if (!route_ok) {
        ds4_gpu_tensor_free(debug_ids);
        ds4_gpu_tensor_free(debug_input);
        return 0;
    }

    const float beta = 1.25f;
    const float linear_beta = 0.75f;
    float expected[HIDDEN] = { 0 };
    float gate_up[USED * 2u * EXPERT_DIM];
    float activation[USED * EXPERT_DIM];
    const uint8_t *quantized = map;
    for (uint32_t slot = 0; slot < USED; slot++) {
        for (uint32_t row = 0; row < 2u * EXPERT_DIM; row++) {
            gate_up[(uint64_t)slot * 2u * EXPERT_DIM + row] =
                synthetic_iq4_expert_dot(quantized, gate_up_offset,
                    (uint32_t)selected[slot], row, 2u * EXPERT_DIM, HIDDEN, input);
        }
        for (uint32_t row = 0; row < EXPERT_DIM; row++) {
            const float gate = gate_up[(uint64_t)slot * 2u * EXPERT_DIM + row];
            const float up = gate_up[(uint64_t)slot * 2u * EXPERT_DIM + EXPERT_DIM + row];
            activation[(uint64_t)slot * EXPERT_DIM + row] =
                beta * tanhf(gate / beta) * sigmoid_ref(gate) *
                linear_beta * tanhf(up / linear_beta);
        }
        for (uint32_t row = 0; row < HIDDEN; row++) {
            expected[row] += route_weights[slot] * synthetic_iq4_expert_dot(
                quantized, down_offset, (uint32_t)selected[slot], row,
                HIDDEN, EXPERT_DIM, activation + (uint64_t)slot * EXPERT_DIM);
        }
    }

    float actual_gate_up[USED * 2u * EXPERT_DIM];
    ds4_gpu_tensor *debug_gate_up = ds4_gpu_tensor_alloc(sizeof(actual_gate_up));
    int gate_up_ok = debug_gate_up && ds4_gpu_inkling2_expert_gemv_tensor(
        debug_gate_up, g_synthetic_moe_map, g_synthetic_moe_size, gate_up_offset,
        20u, HIDDEN, 2u * EXPERT_DIM, EXPERTS, USED, 0, debug_ids, debug_input) &&
        ds4_gpu_tensor_read(debug_gate_up, 0, actual_gate_up, sizeof(actual_gate_up));
    for (uint32_t i = 0; gate_up_ok && i < USED * 2u * EXPERT_DIM; i++) {
        if (!near(actual_gate_up[i], gate_up[i], 3e-4f * fmaxf(1.0f, fabsf(gate_up[i])))) {
            fprintf(stderr, "Inkling2 selected expert GEMV mismatch i=%u got=%g expected=%g\n",
                    i, actual_gate_up[i], gate_up[i]);
            gate_up_ok = 0;
        }
    }
    float actual_activation[USED * EXPERT_DIM];
    ds4_gpu_tensor *debug_activation = ds4_gpu_tensor_alloc(sizeof(actual_activation));
    int activation_ok = gate_up_ok && debug_activation &&
        ds4_gpu_inkling2_situ_tensor(debug_activation, debug_gate_up,
                                      USED, EXPERT_DIM, beta, linear_beta) &&
        ds4_gpu_tensor_read(debug_activation, 0, actual_activation,
                            sizeof(actual_activation));
    for (uint32_t i = 0; activation_ok && i < USED * EXPERT_DIM; i++) {
        if (!near(actual_activation[i], activation[i],
                  3e-5f * fmaxf(1.0f, fabsf(activation[i])))) {
            const uint32_t slot = i / EXPERT_DIM;
            const uint32_t row = i - slot * EXPERT_DIM;
            fprintf(stderr,
                    "Inkling2 SiTU mismatch i=%u gate=%g up=%g got=%g expected=%g\n",
                    i, actual_gate_up[(uint64_t)slot * 2u * EXPERT_DIM + row],
                    actual_gate_up[(uint64_t)slot * 2u * EXPERT_DIM + EXPERT_DIM + row],
                    actual_activation[i], activation[i]);
            activation_ok = 0;
        }
    }
    float actual_down[USED * HIDDEN];
    ds4_gpu_tensor *debug_down = ds4_gpu_tensor_alloc(sizeof(actual_down));
    int down_ok = activation_ok && debug_down &&
        ds4_gpu_inkling2_expert_gemv_tensor(
            debug_down, g_synthetic_moe_map, g_synthetic_moe_size, down_offset,
            20u, EXPERT_DIM, HIDDEN, EXPERTS, USED, 1, debug_ids, debug_activation) &&
        ds4_gpu_tensor_read(debug_down, 0, actual_down, sizeof(actual_down));
    for (uint32_t slot = 0; down_ok && slot < USED; slot++) {
        for (uint32_t row = 0; down_ok && row < HIDDEN; row++) {
            const float expected_row = synthetic_iq4_expert_dot(
                quantized, down_offset, (uint32_t)selected[slot], row,
                HIDDEN, EXPERT_DIM, activation + (uint64_t)slot * EXPERT_DIM);
            const uint64_t i = (uint64_t)slot * HIDDEN + row;
            if (!near(actual_down[i], expected_row,
                      4e-4f * fmaxf(1.0f, fabsf(expected_row)))) {
                fprintf(stderr, "Inkling2 down GEMV mismatch slot=%u row=%u got=%g expected=%g\n",
                        slot, row, actual_down[i], expected_row);
                down_ok = 0;
            }
        }
    }
    ds4_gpu_tensor_free(debug_gate_up);
    ds4_gpu_tensor_free(debug_activation);
    ds4_gpu_tensor_free(debug_down);
    ds4_gpu_tensor_free(debug_ids);
    ds4_gpu_tensor_free(debug_input);
    if (!down_ok) return 0;

    ds4_gpu_tensor *tx = upload(input, sizeof(input));
    ds4_gpu_tensor *tb = upload(bias, sizeof(bias));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(expected));
    if (!tx || !tb || !to) {
        fprintf(stderr, "Inkling2 SiTU MoE fixture tensor allocation failed\n");
    }
    int ok = tx && tb && to && ds4_inkling_moe_situ_metal_with_workspace(
        to, tx, tb, g_synthetic_moe_map, g_synthetic_moe_size,
        router_offset, gate_up_offset, down_offset, HIDDEN, EXPERT_DIM,
        EXPERTS, USED, 20u, 8.0f, beta, linear_beta, &g_operator_context.moe);
    float actual[HIDDEN];
    if (ok && !ds4_gpu_tensor_read(to, 0, actual, sizeof(actual))) {
        fprintf(stderr, "Inkling2 SiTU MoE output read failed\n");
        ok = 0;
    }
    for (uint32_t i = 0; ok && i < HIDDEN; i++) {
        const float tolerance = 1.5e-3f * fmaxf(1.0f, fabsf(expected[i]));
        if (!near(actual[i], expected[i], tolerance)) {
            fprintf(stderr, "Inkling2 SiTU MoE mismatch i=%u got=%g expected=%g\n",
                    i, actual[i], expected[i]);
            ok = 0;
        }
    }
    float q8_expected[HIDDEN] = { 0 };
    float q8_gate_up[USED * 2u * EXPERT_DIM];
    float q8_activation[USED * EXPERT_DIM];
    for (uint32_t slot = 0; ok && slot < USED; slot++) {
        for (uint32_t row = 0; row < 2u * EXPERT_DIM; row++) {
            q8_gate_up[(uint64_t)slot * 2u * EXPERT_DIM + row] =
                synthetic_q8_expert_dot(map, q8_gate_up_offset,
                    (uint32_t)selected[slot], row, 2u * EXPERT_DIM, HIDDEN, input);
        }
        for (uint32_t row = 0; row < EXPERT_DIM; row++) {
            const float gate = q8_gate_up[(uint64_t)slot * 2u * EXPERT_DIM + row];
            const float up = q8_gate_up[(uint64_t)slot * 2u * EXPERT_DIM + EXPERT_DIM + row];
            q8_activation[(uint64_t)slot * EXPERT_DIM + row] =
                beta * tanhf(gate / beta) * sigmoid_ref(gate) *
                linear_beta * tanhf(up / linear_beta);
        }
        for (uint32_t row = 0; row < HIDDEN; row++) {
            q8_expected[row] += route_weights[slot] * synthetic_q8_expert_dot(
                map, q8_down_offset, (uint32_t)selected[slot], row,
                HIDDEN, EXPERT_DIM, q8_activation + (uint64_t)slot * EXPERT_DIM);
        }
    }
    float q8_actual[HIDDEN];
    ds4_gpu_tensor *tq8_out = ds4_gpu_tensor_alloc(sizeof(q8_actual));
    int q8_ok = ok && tq8_out && ds4_inkling_moe_situ_metal_with_workspace(
        tq8_out, tx, tb, g_synthetic_moe_map, g_synthetic_moe_size,
        router_offset, q8_gate_up_offset, q8_down_offset, HIDDEN, EXPERT_DIM,
        EXPERTS, USED, 8u, 8.0f, beta, linear_beta, &g_operator_context.moe) &&
        ds4_gpu_tensor_read(tq8_out, 0, q8_actual, sizeof(q8_actual));
    for (uint32_t i = 0; q8_ok && i < HIDDEN; i++) {
        const float tolerance = 1.5e-3f * fmaxf(1.0f, fabsf(q8_expected[i]));
        if (!near(q8_actual[i], q8_expected[i], tolerance)) {
            fprintf(stderr, "Inkling2 Q8_0 SiTU MoE mismatch i=%u got=%g expected=%g\n",
                    i, q8_actual[i], q8_expected[i]);
            q8_ok = 0;
        }
    }
    if (!q8_ok) ok = 0;
    ds4_gpu_tensor_free(tq8_out);

    enum { BATCH_TOKENS = 3 };
    float batch_input[BATCH_TOKENS * HIDDEN];
    float batch_logits[BATCH_TOKENS * EXPERTS];
    int32_t expected_ids[BATCH_TOKENS * USED];
    float expected_weights[BATCH_TOKENS * USED];
    for (uint32_t token = 0; token < BATCH_TOKENS; token++) {
        for (uint32_t i = 0; i < HIDDEN; i++) {
            batch_input[(uint64_t)token * HIDDEN + i] =
                i == 0 ? (float)((int32_t)token - 1) * 0.7f :
                (float)((int32_t)((token * 13u + i * 7u) % 17u) - 8) / 32.0f;
        }
        for (uint32_t expert = 0; expert < EXPERTS; expert++) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < HIDDEN; i++)
                sum += router[(uint64_t)expert * HIDDEN + i] *
                    batch_input[(uint64_t)token * HIDDEN + i];
            batch_logits[(uint64_t)token * EXPERTS + expert] = sum;
        }
    }
    router_reference(batch_logits, bias, BATCH_TOKENS, EXPERTS, USED, 8.0f,
                     expected_ids, expected_weights);
    ds4_gpu_tensor *batch_x = upload(batch_input, sizeof(batch_input));
    ds4_gpu_tensor *batch_out = ds4_gpu_tensor_alloc(sizeof(batch_input));
    int batch_ok = ok && q8_ok && batch_x && batch_out;
    const uint32_t weight_types[] = { 20u, 8u };
    const uint64_t gate_offsets[] = { gate_up_offset, q8_gate_up_offset };
    const uint64_t down_offsets[] = { down_offset, q8_down_offset };
    for (uint32_t type = 0; batch_ok && type < 2u; type++) {
        batch_ok = ds4_inkling_moe_situ_metal_batch_with_workspace(
            batch_out, batch_x, tb, g_synthetic_moe_map, g_synthetic_moe_size,
            router_offset, gate_offsets[type], down_offsets[type],
            HIDDEN, EXPERT_DIM,
            BATCH_TOKENS, EXPERTS, USED, weight_types[type], 8.0f,
            beta, linear_beta, &g_operator_context.moe);
        float batch_actual[BATCH_TOKENS * HIDDEN];
        int32_t actual_ids[BATCH_TOKENS * USED];
        float actual_weights[BATCH_TOKENS * USED];
        batch_ok = batch_ok &&
            ds4_gpu_tensor_read(batch_out, 0, batch_actual, sizeof(batch_actual)) &&
            ds4_gpu_tensor_read(g_operator_context.moe.ids, 0,
                                actual_ids, sizeof(actual_ids)) &&
            ds4_gpu_tensor_read(g_operator_context.moe.weights, 0,
                                actual_weights, sizeof(actual_weights));
        for (uint32_t i = 0; batch_ok && i < BATCH_TOKENS * USED; i++) {
            if (actual_ids[i] != expected_ids[i] ||
                !near(actual_weights[i], expected_weights[i], 2e-6f)) {
                fprintf(stderr,
                        "Inkling2 batched router mismatch type=%u route=%u got=%d expected=%d\n",
                        weight_types[type], i, actual_ids[i], expected_ids[i]);
                batch_ok = 0;
            }
        }
        for (uint32_t token = 0; batch_ok && token < BATCH_TOKENS; token++) {
            ds4_gpu_tensor *token_x = upload(
                batch_input + (uint64_t)token * HIDDEN,
                (uint64_t)HIDDEN * sizeof(float));
            ds4_gpu_tensor *token_out = ds4_gpu_tensor_alloc(
                (uint64_t)HIDDEN * sizeof(float));
            float token_actual[HIDDEN];
            batch_ok = token_x && token_out &&
                ds4_inkling_moe_situ_metal_with_workspace(
                    token_out, token_x, tb, g_synthetic_moe_map,
                    g_synthetic_moe_size, router_offset, gate_offsets[type],
                    down_offsets[type], HIDDEN, EXPERT_DIM, EXPERTS, USED,
                    weight_types[type], 8.0f, beta, linear_beta,
                    &g_operator_context.moe) &&
                ds4_gpu_tensor_read(token_out, 0, token_actual,
                                    sizeof(token_actual));
            for (uint32_t i = 0; batch_ok && i < HIDDEN; i++) {
                const float expected_row = batch_actual[(uint64_t)token * HIDDEN + i];
                if (!near(token_actual[i], expected_row,
                          2e-4f * fmaxf(1.0f, fabsf(expected_row)))) {
                    fprintf(stderr,
                            "Inkling2 batched MoE mismatch type=%u token=%u row=%u got=%g expected=%g\n",
                            weight_types[type], token, i, token_actual[i], expected_row);
                    batch_ok = 0;
                }
            }
            ds4_gpu_tensor_free(token_x);
            ds4_gpu_tensor_free(token_out);
        }
    }
    if (batch_ok) {
        float no_bias[EXPERTS] = { 0 };
        int32_t no_bias_expected_ids[BATCH_TOKENS * USED];
        float no_bias_expected_weights[BATCH_TOKENS * USED];
        int32_t no_bias_actual_ids[BATCH_TOKENS * USED];
        float no_bias_actual_weights[BATCH_TOKENS * USED];
        router_reference(batch_logits, no_bias, BATCH_TOKENS, EXPERTS,
                         USED, 8.0f, no_bias_expected_ids,
                         no_bias_expected_weights);
        batch_ok = ds4_inkling_moe_situ_metal_batch_with_workspace(
            batch_out, batch_x, NULL, g_synthetic_moe_map, g_synthetic_moe_size,
            router_offset, gate_up_offset, down_offset, HIDDEN, EXPERT_DIM,
            BATCH_TOKENS, EXPERTS, USED, 20u, 8.0f, beta, linear_beta,
            &g_operator_context.moe) &&
            ds4_gpu_tensor_read(g_operator_context.moe.ids, 0,
                no_bias_actual_ids, sizeof(no_bias_actual_ids)) &&
            ds4_gpu_tensor_read(g_operator_context.moe.weights, 0,
                no_bias_actual_weights, sizeof(no_bias_actual_weights));
        for (uint32_t i = 0; batch_ok && i < BATCH_TOKENS * USED; i++) {
            if (no_bias_actual_ids[i] != no_bias_expected_ids[i] ||
                !near(no_bias_actual_weights[i], no_bias_expected_weights[i],
                      2e-6f)) {
                fprintf(stderr,
                        "Inkling2 biasless batched router mismatch route=%u got=%d expected=%d\n",
                        i, no_bias_actual_ids[i], no_bias_expected_ids[i]);
                batch_ok = 0;
            }
        }
    }
    if (!batch_ok) ok = 0;
    ds4_gpu_tensor_free(batch_x);
    ds4_gpu_tensor_free(batch_out);
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tb);
    ds4_gpu_tensor_free(to);
    return ok;
}

/* The grouped IQ4_NL operator uses a stable per-expert route map.
 * This fixture reuses the small synthetic weights from test_situ_moe_metal,
 * while the high-level path is gated to the real 2304/1152 Inkling shape. */
static int test_expert_grouped_iq4(void) {
    enum { TOKENS = 32, EXPERTS = 128, USED = 8,
           HIDDEN = 32, FFN = 32, ROUTES = TOKENS * USED };
    int ok = 1;
    const uint64_t gate_offset = align_up_u64(
        64u + (uint64_t)HIDDEN * EXPERTS * sizeof(float), 64u);
    const uint64_t gate_bytes =
        (uint64_t)EXPERTS * 2u * FFN * (HIDDEN / 32u * 18u);
    const uint64_t down_offset = align_up_u64(gate_offset + gate_bytes, 64u);
    int32_t ids[ROUTES];
    float input[TOKENS * HIDDEN];
    for (uint32_t token = 0; token < TOKENS; token++) {
        for (uint32_t col = 0; col < HIDDEN; col++)
            input[token * HIDDEN + col] =
                (float)((int32_t)((token * 17u + col * 3u) % 31u) - 15) / 64.0f;
        for (uint32_t slot = 0; slot < USED; slot++)
            ids[token * USED + slot] = (int32_t)((token * 7u + slot * 13u) % EXPERTS);
    }
    ds4_gpu_tensor *tid = upload(ids, sizeof(ids));
    ds4_gpu_tensor *tx = upload(input, sizeof(input));
    ds4_gpu_tensor *gate_ref = ds4_gpu_tensor_alloc(
        (uint64_t)ROUTES * 2u * FFN * sizeof(float));
    ds4_gpu_tensor *gate_gemm = ds4_gpu_tensor_alloc(
        (uint64_t)ROUTES * 2u * FFN * sizeof(float));
    ds4_gpu_tensor *act = ds4_gpu_tensor_alloc(
        (uint64_t)ROUTES * FFN * sizeof(float));
    ds4_gpu_tensor *down_ref = ds4_gpu_tensor_alloc(
        (uint64_t)ROUTES * HIDDEN * sizeof(float));
    ds4_gpu_tensor *down_gemm = ds4_gpu_tensor_alloc(
        (uint64_t)ROUTES * HIDDEN * sizeof(float));
    uint32_t counts[EXPERTS];
    uint32_t routes[EXPERTS * ROUTES];
    uint8_t seen[ROUTES];
    ds4_gpu_tensor *small = NULL, *small_counts = NULL;
    CHECK(tid && tx && gate_ref && gate_gemm && act && down_ref && down_gemm);
    CHECK(ds4_gpu_inkling2_expert_grouped_available());

    /* Validate the mapper at 1, 8 and 32 tokens, including stable ordering. */
    const uint32_t tested_sizes[] = {1u, 8u, 32u};
    for (uint32_t j = 0; j < 3u; j++) {
        const uint32_t n = tested_sizes[j];
        const uint32_t nr = n * USED;
        CHECK(ds4_gpu_inkling2_expert_route_map_rows_tensor(
            g_operator_context.moe.route_map,
            g_operator_context.moe.route_counts, tid, n, EXPERTS, USED));
        CHECK(ds4_gpu_tensor_read(g_operator_context.moe.route_counts, 0,
                                  counts, sizeof(counts)));
        CHECK(ds4_gpu_tensor_read(g_operator_context.moe.route_map, 0,
            routes, (uint64_t)EXPERTS * nr * sizeof(uint32_t)));
        memset(seen, 0, sizeof(seen));
        uint32_t sum = 0;
        int saw_one = 0, saw_many = 0;
        for (uint32_t expert = 0; expert < EXPERTS; expert++) {
            if (counts[expert] == 1u) saw_one = 1;
            else if (counts[expert] > 1u) saw_many = 1;
            uint32_t prev = 0;
            for (uint32_t j = 0; j < counts[expert]; j++) {
                const uint32_t route = routes[expert * nr + j];
                CHECK(route < nr);
                CHECK(ids[route] == (int32_t)expert);
                CHECK(j == 0u || route > prev);
                CHECK(!seen[route]);
                seen[route] = 1;
                prev = route;
                sum++;
            }
        }
        CHECK(sum == nr);
        for (uint32_t r = 0; r < nr; r++) CHECK(seen[r] == 1u);
        if (j == 2u) CHECK(saw_one && saw_many);
    }
    /* Exercise full-count and unused-expert map behavior, even when the
     * repeated IDs would not arise from the production top-8 selector. */
    for (uint32_t r = 0; r < ROUTES; r++) ids[r] = 17;
    CHECK(ds4_gpu_tensor_write(tid, 0, ids, sizeof(ids)));
    CHECK(ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS, EXPERTS, USED));
    CHECK(ds4_gpu_tensor_read(g_operator_context.moe.route_counts, 0,
                              counts, sizeof(counts)));
    CHECK(counts[17] == ROUTES && counts[0] == 0u && counts[127] == 0u);
    for (uint32_t r = 0; r < ROUTES; r++)
        ids[r] = (int32_t)(((r / USED) * 7u + (r % USED) * 13u) % EXPERTS);
    CHECK(ds4_gpu_tensor_write(tid, 0, ids, sizeof(ids)));
    CHECK(ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS, EXPERTS, USED));
    small = ds4_gpu_tensor_alloc(4u);
    small_counts = ds4_gpu_tensor_alloc(4u);
    CHECK(small && small_counts);
    CHECK(!ds4_gpu_inkling2_expert_route_map_rows_tensor(
        small, g_operator_context.moe.route_counts, tid, TOKENS, EXPERTS, USED));
    CHECK(!ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map, small_counts, tid,
        TOKENS, EXPERTS, USED));
    CHECK(!ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS + 1u, EXPERTS, USED));
    CHECK(!ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, 0u, EXPERTS, USED));
    CHECK(!ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS, 0u, USED));
    CHECK(!ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS, EXPERTS + 1u, USED));
    CHECK(!ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS, EXPERTS, 0u));
    CHECK(!ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS, EXPERTS, USED + 1u));
    CHECK(!ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS, USED - 1u, USED));

    CHECK(ds4_gpu_inkling2_expert_gemv_rows_tensor(
        gate_ref, g_synthetic_moe_map, g_synthetic_moe_size, gate_offset,
        20u, HIDDEN, 2u * FFN, EXPERTS, USED, TOKENS, 0, tid, tx));
    CHECK(ds4_gpu_inkling2_expert_gemm_id_rows_tensor(
        gate_gemm, g_synthetic_moe_map, g_synthetic_moe_size, gate_offset,
        HIDDEN, 2u * FFN, EXPERTS, USED, TOKENS, 0,
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tx));
    float gate_a[ROUTES * 2u * FFN], gate_b[ROUTES * 2u * FFN];
    CHECK(ds4_gpu_tensor_read(gate_ref, 0, gate_a, sizeof(gate_a)));
    CHECK(ds4_gpu_tensor_read(gate_gemm, 0, gate_b, sizeof(gate_b)));
    for (uint32_t i = 0; i < ROUTES * 2u * FFN; i++)
        CHECK(near(gate_a[i], gate_b[i],
                   6e-4f * fmaxf(1.0f, fabsf(gate_a[i]))));

    CHECK(ds4_gpu_inkling2_situ_rows_tensor(
        act, gate_ref, TOKENS, USED, FFN, 1.25f, 0.75f));
    CHECK(ds4_gpu_inkling2_expert_gemv_rows_tensor(
        down_ref, g_synthetic_moe_map, g_synthetic_moe_size, down_offset,
        20u, FFN, HIDDEN, EXPERTS, USED, TOKENS, 1, tid, act));
    CHECK(ds4_gpu_inkling2_expert_gemm_id_rows_tensor(
        down_gemm, g_synthetic_moe_map, g_synthetic_moe_size, down_offset,
        FFN, HIDDEN, EXPERTS, USED, TOKENS, 1,
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, act));
    float down_a[ROUTES * HIDDEN], down_b[ROUTES * HIDDEN];
    CHECK(ds4_gpu_tensor_read(down_ref, 0, down_a, sizeof(down_a)));
    CHECK(ds4_gpu_tensor_read(down_gemm, 0, down_b, sizeof(down_b)));
    for (uint32_t i = 0; i < ROUTES * HIDDEN; i++)
        CHECK(near(down_a[i], down_b[i],
                   6e-4f * fmaxf(1.0f, fabsf(down_a[i]))));

cleanup:
    ds4_gpu_tensor_free(small);
    ds4_gpu_tensor_free(small_counts);
    ds4_gpu_tensor_free(tid);
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(gate_ref);
    ds4_gpu_tensor_free(gate_gemm);
    ds4_gpu_tensor_free(act);
    ds4_gpu_tensor_free(down_ref);
    ds4_gpu_tensor_free(down_gemm);
    return ok;
}

static int test_expert_route_map_512(void) {
    enum { TOKENS = 512, EXPERTS = 128, USED = 8, ROUTES = TOKENS * USED };
    const uint64_t route_map_bytes =
        (uint64_t)EXPERTS * ROUTES * sizeof(uint32_t);
    int ok = 1;
    int32_t ids[ROUTES];
    uint32_t counts[EXPERTS];
    uint32_t *routes = malloc((size_t)route_map_bytes);
    ds4_gpu_tensor *tid = NULL;

    for (uint32_t token = 0; token < TOKENS; token++) {
        for (uint32_t slot = 0; slot < USED; slot++) {
            ids[token * USED + slot] =
                (int32_t)((token * 7u + slot * 13u) % EXPERTS);
        }
    }
    tid = upload(ids, sizeof(ids));
    CHECK(routes && tid);
    CHECK(ds4_gpu_tensor_bytes(g_operator_context.moe.route_map) >=
          route_map_bytes);
    CHECK(ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tid, TOKENS, EXPERTS, USED));
    CHECK(ds4_gpu_tensor_read(g_operator_context.moe.route_counts, 0,
                              counts, sizeof(counts)));
    CHECK(ds4_gpu_tensor_read(g_operator_context.moe.route_map, 0,
                              routes, route_map_bytes));

    uint32_t total = 0;
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
        CHECK(counts[expert] <= ROUTES);
        uint32_t previous = 0;
        for (uint32_t rank = 0; rank < counts[expert]; rank++) {
            const uint32_t route = routes[expert * ROUTES + rank];
            CHECK(route < ROUTES);
            CHECK(ids[route] == (int32_t)expert);
            CHECK(rank == 0u || route > previous);
            previous = route;
            total++;
        }
    }
    CHECK(total == ROUTES);

cleanup:
    free(routes);
    ds4_gpu_tensor_free(tid);
    return ok;
}

static int report_expert_iq4_parity(const char *label,
                                    const float *gemv,
                                    const float *gemm,
                                    size_t count) {
    double max_abs = 0.0, max_rel = 0.0;
    uint64_t out_of_tolerance = 0, nonfinite = 0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(gemv[i]) || !isfinite(gemm[i])) {
            nonfinite++;
            out_of_tolerance++;
            continue;
        }
        const double abs_ref = fabs((double)gemv[i]);
        const double diff = fabs((double)gemm[i] - (double)gemv[i]);
        const double rel = diff / fmax(abs_ref, 1e-6);
        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > 6e-4 * fmax(1.0, abs_ref)) out_of_tolerance++;
    }
    fprintf(stderr,
            "Inkling2 full-shape IQ4_NL %s GEMV/GEMM: max_abs=%.9g "
            "max_rel=%.9g out_of_tolerance=%llu/%zu nonfinite=%llu\n",
            label, max_abs, max_rel,
            (unsigned long long)out_of_tolerance, count,
            (unsigned long long)nonfinite);
    return out_of_tolerance == 0 && nonfinite == 0;
}

/* Compare the legacy GEMV and grouped GEMM at the model's real MoE shapes.
 * Only experts 0..7 have routes, so the remaining weight pages stay zero while
 * the buffer still describes the complete 128-expert tensor layout. */
static int test_expert_grouped_iq4_full_shape(void) {
    enum { TOKENS = 512, EXPERTS = 128, USED = 8,
           HIDDEN = 2304, FFN = 1152, ROUTES = TOKENS * USED };
    const uint32_t gate_out_dim = 2u * FFN;
    const uint32_t gate_row_bytes = HIDDEN / 32u * 18u;
    const uint32_t down_row_bytes = FFN / 32u * 18u;
    const uint64_t gate_offset = 64u;
    const uint64_t gate_expert_bytes = (uint64_t)gate_out_dim * gate_row_bytes;
    const uint64_t gate_bytes = (uint64_t)EXPERTS * gate_expert_bytes;
    const uint64_t down_offset = align_up_u64(gate_offset + gate_bytes, 64u);
    const uint64_t down_expert_bytes = (uint64_t)HIDDEN * down_row_bytes;
    const uint64_t down_bytes = (uint64_t)EXPERTS * down_expert_bytes;
    const uint64_t model_size = down_offset + down_bytes;
    const uint64_t output_count = (uint64_t)ROUTES * HIDDEN;
    const uint64_t output_bytes = output_count * sizeof(float);
    const uint64_t input_bytes = (uint64_t)TOKENS * HIDDEN * sizeof(float);
    const long page = sysconf(_SC_PAGESIZE);
    uint64_t allocation = 0;
    int ok = 0, model_map_set = 0;
    int scalar_env_modified = 0;
    char *saved_scalar_env = NULL;
    void *input = NULL, *gate_a = NULL, *gate_b = NULL, *gate_scalar = NULL;
    void *down_a = NULL, *down_b = NULL, *stale = NULL;
    uint32_t *stale_route_map = NULL, *stale_route_counts = NULL;
    ds4_gpu_tensor *tid = NULL, *tx = NULL, *gate_ref = NULL;
    ds4_gpu_tensor *gate_gemm = NULL, *activation = NULL;
    ds4_gpu_tensor *down_ref = NULL, *down_gemm = NULL;

    if (page <= 0 || model_size > SIZE_MAX || output_bytes > SIZE_MAX ||
        input_bytes > SIZE_MAX) {
        fprintf(stderr, "Inkling2 full-shape GEMM fixture size invalid\n");
        return 0;
    }
    allocation = align_up_u64(model_size, (uint64_t)page);
    if (allocation > SIZE_MAX || posix_memalign(&g_full_shape_iq4_map,
            (size_t)page, (size_t)allocation) != 0) {
        g_full_shape_iq4_map = NULL;
        fprintf(stderr, "Inkling2 full-shape GEMM model map allocation failed\n");
        return 0;
    }
    memset(g_full_shape_iq4_map, 0, (size_t)allocation);
    pack_synthetic_iq4_experts((uint8_t *)g_full_shape_iq4_map + gate_offset,
                               USED, gate_out_dim, HIDDEN, 37u);
    pack_synthetic_iq4_experts((uint8_t *)g_full_shape_iq4_map + down_offset,
                               USED, HIDDEN, FFN, 53u);
    if (!ds4_gpu_set_model_map(g_full_shape_iq4_map, allocation)) {
        fprintf(stderr, "Inkling2 full-shape GEMM model map registration failed\n");
        goto cleanup;
    }
    model_map_set = 1;

    input = malloc((size_t)input_bytes);
    gate_a = malloc((size_t)output_bytes);
    gate_b = malloc((size_t)output_bytes);
    gate_scalar = malloc((size_t)output_bytes);
    down_a = malloc((size_t)output_bytes);
    down_b = malloc((size_t)output_bytes);
    stale = malloc((size_t)output_bytes);
    const uint64_t route_map_bytes =
        ds4_gpu_tensor_bytes(g_operator_context.moe.route_map);
    const uint64_t route_counts_bytes =
        ds4_gpu_tensor_bytes(g_operator_context.moe.route_counts);
    if (route_map_bytes < (uint64_t)EXPERTS * ROUTES * sizeof(uint32_t) ||
        route_counts_bytes < (uint64_t)EXPERTS * sizeof(uint32_t) ||
        route_map_bytes > SIZE_MAX || route_counts_bytes > SIZE_MAX) {
        fprintf(stderr, "Inkling2 full-shape route-map workspace is undersized\n");
        goto cleanup;
    }
    stale_route_map = malloc((size_t)route_map_bytes);
    stale_route_counts = malloc((size_t)route_counts_bytes);
    if (!input || !gate_a || !gate_b || !gate_scalar || !down_a || !down_b || !stale ||
        !stale_route_map || !stale_route_counts) {
        fprintf(stderr, "Inkling2 full-shape GEMM host allocation failed\n");
        goto cleanup;
    }

    int32_t ids[ROUTES];
    for (uint32_t token = 0; token < TOKENS; token++) {
        for (uint32_t col = 0; col < HIDDEN; col++) {
            const int32_t q = (int32_t)((token * 31u + col * 17u) % 251u) - 125;
            const float jitter = 0.00048828125f *
                sinf((float)(token + 1u) * (float)(col + 3u) * 0.017f);
            ((float *)input)[(uint64_t)token * HIDDEN + col] =
                (float)q / 2048.0f + jitter;
        }
        for (uint32_t slot = 0; slot < USED; slot++)
            ids[token * USED + slot] = (int32_t)slot;
    }
    memset(stale_route_map, 0xa5, (size_t)route_map_bytes);
    memset(stale_route_counts, 0xa5, (size_t)route_counts_bytes);
    for (uint64_t i = 0; i < output_count; i++) ((float *)stale)[i] = NAN;

    tid = upload(ids, sizeof(ids));
    tx = upload(input, input_bytes);
    gate_ref = ds4_gpu_tensor_alloc(output_bytes);
    gate_gemm = ds4_gpu_tensor_alloc(output_bytes);
    activation = ds4_gpu_tensor_alloc((uint64_t)ROUTES * FFN * sizeof(float));
    down_ref = ds4_gpu_tensor_alloc(output_bytes);
    down_gemm = ds4_gpu_tensor_alloc(output_bytes);
    if (!tid || !tx || !gate_ref || !gate_gemm || !activation ||
        !down_ref || !down_gemm) {
        fprintf(stderr, "Inkling2 full-shape GEMM tensor allocation failed\n");
        goto cleanup;
    }

    CHECK(ds4_gpu_tensor_write(g_operator_context.moe.route_map, 0,
                               stale_route_map, route_map_bytes));
    CHECK(ds4_gpu_tensor_write(g_operator_context.moe.route_counts, 0,
                               stale_route_counts, route_counts_bytes));
    CHECK(ds4_gpu_inkling2_expert_route_map_rows_tensor(
        g_operator_context.moe.route_map, g_operator_context.moe.route_counts,
        tid, TOKENS, EXPERTS, USED));
    uint32_t counts[EXPERTS];
    uint32_t mapped_routes[EXPERTS * ROUTES];
    CHECK(ds4_gpu_tensor_read(g_operator_context.moe.route_counts, 0,
                              counts, sizeof(counts)));
    CHECK(ds4_gpu_tensor_read(g_operator_context.moe.route_map, 0,
                              mapped_routes, sizeof(mapped_routes)));
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
        const uint32_t expected_count = expert < USED ? TOKENS : 0u;
        CHECK(counts[expert] == expected_count);
        for (uint32_t rank = 0; rank < expected_count; rank++)
            CHECK(mapped_routes[expert * ROUTES + rank] == rank * USED + expert);
    }

    CHECK(ds4_gpu_tensor_write(gate_ref, 0, stale, output_bytes));
    CHECK(ds4_gpu_tensor_write(gate_gemm, 0, stale, output_bytes));
    CHECK(ds4_gpu_inkling2_expert_gemv_rows_tensor(
        gate_ref, g_full_shape_iq4_map, allocation, gate_offset, 20u,
        HIDDEN, gate_out_dim, EXPERTS, USED, TOKENS, 0, tid, tx));
    const char *prior_scalar_env = getenv("DS4_INKLING2_MOE_FORCE_SCALAR");
    if (prior_scalar_env) {
        const size_t prior_scalar_len = strlen(prior_scalar_env) + 1u;
        saved_scalar_env = malloc(prior_scalar_len);
        CHECK(saved_scalar_env != NULL);
        memcpy(saved_scalar_env, prior_scalar_env, prior_scalar_len);
    }
    scalar_env_modified = 1;
    CHECK(unsetenv("DS4_INKLING2_MOE_FORCE_SCALAR") == 0);
    CHECK(ds4_gpu_inkling2_expert_gemm_id_rows_tensor(
        gate_gemm, g_full_shape_iq4_map, allocation, gate_offset,
        HIDDEN, gate_out_dim, EXPERTS, USED, TOKENS, 0,
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tx));
    CHECK(ds4_gpu_inkling2_situ_rows_tensor(
        activation, gate_ref, TOKENS, USED, FFN, 1.25f, 0.75f));

    CHECK(ds4_gpu_tensor_write(down_ref, 0, stale, output_bytes));
    CHECK(ds4_gpu_tensor_write(down_gemm, 0, stale, output_bytes));
    CHECK(ds4_gpu_inkling2_expert_gemv_rows_tensor(
        down_ref, g_full_shape_iq4_map, allocation, down_offset, 20u,
        FFN, HIDDEN, EXPERTS, USED, TOKENS, 1, tid, activation));
    CHECK(ds4_gpu_inkling2_expert_gemm_id_rows_tensor(
        down_gemm, g_full_shape_iq4_map, allocation, down_offset,
        FFN, HIDDEN, EXPERTS, USED, TOKENS, 1,
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, activation));
    CHECK(ds4_gpu_tensor_read(gate_ref, 0, gate_a, output_bytes));
    CHECK(ds4_gpu_tensor_read(gate_gemm, 0, gate_b, output_bytes));
    CHECK(ds4_gpu_tensor_read(down_ref, 0, down_a, output_bytes));
    CHECK(ds4_gpu_tensor_read(down_gemm, 0, down_b, output_bytes));

    CHECK(setenv("DS4_INKLING2_MOE_FORCE_SCALAR", "1", 1) == 0);
    CHECK(ds4_gpu_inkling2_expert_gemm_id_rows_tensor(
        gate_gemm, g_full_shape_iq4_map, allocation, gate_offset,
        HIDDEN, gate_out_dim, EXPERTS, USED, TOKENS, 0,
        g_operator_context.moe.route_map,
        g_operator_context.moe.route_counts, tx));
    CHECK(ds4_gpu_tensor_read(gate_gemm, 0, gate_scalar, output_bytes));
    CHECK(unsetenv("DS4_INKLING2_MOE_FORCE_SCALAR") == 0);
    double max_matrix_delta = 0.0;
    uint64_t nonfinite_matrix_delta = 0;
    for (uint64_t i = 0; i < output_count; i++) {
        if (!isfinite(((float *)gate_b)[i]) || !isfinite(((float *)gate_scalar)[i])) {
            nonfinite_matrix_delta++;
            continue;
        }
        const double delta = fabs((double)((float *)gate_b)[i] -
                                  (double)((float *)gate_scalar)[i]);
        if (delta > max_matrix_delta) max_matrix_delta = delta;
    }
    fprintf(stderr,
            "Inkling2 full-shape IQ4_NL tiled/scalar: max_abs=%.9g nonfinite=%llu\n",
            max_matrix_delta, (unsigned long long)nonfinite_matrix_delta);
    CHECK(nonfinite_matrix_delta == 0);
    CHECK(max_matrix_delta > 1e-7);
    CHECK(report_expert_iq4_parity("gate/up", gate_a, gate_b,
                                  (size_t)output_count));
    CHECK(report_expert_iq4_parity("down", down_a, down_b,
                                  (size_t)output_count));
    ok = 1;

cleanup:
    if (scalar_env_modified) {
        if (saved_scalar_env) setenv("DS4_INKLING2_MOE_FORCE_SCALAR", saved_scalar_env, 1);
        else unsetenv("DS4_INKLING2_MOE_FORCE_SCALAR");
    }
    if (model_map_set &&
        !ds4_gpu_set_model_map(g_synthetic_moe_map, g_synthetic_moe_size)) {
        fprintf(stderr, "Inkling2 full-shape GEMM model map restore failed\n");
        ok = 0;
    }
    ds4_gpu_tensor_free(tid);
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(gate_ref);
    ds4_gpu_tensor_free(gate_gemm);
    ds4_gpu_tensor_free(activation);
    ds4_gpu_tensor_free(down_ref);
    ds4_gpu_tensor_free(down_gemm);
    free(input);
    free(gate_a);
    free(gate_b);
    free(gate_scalar);
    free(down_a);
    free(down_b);
    free(stale);
    free(stale_route_map);
    free(stale_route_counts);
    free(saved_scalar_env);
    return ok;
}

static int test_loop_mix_and_control_metal(void) {
    enum { HIDDEN = 2304, STATES = 5, QUERIES = 4, RANK = 64 };
    const uint64_t down_offset = 64u;
    const uint64_t down_bytes = (uint64_t)RANK * 2u * HIDDEN * sizeof(float);
    const uint64_t gain_offset = align_up_u64(down_offset + down_bytes, 64u);
    const uint64_t gain_bytes = (uint64_t)HIDDEN * RANK * sizeof(float);
    const uint64_t forget_offset = align_up_u64(gain_offset + gain_bytes, 64u);
    const uint64_t forget_bytes = gain_bytes;
    const uint64_t model_size = forget_offset + forget_bytes + 64u;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0 || model_size > SIZE_MAX) return 0;
    const uint64_t allocation = align_up_u64(model_size, (uint64_t)page);
    if (posix_memalign(&g_synthetic_loop_map, (size_t)page,
                       (size_t)allocation) != 0) return 0;
    memset(g_synthetic_loop_map, 0, (size_t)allocation);

    uint8_t *map = g_synthetic_loop_map;
    float *down = (float *)(void *)(map + down_offset);
    float *gain = (float *)(void *)(map + gain_offset);
    float *forget = (float *)(void *)(map + forget_offset);
    for (uint32_t row = 0; row < RANK; row++) {
        for (uint32_t col = 0; col < 2u * HIDDEN; col++) {
            down[(uint64_t)row * 2u * HIDDEN + col] =
                (float)((int32_t)((row * 7u + col * 3u) % 31u) - 15) / 256.0f;
        }
    }
    for (uint32_t row = 0; row < HIDDEN; row++) {
        for (uint32_t col = 0; col < RANK; col++) {
            gain[(uint64_t)row * RANK + col] =
                (float)((int32_t)((row * 5u + col * 11u) % 29u) - 14) / 64.0f;
            forget[(uint64_t)row * RANK + col] =
                (float)((int32_t)((row * 13u + col * 3u) % 27u) - 13) / 72.0f;
        }
    }
    if (!ds4_gpu_set_model_map(g_synthetic_loop_map, model_size)) {
        fprintf(stderr, "Inkling2 loop synthetic model map registration failed\n");
        return 0;
    }

    float states[STATES * HIDDEN], queries[QUERIES * HIDDEN];
    float expected_mix[QUERIES * HIDDEN], actual_mix[QUERIES * HIDDEN];
    for (uint32_t state = 0; state < STATES; state++) {
        for (uint32_t i = 0; i < HIDDEN; i++) {
            states[(uint64_t)state * HIDDEN + i] =
                sinf((float)(state + 1u) * (float)(i + 1u) * 0.13f) +
                (float)state * 0.08f;
        }
    }
    for (uint32_t query = 0; query < QUERIES; query++) {
        for (uint32_t i = 0; i < HIDDEN; i++) {
            queries[(uint64_t)query * HIDDEN + i] =
                cosf((float)(query + 2u) * (float)(i + 3u) * 0.071f);
        }
    }
    ds4_gpu_tensor *tstates = upload(states, sizeof(states));
    ds4_gpu_tensor *tqueries = upload(queries, sizeof(queries));
    ds4_gpu_tensor *tmix = ds4_gpu_tensor_alloc(sizeof(actual_mix));
    float max_mix_error = 0.0f;
    int mix_ok = tstates && tqueries && tmix &&
        ds4_inkling_mix_ref(expected_mix, states, queries, STATES, HIDDEN,
                            QUERIES, 1e-6f) &&
        ds4_inkling_mix_metal(tmix, tstates, tqueries, STATES, HIDDEN,
                              QUERIES, 1e-6f) &&
        ds4_gpu_tensor_read(tmix, 0, actual_mix, sizeof(actual_mix));
    for (uint32_t i = 0; mix_ok && i < QUERIES * HIDDEN; i++) {
        if (!near_record(actual_mix[i], expected_mix[i], 4e-5f,
                         &max_mix_error)) {
            fprintf(stderr, "Inkling2 loop mix mismatch i=%u got=%g expected=%g\n",
                    i, actual_mix[i], expected_mix[i]);
            mix_ok = 0;
        }
    }

    enum { MIX_ROWS = 3 };
    float batch_states[MIX_ROWS * STATES * HIDDEN];
    float batch_current[MIX_ROWS * HIDDEN];
    float reference_states[MIX_ROWS * STATES * HIDDEN];
    float expected_batch_mix[QUERIES * MIX_ROWS * HIDDEN];
    float actual_batch_mix[QUERIES * MIX_ROWS * HIDDEN];
    ds4_gpu_tensor *tbatch_states = NULL;
    ds4_gpu_tensor *tbatch_current_for_mix = NULL;
    ds4_gpu_tensor *tbatch_mix = NULL;
    float max_batch_mix_error = 0.0f;
    for (uint32_t scenario = 0; mix_ok && scenario < 2u; scenario++) {
        const uint32_t state_count = scenario == 0u ? STATES : STATES;
        const uint32_t source_states = scenario == 0u ? STATES : STATES - 1u;
        for (uint32_t row = 0; row < MIX_ROWS; row++) {
            for (uint32_t state = 0; state < source_states; state++) {
                for (uint32_t i = 0; i < HIDDEN; i++) {
                    const uint64_t index =
                        ((uint64_t)row * source_states + state) * HIDDEN + i;
                    batch_states[index] =
                        sinf((float)(state + row + 1u) *
                             (float)(i + 1u) * 0.071f) +
                        (float)(row + state) * 0.03f;
                }
            }
            for (uint32_t i = 0; i < HIDDEN; i++)
                batch_current[(uint64_t)row * HIDDEN + i] =
                    cosf((float)(row + 2u) * (float)(i + 1u) * 0.053f);

            for (uint32_t state = 0; state < state_count; state++) {
                const uint32_t source_state = scenario == 0u ? state :
                    (state + 1u == state_count ? 0u : source_states - state - 1u);
                const float *source = scenario == 0u ?
                    batch_states + ((uint64_t)row * source_states + source_state) * HIDDEN :
                    (state + 1u == state_count ?
                        batch_current + (uint64_t)row * HIDDEN :
                        batch_states + ((uint64_t)row * source_states + source_state) * HIDDEN);
                memcpy(reference_states +
                        ((uint64_t)row * state_count + state) * HIDDEN,
                       source, (size_t)HIDDEN * sizeof(float));
            }
            float reference_row[QUERIES * HIDDEN];
            if (!ds4_inkling_mix_ref(reference_row,
                    reference_states + (uint64_t)row * state_count * HIDDEN,
                    queries, state_count, HIDDEN, QUERIES, 1e-6f)) {
                mix_ok = 0;
                break;
            }
            for (uint32_t query = 0; query < QUERIES; query++)
                memcpy(expected_batch_mix +
                        ((uint64_t)query * MIX_ROWS + row) * HIDDEN,
                       reference_row + (uint64_t)query * HIDDEN,
                       (size_t)HIDDEN * sizeof(float));
        }
        if (!mix_ok) break;
        ds4_gpu_tensor_free(tbatch_states);
        ds4_gpu_tensor_free(tbatch_current_for_mix);
        ds4_gpu_tensor_free(tbatch_mix);
        const uint64_t state_bytes =
            (uint64_t)MIX_ROWS * source_states * HIDDEN * sizeof(float);
        tbatch_states = upload(batch_states, state_bytes);
        tbatch_current_for_mix = scenario == 0u ? NULL :
            upload(batch_current, sizeof(batch_current));
        tbatch_mix = ds4_gpu_tensor_alloc(sizeof(actual_batch_mix));
        mix_ok = tbatch_states && tbatch_mix &&
            (scenario == 0u || tbatch_current_for_mix);
        for (uint32_t query = 0; mix_ok && query < QUERIES; query++) {
            ds4_gpu_tensor *query_output = ds4_gpu_tensor_view(tbatch_mix,
                (uint64_t)query * MIX_ROWS * HIDDEN * sizeof(float),
                (uint64_t)MIX_ROWS * HIDDEN * sizeof(float));
            mix_ok = query_output && ds4_gpu_inkling2_mix_prefill_tensor(
                query_output, tbatch_states, tbatch_current_for_mix,
                tqueries, state_count, source_states, HIDDEN, QUERIES,
                MIX_ROWS, query,
                scenario != 0u, scenario != 0u, 1e-6f);
            ds4_gpu_tensor_free(query_output);
        }
        mix_ok = mix_ok && ds4_gpu_tensor_read(tbatch_mix, 0,
            actual_batch_mix, sizeof(actual_batch_mix));
        for (uint32_t i = 0; mix_ok && i < QUERIES * MIX_ROWS * HIDDEN; i++) {
            if (!near_record(actual_batch_mix[i], expected_batch_mix[i],
                             4e-5f, &max_batch_mix_error)) {
                fprintf(stderr,
                        "Inkling2 prefill mix mismatch scenario=%u i=%u\n",
                        scenario, i);
                mix_ok = 0;
            }
        }
    }
    ds4_gpu_tensor_free(tbatch_states);
    ds4_gpu_tensor_free(tbatch_current_for_mix);
    ds4_gpu_tensor_free(tbatch_mix);

    float current[HIDDEN], message[HIDDEN], expected_control[HIDDEN];
    float actual_control[HIDDEN];
    for (uint32_t i = 0; i < HIDDEN; i++) {
        current[i] = sinf((float)i * 0.17f) * 0.8f;
        message[i] = cosf((float)i * 0.11f) * 0.6f;
    }
    ds4_gpu_tensor *tcurrent = upload(current, sizeof(current));
    ds4_gpu_tensor *tmessage = upload(message, sizeof(message));
    ds4_gpu_tensor *tcontrol = ds4_gpu_tensor_alloc(sizeof(actual_control));
    float max_control_error = 0.0f;
    int control_ok = mix_ok && tcurrent && tmessage && tcontrol &&
        ds4_inkling_control_ref(expected_control, current, message, down,
                                gain, forget, HIDDEN, RANK, 1e-6f) &&
        ds4_inkling_control_metal_with_workspace(tcontrol, tcurrent, tmessage,
                                  g_synthetic_loop_map, model_size,
                                  down_offset, gain_offset, forget_offset,
                                  HIDDEN, RANK, 1e-6f,
                                  &g_operator_context.control) &&
        ds4_gpu_tensor_read(tcontrol, 0, actual_control,
                            sizeof(actual_control));
    for (uint32_t i = 0; control_ok && i < HIDDEN; i++) {
        if (!near_record(actual_control[i], expected_control[i], 5e-5f,
                         &max_control_error)) {
            fprintf(stderr,
                    "Inkling2 loop control mismatch i=%u got=%g expected=%g\n",
                    i, actual_control[i], expected_control[i]);
            control_ok = 0;
        }
    }

    enum { CONTROL_ROWS = 3 };
    float control_batch_current[CONTROL_ROWS * HIDDEN];
    float control_batch_message[CONTROL_ROWS * HIDDEN];
    float expected_batch_control[CONTROL_ROWS * HIDDEN];
    float actual_batch_control[CONTROL_ROWS * HIDDEN];
    for (uint32_t row = 0; row < CONTROL_ROWS; row++) {
        for (uint32_t i = 0; i < HIDDEN; i++) {
            const uint64_t index = (uint64_t)row * HIDDEN + i;
            control_batch_current[index] =
                sinf((float)(i + row * 3u) * 0.17f) * 0.8f;
            control_batch_message[index] =
                cosf((float)(i + row * 5u) * 0.11f) * 0.6f;
        }
        if (!ds4_inkling_control_ref(
                expected_batch_control + (uint64_t)row * HIDDEN,
                control_batch_current + (uint64_t)row * HIDDEN,
                control_batch_message + (uint64_t)row * HIDDEN,
                down, gain, forget, HIDDEN, RANK, 1e-6f)) control_ok = 0;
    }
    ds4_gpu_tensor *tbatch_current = upload(
        control_batch_current, sizeof(control_batch_current));
    ds4_gpu_tensor *tbatch_message = upload(
        control_batch_message, sizeof(control_batch_message));
    ds4_gpu_tensor *tbatch_control =
        ds4_gpu_tensor_alloc(sizeof(actual_batch_control));
    ds4_inkling_gpu_control_workspace batch_workspace = {0};
    batch_workspace.normalized = ds4_gpu_tensor_alloc(
        (uint64_t)CONTROL_ROWS * 2u * HIDDEN * sizeof(float));
    batch_workspace.bottleneck = ds4_gpu_tensor_alloc(
        (uint64_t)CONTROL_ROWS * RANK * sizeof(float));
    batch_workspace.gain = ds4_gpu_tensor_alloc(
        (uint64_t)CONTROL_ROWS * HIDDEN * sizeof(float));
    batch_workspace.forget = ds4_gpu_tensor_alloc(
        (uint64_t)CONTROL_ROWS * HIDDEN * sizeof(float));
    float max_batch_control_error = 0.0f;
    int batch_control_ok = control_ok && tbatch_current && tbatch_message &&
        tbatch_control && batch_workspace.normalized &&
        batch_workspace.bottleneck && batch_workspace.gain &&
        batch_workspace.forget &&
        ds4_inkling_control_metal_batch_with_workspace(
            tbatch_control, tbatch_current, tbatch_message,
            g_synthetic_loop_map, model_size, down_offset, gain_offset,
            forget_offset, HIDDEN, RANK, CONTROL_ROWS, 1e-6f,
            &batch_workspace) &&
        ds4_gpu_tensor_read(tbatch_control, 0, actual_batch_control,
                            sizeof(actual_batch_control));
    for (uint32_t i = 0; batch_control_ok &&
                         i < CONTROL_ROWS * HIDDEN; i++) {
        if (!near_record(actual_batch_control[i], expected_batch_control[i],
                         5e-5f, &max_batch_control_error)) {
            fprintf(stderr,
                    "Inkling2 batched loop control mismatch i=%u got=%g expected=%g\n",
                    i, actual_batch_control[i], expected_batch_control[i]);
            batch_control_ok = 0;
        }
    }
    ds4_gpu_tensor_free(tbatch_current);
    ds4_gpu_tensor_free(tbatch_message);
    ds4_gpu_tensor_free(tbatch_control);
    ds4_gpu_tensor_free(batch_workspace.normalized);
    ds4_gpu_tensor_free(batch_workspace.bottleneck);
    ds4_gpu_tensor_free(batch_workspace.gain);
    ds4_gpu_tensor_free(batch_workspace.forget);
    control_ok = control_ok && batch_control_ok;

    if (mix_ok && control_ok) {
        printf("Inkling2 loop mix/control max absolute error: %.9g / %.9g (batch %.9g)\n",
               max_mix_error, max_control_error, max_batch_control_error);
    }

    ds4_gpu_tensor_free(tstates);
    ds4_gpu_tensor_free(tqueries);
    ds4_gpu_tensor_free(tmix);
    ds4_gpu_tensor_free(tcurrent);
    ds4_gpu_tensor_free(tmessage);
    ds4_gpu_tensor_free(tcontrol);
    return mix_ok && control_ok;
}

static void make_small_plan(ds4_inkling_plan *plan) {
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

static int test_gpu_context_owns_independent_visit_state(void) {
    int ok = 1;
    ds4_inkling_plan plan;
    ds4_inkling_gpu_context context;
    uint64_t bytes_2048 = 0, bytes_8192 = 0, bytes_65536 = 0, bytes_max = 0;
    memset(&context, 0, sizeof(context));
    make_small_plan(&plan);

    CHECK(ds4_inkling_gpu_context_required_bytes(&plan, 2048u, &bytes_2048));
    CHECK(ds4_inkling_gpu_context_required_bytes(&plan, 8192u, &bytes_8192));
    CHECK(bytes_8192 > bytes_2048);
    CHECK(bytes_8192 > 690u * 1024u * 1024u &&
          bytes_8192 < 720u * 1024u * 1024u);
    CHECK(ds4_inkling_gpu_context_required_bytes(&plan, 65536u,
                                                &bytes_65536));
    CHECK(bytes_65536 > bytes_8192);
    CHECK(bytes_65536 > (2ull << 30) && bytes_65536 < (3ull << 30));
    CHECK(ds4_inkling_gpu_context_required_bytes(&plan,
            DS4_INKLING_CONTEXT_MAX, &bytes_max));
    CHECK(bytes_max > bytes_65536);
    CHECK(bytes_max > (32ull << 30) && bytes_max < (33ull << 30));
    CHECK(!ds4_inkling_gpu_context_required_bytes(&plan,
            DS4_INKLING_CONTEXT_MAX + 1u, &bytes_max));
    CHECK(ds4_inkling_gpu_context_required_bytes(&plan, 8192u, &bytes_8192));
    printf("Inkling2 8192-token GPU context allocation: %llu bytes (%.2f MiB)\n",
           (unsigned long long)bytes_8192,
           (double)bytes_8192 / (1024.0 * 1024.0));
    CHECK(ds4_inkling_gpu_context_init(&context, &plan, 2048u));
    CHECK(context.plan.visit_count == DS4_INKLING_VISITS);
    CHECK(context.weight_owner_count == DS4_INKLING_PHYSICAL_LAYERS);
    CHECK(context.slots[6].capacity == DS4_INKLING_SLIDING_WINDOW);
    CHECK(context.slots[24].capacity == DS4_INKLING_SLIDING_WINDOW);
    CHECK(context.slots[36].capacity == DS4_INKLING_SLIDING_WINDOW);
    CHECK(context.slots[5].capacity == 2048u);
    CHECK(context.slots[0].capacity == DS4_INKLING_SLIDING_WINDOW);
    CHECK(context.slots[6].key_offset != context.slots[24].key_offset);
    CHECK(context.slots[24].key_offset != context.slots[36].key_offset);
    CHECK(context.slots[6].value_offset != context.slots[24].value_offset);
    CHECK(context.slots[24].value_offset != context.slots[36].value_offset);
    CHECK(context.slots[11].capacity == 2048u);
    CHECK(context.slots[29].capacity == 2048u);
    CHECK(context.slots[41].capacity == 2048u);
    CHECK(context.slots[11].key_offset != context.slots[29].key_offset);
    CHECK(context.slots[29].key_offset != context.slots[41].key_offset);
    CHECK(context.slots[6].conv_offset[DS4_INKLING_CONV_KEY] !=
          context.slots[24].conv_offset[DS4_INKLING_CONV_KEY]);
    CHECK(context.weight_owner[6] == 6u);
    CHECK(context.weight_owner[18] == 18u);
    CHECK(context.current_blocks && context.previous_blocks &&
          context.current_blocks != context.previous_blocks);
    CHECK(context.input_token && context.embedding && context.activation &&
          context.logits && context.scratch);
    CHECK(context.batch.relative_bias_storage != NULL);
    CHECK(context.control.normalized && context.control.bottleneck &&
          context.control.gain && context.control.forget);
    CHECK(context.moe.logits && context.moe.ids && context.moe.weights &&
          context.moe.gate_up && context.moe.activation &&
          context.moe.expert_outputs && context.moe.zero_bias);
    CHECK(ds4_gpu_tensor_contents(context.moe.gate_up) !=
          ds4_gpu_tensor_contents(context.moe.activation));

    for (uint32_t i = 0; i < DS4_INKLING_CACHE_SLOTS; i++) {
        CHECK(context.slots[i].key && context.slots[i].value);
        CHECK(context.slots[i].length == 0u);
        CHECK(context.slots[i].next_position == 0u);
        for (uint32_t j = i + 1u; j < DS4_INKLING_CACHE_SLOTS; j++) {
            CHECK(context.slots[i].key_offset != context.slots[j].key_offset);
            CHECK(context.slots[i].value_offset != context.slots[j].value_offset);
            CHECK(context.slots[i].conv_offset[DS4_INKLING_CONV_FFN_OUT] !=
                  context.slots[j].conv_offset[DS4_INKLING_CONV_FFN_OUT]);
        }
    }

cleanup:
    ds4_inkling_gpu_context_free(&context);
    return ok;
}

static int test_qk_and_relative_bias(void) {
    const float input[] = { 3, 4, 0, 0, 0, -2, 1, 0 };
    const float learned_scale[] = { 2, 1, 1, 1, 1, 0.5f, 1, 1 };
    const uint64_t positions[] = { 7, 31 };
    float expected[8], actual[8];
    if (!ds4_inkling_prepare_qk(expected, input, learned_scale, positions,
        2, 4, 3.0f, 0.5f, 8.0f, 1)) return 0;
    ds4_gpu_tensor *tx = upload(input, sizeof(input));
    ds4_gpu_tensor *ts = upload(learned_scale, sizeof(learned_scale));
    ds4_gpu_tensor *tp = upload(positions, sizeof(positions));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(actual));
    int ok = tx && ts && tp && to &&
        ds4_gpu_inkling2_qk_prepare_tensor(
            to, tx, ts, tp, 2, 4, 3.0f, 0.5f, 8.0f, 1) &&
        ds4_gpu_tensor_read(to, 0, actual, sizeof(actual));
    for (uint32_t i = 0; ok && i < 8; i++) {
        if (!near_record(actual[i], expected[i], 2e-5f,
                         &g_m4_qk_relative_max_error)) {
            fprintf(stderr, "Inkling2 QK prepare mismatch i=%u got=%g expected=%g\n",
                    i, actual[i], expected[i]);
            ok = 0;
        }
    }
    float scalar_actual[4];
    ok = ok && ds4_gpu_inkling2_qk_prepare_decode_tensor(
        to, tx, ts, 1, 4, 3.0f, 0.5f, 8.0f, positions[0], 1) &&
        ds4_gpu_tensor_read(to, 0, scalar_actual, sizeof(scalar_actual));
    for (uint32_t i = 0; ok && i < 4; i++) {
        if (!near_record(scalar_actual[i], expected[i], 2e-5f,
                         &g_m4_qk_relative_max_error)) {
            fprintf(stderr,
                    "Inkling2 scalar-position QK prepare mismatch i=%u got=%g expected=%g\n",
                    i, scalar_actual[i], expected[i]);
            ok = 0;
        }
    }

    const float prefill_input[] = {
        3.0f, 4.0f, 1.0f, -2.0f,
        2.0f, 1.0f, -3.0f, 4.0f,
        5.0f, 2.0f, -2.0f, -2.0f,
    };
    const float prefill_scale[] = { 1.0f, 0.5f, 2.0f, 1.5f };
    const float expanded_prefill_scale[] = {
        1.0f, 0.5f, 2.0f, 1.5f,
        1.0f, 0.5f, 2.0f, 1.5f,
        1.0f, 0.5f, 2.0f, 1.5f,
    };
    const uint64_t prefill_positions[] = { 5u, 5u, 6u, 6u, 7u, 7u };
    const uint64_t prefill_first_position = 5u;
    float prefill_expected[12], prefill_actual[12];
    if (!ds4_inkling_prepare_qk(prefill_expected, prefill_input,
            expanded_prefill_scale, prefill_positions,
            6u, 2u, 3.0f, 0.5f, 8.0f, 1)) return 0;
    ds4_gpu_tensor *tpi = upload(prefill_input, sizeof(prefill_input));
    ds4_gpu_tensor *tps = upload(prefill_scale, sizeof(prefill_scale));
    ds4_gpu_tensor *tpo = ds4_gpu_tensor_alloc(sizeof(prefill_actual));
    ok = tpi && tps && tpo &&
        ds4_gpu_inkling2_qk_prepare_prefill_tensor(
            tpo, tpi, tps, 3u, 2u, 2u, 3.0f, 0.5f, 8.0f,
            prefill_first_position, 1) &&
        ds4_gpu_tensor_read(tpo, 0, prefill_actual, sizeof(prefill_actual));
    for (uint32_t i = 0; ok && i < 12u; i++) {
        if (!near_record(prefill_actual[i], prefill_expected[i], 2e-5f,
                         &g_m4_qk_relative_max_error)) {
            fprintf(stderr,
                    "Inkling2 prefill QK prepare mismatch i=%u got=%g expected=%g\n",
                    i, prefill_actual[i], prefill_expected[i]);
            ok = 0;
        }
    }
    ds4_gpu_tensor_free(tpi);
    ds4_gpu_tensor_free(tps);
    ds4_gpu_tensor_free(tpo);
    if (!ok) {
        ds4_gpu_tensor_free(tx);
        ds4_gpu_tensor_free(ts);
        ds4_gpu_tensor_free(tp);
        ds4_gpu_tensor_free(to);
        return 0;
    }
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(ts);
    ds4_gpu_tensor_free(tp);
    ds4_gpu_tensor_free(to);
    if (!ok) return 0;

    const float projected[] = { 1, 2, -1, 0.5f };
    const float table[] = { 1, 2, 3, -2, 1, 0 };
    const int64_t distances[] = { 0, 2, 3, -1 };
    float expected_bias[4], actual_bias[4];
    if (!ds4_inkling_relative_bias(expected_bias, projected, table, distances,
                                    1, 2, 2, 3, 2)) return 0;
    ds4_gpu_tensor *tr = upload(projected, sizeof(projected));
    ds4_gpu_tensor *tt = upload(table, sizeof(table));
    ds4_gpu_tensor *td = upload(distances, sizeof(distances));
    ds4_gpu_tensor *tb = ds4_gpu_tensor_alloc(sizeof(actual_bias));
    ok = tr && tt && td && tb &&
        ds4_gpu_inkling2_relative_bias_tensor(tb, tr, tt, td,
            1, 2, 2, 3, 2, 2) &&
        ds4_gpu_tensor_read(tb, 0, actual_bias, sizeof(actual_bias));
    for (uint32_t i = 0; ok && i < 4; i++) {
        if (!near_record(actual_bias[i], expected_bias[i], 2e-6f,
                         &g_m4_qk_relative_max_error)) {
            fprintf(stderr, "Inkling2 relative bias mismatch i=%u\n", i);
            ok = 0;
        }
    }

    const float batch_states[] = {
        1.0f, 2.0f, -1.0f, 0.5f,
        2.0f, -1.0f, 0.25f, 3.0f,
    };
    const float batch_projection[] = { 1.0f, 2.0f, 3.0f, -2.0f, 1.0f, 0.0f };
    const int64_t shared_distances[] = { 0, 2 };
    const int64_t expanded_distances[] = { 0, 2, 0, 2 };
    float expected_batch_bias[8], actual_batch_bias[8];
    ok = ok && ds4_inkling_relative_bias(expected_batch_bias, batch_states,
        batch_projection, expanded_distances, 2, 2, 2, 3, 2);
    ds4_gpu_tensor *tbrs = upload(batch_states, sizeof(batch_states));
    ds4_gpu_tensor *tbrp = upload(batch_projection, sizeof(batch_projection));
    ds4_gpu_tensor *tbrd = upload(shared_distances, sizeof(shared_distances));
    ds4_gpu_tensor *tbbo = ds4_gpu_tensor_alloc(sizeof(actual_batch_bias));
    ok = ok && tbrs && tbrp && tbrd && tbbo &&
        ds4_gpu_inkling2_relative_bias_tensor(tbbo, tbrs, tbrp, tbrd,
            2u, 2u, 2u, 3u, 2u, 0u) &&
        ds4_gpu_tensor_read(tbbo, 0, actual_batch_bias,
                            sizeof(actual_batch_bias));
    for (uint32_t i = 0; ok && i < 8u; i++) {
        if (!near_record(actual_batch_bias[i], expected_batch_bias[i], 2e-6f,
                         &g_m4_qk_relative_max_error)) {
            fprintf(stderr,
                    "Inkling2 shared-distance relative bias mismatch i=%u\n", i);
            ok = 0;
        }
    }
    ds4_gpu_tensor_free(tbrs);
    ds4_gpu_tensor_free(tbrp);
    ds4_gpu_tensor_free(tbrd);
    ds4_gpu_tensor_free(tbbo);
    ds4_gpu_tensor_free(tr);
    ds4_gpu_tensor_free(tt);
    ds4_gpu_tensor_free(td);
    ds4_gpu_tensor_free(tb);
    return ok;
}

static int test_attention_output_gate(void) {
    enum { N = 8 };
    const float attention[N] = { -4.0f, -1.0f, 0.0f, 0.5f, 2.0f, 5.0f, -3.0f, 1.0f };
    const float gates[N] = { -INFINITY, INFINITY, 0.0f, -0.5f, 0.5f, 100.0f, -100.0f, NAN };
    float actual[N];
    ds4_gpu_tensor *ta = upload(attention, sizeof(attention));
    ds4_gpu_tensor *tg = upload(gates, sizeof(gates));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(actual));
    int ok = ta && tg && to &&
        ds4_gpu_inkling2_attention_output_gate_tensor(to, ta, tg, N) &&
        ds4_gpu_tensor_read(to, 0, actual, sizeof(actual));
    for (uint32_t i = 0; ok && i < N; i++) {
        if (isnan(gates[i])) {
            if (!isnan(actual[i])) ok = 0;
        } else if (!near_record(actual[i], attention[i] * sigmoid_ref(gates[i]),
                                2e-6f, &g_m4_gate_max_error)) {
            fprintf(stderr, "Inkling2 attention gate mismatch i=%u\n", i);
            ok = 0;
        }
    }
    ds4_gpu_tensor_free(ta);
    ds4_gpu_tensor_free(tg);
    ds4_gpu_tensor_free(to);
    return ok;
}

static int test_attention_prefill_matches_decode(void) {
    enum { CAPACITY = 8, MAX_TOKENS = 4, QH = 4, KVH = 2,
           DIM = 4, EXTENT = 4, KV_WIDTH = KVH * DIM,
           QUERY_WIDTH = QH * DIM };
    static const float half_values[8] = {
        0.25f, 0.5f, 1.0f, -0.25f, -0.5f, -1.0f, 1.5f, 2.0f,
    };
    static const uint16_t half_bits[8] = {
        0x3400u, 0x3800u, 0x3c00u, 0xb400u,
        0xb800u, 0xbc00u, 0x3e00u, 0x4000u,
    };
    const uint64_t cache_elements = (uint64_t)CAPACITY * KV_WIDTH;
    const uint64_t cache_bytes = cache_elements * sizeof(uint16_t);
    const uint64_t query_row_bytes = (uint64_t)QUERY_WIDTH * sizeof(float);
    const uint64_t key_row_bytes = (uint64_t)KV_WIDTH * sizeof(float);
    const uint64_t bias_row_bytes = (uint64_t)QH * EXTENT * sizeof(float);
    const uint64_t sink_row_bytes = (uint64_t)QH * sizeof(float);
    const uint64_t output_bytes =
        (uint64_t)MAX_TOKENS * QUERY_WIDTH * sizeof(float);
    uint16_t old_keys[CAPACITY * KV_WIDTH] = { 0 };
    uint16_t old_values[CAPACITY * KV_WIDTH] = { 0 };
    uint16_t reference_keys[CAPACITY * KV_WIDTH] = { 0 };
    uint16_t reference_values[CAPACITY * KV_WIDTH] = { 0 };
    float query[MAX_TOKENS * QUERY_WIDTH];
    float new_keys[MAX_TOKENS * KV_WIDTH];
    float new_values[MAX_TOKENS * KV_WIDTH];
    float relative_bias[MAX_TOKENS * QH * EXTENT];
    float sinks[MAX_TOKENS * QH];
    float expected[MAX_TOKENS * QUERY_WIDTH];
    float actual[MAX_TOKENS * QUERY_WIDTH];
    ds4_gpu_tensor *tq = NULL, *tk = NULL, *tv = NULL;
    ds4_gpu_tensor *tnk = NULL, *tnv = NULL, *trb = NULL, *tsink = NULL;
    ds4_gpu_tensor *tactual = NULL, *tqr = NULL, *trbr = NULL;
    ds4_gpu_tensor *tsinkr = NULL, *treference = NULL;
    ds4_gpu_tensor *treference_keys = NULL, *treference_values = NULL;
    int ok = 1;

    for (uint32_t scenario = 0; scenario < 2u; scenario++) {
        const uint32_t old_count = scenario == 0u ? 3u : 6u;
        const uint32_t n_tokens = scenario == 0u ? 3u : 4u;
        const uint32_t sliding_window = scenario == 0u ? 0u : 4u;
        const uint64_t first_position = old_count;
        memset(old_keys, 0, sizeof(old_keys));
        memset(old_values, 0, sizeof(old_values));
        memset(reference_keys, 0, sizeof(reference_keys));
        memset(reference_values, 0, sizeof(reference_values));

        for (uint32_t position = 0; position < old_count; position++) {
            for (uint32_t head = 0; head < KVH; head++) {
                for (uint32_t dim = 0; dim < DIM; dim++) {
                    const uint32_t index =
                        (position * 7u + head * 3u + dim) % 8u;
                    const uint64_t offset =
                        (uint64_t)position * KV_WIDTH + head * DIM + dim;
                    old_keys[offset] = reference_keys[offset] = half_bits[index];
                    const uint32_t value_index =
                        (position * 5u + head * 7u + dim * 3u + 1u) % 8u;
                    old_values[offset] = reference_values[offset] =
                        half_bits[value_index];
                }
            }
        }

        for (uint32_t token = 0; token < n_tokens; token++) {
            for (uint32_t head = 0; head < QH; head++) {
                for (uint32_t dim = 0; dim < DIM; dim++) {
                    const uint32_t query_index =
                        (token * 5u + head * 3u + dim + 1u) % 8u;
                    query[(uint64_t)token * QUERY_WIDTH + head * DIM + dim] =
                        half_values[query_index] * 0.25f;
                }
                sinks[(uint64_t)token * QH + head] =
                    half_values[(token + head * 2u) % 8u] * 0.125f;
                for (uint32_t distance = 0; distance < EXTENT; distance++) {
                    const uint32_t bias_index =
                        (token * 3u + head + distance * 5u) % 8u;
                    relative_bias[((uint64_t)token * QH + head) * EXTENT +
                                  distance] = half_values[bias_index] * 0.0625f;
                }
            }
            for (uint32_t head = 0; head < KVH; head++) {
                for (uint32_t dim = 0; dim < DIM; dim++) {
                    const uint32_t key_index =
                        (token * 7u + head * 3u + dim * 5u + 2u) % 8u;
                    const uint32_t value_index =
                        (token * 3u + head * 5u + dim * 7u + 4u) % 8u;
                    const uint64_t batch_offset =
                        (uint64_t)token * KV_WIDTH + head * DIM + dim;
                    const uint64_t cache_offset =
                        ((first_position + token) % CAPACITY) * KV_WIDTH +
                        head * DIM + dim;
                    new_keys[batch_offset] = half_values[key_index];
                    new_values[batch_offset] = half_values[value_index];
                    reference_keys[cache_offset] = half_bits[key_index];
                    reference_values[cache_offset] = half_bits[value_index];
                }
            }
        }

        tq = upload(query, (uint64_t)n_tokens * query_row_bytes);
        tk = upload(old_keys, cache_bytes);
        tv = upload(old_values, cache_bytes);
        treference_keys = upload(reference_keys, cache_bytes);
        treference_values = upload(reference_values, cache_bytes);
        tnk = upload(new_keys, (uint64_t)n_tokens * key_row_bytes);
        tnv = upload(new_values, (uint64_t)n_tokens * key_row_bytes);
        trb = upload(relative_bias,
                     (uint64_t)n_tokens * bias_row_bytes);
        tsink = upload(sinks, (uint64_t)n_tokens * sink_row_bytes);
        tactual = ds4_gpu_tensor_alloc(output_bytes);
        tqr = ds4_gpu_tensor_alloc(query_row_bytes);
        trbr = ds4_gpu_tensor_alloc(bias_row_bytes);
        tsinkr = ds4_gpu_tensor_alloc(sink_row_bytes);
        treference = ds4_gpu_tensor_alloc(query_row_bytes);
        CHECK(tq && tk && tv && treference_keys && treference_values &&
              tnk && tnv && trb && tsink && tactual && tqr && trbr &&
              tsinkr && treference);
        CHECK(ds4_gpu_inkling2_attention_prefill_tensor(
            tactual, tq, tk, tv, tnk, tnv, trb, tsink,
            QH, KVH, DIM, CAPACITY, old_count, n_tokens,
            first_position - old_count, first_position,
            sliding_window, EXTENT));
        CHECK(ds4_gpu_tensor_read(tactual, 0, actual,
                                  (uint64_t)n_tokens * query_row_bytes));
        CHECK(ds4_gpu_inkling2_kv_append_rows_tensor(
            tk, tv, tnk, tnv, n_tokens, KVH, DIM, CAPACITY,
            first_position));
        uint16_t appended_keys[CAPACITY * KV_WIDTH];
        uint16_t appended_values[CAPACITY * KV_WIDTH];
        CHECK(ds4_gpu_tensor_read(tk, 0, appended_keys, cache_bytes));
        CHECK(ds4_gpu_tensor_read(tv, 0, appended_values, cache_bytes));
        for (uint64_t i = 0; i < cache_elements; i++) {
            CHECK(appended_keys[i] == reference_keys[i]);
            CHECK(appended_values[i] == reference_values[i]);
        }

        for (uint32_t token = 0; token < n_tokens; token++) {
            CHECK(ds4_gpu_tensor_write(tqr, 0,
                query + (uint64_t)token * QUERY_WIDTH, query_row_bytes));
            CHECK(ds4_gpu_tensor_write(trbr, 0,
                relative_bias + (uint64_t)token * QH * EXTENT,
                bias_row_bytes));
            CHECK(ds4_gpu_tensor_write(tsinkr, 0,
                sinks + (uint64_t)token * QH, sink_row_bytes));
            const uint32_t total = old_count + token + 1u;
            const uint32_t count = total < CAPACITY ? total : CAPACITY;
            const uint64_t oldest = first_position + token + 1u - count;
            CHECK(ds4_gpu_inkling2_attention_decode_tensor(
                treference, tqr, treference_keys, treference_values,
                trbr, tsinkr, QH, KVH, DIM, CAPACITY, count,
                oldest, first_position + token, sliding_window, EXTENT));
            CHECK(ds4_gpu_tensor_read(treference, 0,
                expected + (uint64_t)token * QUERY_WIDTH, query_row_bytes));
            for (uint32_t i = 0; i < QUERY_WIDTH; i++) {
                const float got = actual[(uint64_t)token * QUERY_WIDTH + i];
                const float ref = expected[(uint64_t)token * QUERY_WIDTH + i];
                if (!near_record(got, ref, 2e-6f,
                                 &g_m4_attention_max_error)) {
                    fprintf(stderr,
                        "Inkling2 batched attention mismatch scenario=%u token=%u i=%u got=%g expected=%g\n",
                        scenario, token, i, got, ref);
                    ok = 0;
                    goto cleanup;
                }
            }
        }

        ds4_gpu_tensor_free(tq); tq = NULL;
        ds4_gpu_tensor_free(tk); tk = NULL;
        ds4_gpu_tensor_free(tv); tv = NULL;
        ds4_gpu_tensor_free(treference_keys); treference_keys = NULL;
        ds4_gpu_tensor_free(treference_values); treference_values = NULL;
        ds4_gpu_tensor_free(tnk); tnk = NULL;
        ds4_gpu_tensor_free(tnv); tnv = NULL;
        ds4_gpu_tensor_free(trb); trb = NULL;
        ds4_gpu_tensor_free(tsink); tsink = NULL;
        ds4_gpu_tensor_free(tactual); tactual = NULL;
        ds4_gpu_tensor_free(tqr); tqr = NULL;
        ds4_gpu_tensor_free(trbr); trbr = NULL;
        ds4_gpu_tensor_free(tsinkr); tsinkr = NULL;
        ds4_gpu_tensor_free(treference); treference = NULL;
    }

cleanup:
    ds4_gpu_tensor_free(tq);
    ds4_gpu_tensor_free(tk);
    ds4_gpu_tensor_free(tv);
    ds4_gpu_tensor_free(treference_keys);
    ds4_gpu_tensor_free(treference_values);
    ds4_gpu_tensor_free(tnk);
    ds4_gpu_tensor_free(tnv);
    ds4_gpu_tensor_free(trb);
    ds4_gpu_tensor_free(tsink);
    ds4_gpu_tensor_free(tactual);
    ds4_gpu_tensor_free(tqr);
    ds4_gpu_tensor_free(trbr);
    ds4_gpu_tensor_free(tsinkr);
    ds4_gpu_tensor_free(treference);
    return ok;
}

static int test_attention_and_shortconv(void) {
    ds4_inkling_plan plan;
    make_small_plan(&plan);
    const ds4_inkling_state_config config = {
        .context_capacity = 16,
        .sliding_window = 4,
        .kv_width = 4,
        .conv_width = { 4, 4, 4, 4 },
    };
    ds4_inkling_state_session session;
    if (!ds4_inkling_state_session_init(&session, &plan, &config)) return 0;
    ds4_inkling_state_slot *slot = &session.slots[0];
    for (uint32_t pos = 0; pos < 6; pos++) {
        const uint16_t key[4] = {
            (uint16_t)(0x3c00u + pos), (uint16_t)(0x3800u + pos),
            (uint16_t)(0xbc00u + pos), (uint16_t)(0x3400u + pos),
        };
        const uint16_t value[4] = {
            (uint16_t)(0x4000u + pos), (uint16_t)(0xc000u + pos),
            (uint16_t)(0x3c00u + pos), (uint16_t)(0xbc00u + pos),
        };
        if (!ds4_inkling_state_append_kv_fp16(slot, pos, key, value)) {
            ds4_inkling_state_session_free(&session);
            return 0;
        }
    }
    enum { QH = 2, KVH = 1, DIM = 4, EXTENT = 4 };
    const float query[QH * DIM] = { 0.25f, -0.5f, 1.0f, 0.125f,
                                    -1.0f, 0.25f, 0.5f, 0.75f };
    const float bias[QH * EXTENT] = { 0.5f, 0, -0.25f, 0.125f,
                                      -0.5f, 0.25f, 0, 0.375f };
    const float sink[QH] = { 0.2f, -0.3f };
    float expected[QH * DIM], actual[QH * DIM];
    int ok = ds4_inkling_state_attention(slot, query, QH, KVH, DIM, 5,
        bias, EXTENT, sink, expected);
    ds4_gpu_tensor *tq = upload(query, sizeof(query));
    ds4_gpu_tensor *tk = upload(slot->key,
        (uint64_t)slot->capacity * slot->kv_width * sizeof(uint16_t));
    ds4_gpu_tensor *tv = upload(slot->value,
        (uint64_t)slot->capacity * slot->kv_width * sizeof(uint16_t));
    ds4_gpu_tensor *tb = upload(bias, sizeof(bias));
    ds4_gpu_tensor *ts = upload(sink, sizeof(sink));
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc(sizeof(actual));
    ok = ok && tq && tk && tv && tb && ts && to &&
        ds4_gpu_inkling2_attention_decode_tensor(
            to, tq, tk, tv, tb, ts, QH, KVH, DIM,
            slot->capacity, slot->count, ds4_inkling_state_oldest_position(slot),
            5, 4, EXTENT) &&
        ds4_gpu_tensor_read(to, 0, actual, sizeof(actual));
    for (uint32_t i = 0; ok && i < QH * DIM; i++) {
        if (!near_record(actual[i], expected[i], 4e-5f,
                         &g_m4_attention_max_error)) {
            fprintf(stderr, "Inkling2 attention mismatch i=%u got=%g expected=%g\n",
                    i, actual[i], expected[i]);
            ok = 0;
        }
    }
    ds4_gpu_tensor_free(tq);
    ds4_gpu_tensor_free(tk);
    ds4_gpu_tensor_free(tv);
    ds4_gpu_tensor_free(tb);
    ds4_gpu_tensor_free(ts);
    ds4_gpu_tensor_free(to);

    enum { FULL_CAPACITY = 1030, FULL_COUNT = 1025 };
    uint16_t *global_keys = calloc((size_t)FULL_CAPACITY * 2, sizeof(uint16_t));
    uint16_t *global_values = malloc((size_t)FULL_CAPACITY * 2 * sizeof(uint16_t));
    if (!global_keys || !global_values) {
        free(global_keys);
        free(global_values);
        ds4_inkling_state_session_free(&session);
        return 0;
    }
    for (uint32_t i = 0; i < FULL_CAPACITY * 2u; i++) global_values[i] = 0x3c00u;
    const float global_query[2] = { 0, 0 };
    const float global_sink[1] = { 0 };
    float global_actual[2];
    ds4_gpu_tensor *tgk = upload(global_keys,
        (uint64_t)FULL_CAPACITY * 2u * sizeof(uint16_t));
    ds4_gpu_tensor *tgv = upload(global_values,
        (uint64_t)FULL_CAPACITY * 2u * sizeof(uint16_t));
    ds4_gpu_tensor *tgq = upload(global_query, sizeof(global_query));
    ds4_gpu_tensor *tgs = upload(global_sink, sizeof(global_sink));
    ds4_gpu_tensor *tgo = ds4_gpu_tensor_alloc(sizeof(global_actual));
    const float global_expected = (float)FULL_COUNT / (float)(FULL_COUNT + 1u);
    ok = ok && tgk && tgv && tgq && tgs && tgo &&
        ds4_gpu_inkling2_attention_decode_tensor(
            tgo, tgq, tgk, tgv, NULL, tgs, 1, 1, 2,
            FULL_CAPACITY, FULL_COUNT, 0, FULL_COUNT - 1u, 0, 0) &&
        ds4_gpu_tensor_read(tgo, 0, global_actual, sizeof(global_actual));
    for (uint32_t i = 0; ok && i < 2; i++) {
        if (!near_record(global_actual[i], global_expected, 2e-6f,
                         &g_m4_full_attention_max_error)) {
            fprintf(stderr,
                    "Inkling2 full attention truncated global cache: got=%g expected=%g\n",
                    global_actual[i], global_expected);
            ok = 0;
        }
    }
    ds4_gpu_tensor_free(tgk);
    ds4_gpu_tensor_free(tgv);
    ds4_gpu_tensor_free(tgq);
    ds4_gpu_tensor_free(tgs);
    ds4_gpu_tensor_free(tgo);
    free(global_keys);
    free(global_values);

    enum { TOKENS = 7, CHANNELS = 4, HEADS = 2, KERNEL = 4 };
    float input[TOKENS * CHANNELS];
    float static_taps[KERNEL * CHANNELS];
    float dynamic_taps[TOKENS * HEADS * KERNEL];
    float expected_conv[TOKENS * CHANNELS], cpu_conv[TOKENS * CHANNELS];
    float actual_conv[TOKENS * CHANNELS];
    for (uint32_t i = 0; i < TOKENS * CHANNELS; i++)
        input[i] = (float)((int)i - 9) / 7.0f;
    for (uint32_t i = 0; i < KERNEL * CHANNELS; i++)
        static_taps[i] = (float)((int)(i % 7) - 3) / 9.0f;
    for (uint32_t i = 0; i < TOKENS * HEADS * KERNEL; i++)
        dynamic_taps[i] = (float)((int)(i % 5) - 2) / 11.0f;
    ds4_inkling_state_slot *conv_slot = &session.slots[1];
    const uint32_t head_width = CHANNELS / HEADS;
    const float conv_scale = 1.0f / sqrtf((float)(2u * KERNEL + 1u));
    for (uint32_t token = 0; token < TOKENS; token++) {
        for (uint32_t channel = 0; channel < CHANNELS; channel++) {
            const uint32_t head = channel / head_width;
            float sum = 0.0f;
            for (uint32_t lag = 0; lag < KERNEL; lag++) {
                const float sample = token >= lag
                    ? input[(uint64_t)(token - lag) * CHANNELS + channel]
                    : 0.0f;
                const float static_tap = static_taps[
                    (uint64_t)channel * KERNEL + (KERNEL - 1u - lag)];
                const float dynamic_tap = dynamic_taps[
                    ((uint64_t)token * HEADS + head) * KERNEL + lag];
                sum += sample * (static_tap + dynamic_tap);
            }
            expected_conv[(uint64_t)token * CHANNELS + channel] =
                (input[(uint64_t)token * CHANNELS + channel] + sum) * conv_scale;
        }
    }
    ok = ok && ds4_inkling_state_shortconv(
        conv_slot, DS4_INKLING_CONV_KEY, input, static_taps, dynamic_taps,
        cpu_conv, TOKENS, CHANNELS, HEADS, KERNEL);
    for (uint32_t i = 0; ok && i < TOKENS * CHANNELS; i++) {
        if (!near_record(cpu_conv[i], expected_conv[i], 3e-5f,
                         &g_m4_conv_max_error)) {
            fprintf(stderr,
                    "Inkling2 reference shortconv mismatch i=%u got=%g expected=%g\n",
                    i, cpu_conv[i], expected_conv[i]);
            ok = 0;
        }
    }
    ds4_gpu_tensor *tx = upload(input, sizeof(input));
    ds4_gpu_tensor *tt = upload(static_taps, sizeof(static_taps));
    ds4_gpu_tensor *td = upload(dynamic_taps, sizeof(dynamic_taps));
    ds4_gpu_tensor *th = ds4_gpu_tensor_alloc(
        DS4_INKLING_CONV_HISTORY * CHANNELS * sizeof(float));
    ds4_gpu_tensor *tout = ds4_gpu_tensor_alloc(sizeof(actual_conv));
    ok = ok && tx && tt && td && th && tout &&
        ds4_gpu_tensor_fill_f32(th, 1.0f, DS4_INKLING_CONV_HISTORY * CHANNELS) &&
        ds4_gpu_inkling2_shortconv_tensor(
            tout, th, tx, tt, td, TOKENS, CHANNELS, HEADS, KERNEL, 0) &&
        ds4_gpu_tensor_read(tout, 0, actual_conv, sizeof(actual_conv));
    for (uint32_t i = 0; ok && i < TOKENS * CHANNELS; i++) {
        if (!near_record(actual_conv[i], expected_conv[i], 3e-5f,
                         &g_m4_conv_max_error)) {
            fprintf(stderr, "Inkling2 dynamic shortconv mismatch i=%u got=%g expected=%g\n",
                    i, actual_conv[i], expected_conv[i]);
            ok = 0;
        }
    }

    ds4_gpu_tensor *tchunk_history = ds4_gpu_tensor_alloc(
        DS4_INKLING_CONV_HISTORY * CHANNELS * sizeof(float));
    float chunked_conv[TOKENS * CHANNELS];
    const uint32_t split[] = { 2, 1, 4 };
    uint32_t offset = 0;
    ok = ok && tchunk_history && ds4_gpu_tensor_fill_f32(
        tchunk_history, 2.0f, DS4_INKLING_CONV_HISTORY * CHANNELS);
    for (uint32_t part = 0; ok && part < 3; part++) {
        const uint32_t count = split[part];
        ds4_gpu_tensor *tpart_x = upload(input + (uint64_t)offset * CHANNELS,
            (uint64_t)count * CHANNELS * sizeof(float));
        ds4_gpu_tensor *tpart_dynamic = upload(
            dynamic_taps + (uint64_t)offset * HEADS * KERNEL,
            (uint64_t)count * HEADS * KERNEL * sizeof(float));
        ds4_gpu_tensor *tpart_out = ds4_gpu_tensor_alloc(
            (uint64_t)count * CHANNELS * sizeof(float));
        const uint32_t previous_count = offset < 3u ? offset : 3u;
        ok = tpart_x && tpart_dynamic && tpart_out &&
            ds4_gpu_inkling2_shortconv_tensor(
                tpart_out, tchunk_history, tpart_x, tt, tpart_dynamic,
                count, CHANNELS, HEADS, KERNEL, previous_count) &&
            ds4_gpu_tensor_read(tpart_out, 0,
                chunked_conv + (uint64_t)offset * CHANNELS,
                (uint64_t)count * CHANNELS * sizeof(float));
        ds4_gpu_tensor_free(tpart_x);
        ds4_gpu_tensor_free(tpart_dynamic);
        ds4_gpu_tensor_free(tpart_out);
        offset += count;
    }
    for (uint32_t i = 0; ok && i < TOKENS * CHANNELS; i++) {
        if (!near_record(chunked_conv[i], actual_conv[i], 3e-5f,
                         &g_m4_conv_chunk_max_error)) {
            fprintf(stderr, "Inkling2 Metal conv chunk mismatch i=%u\n", i);
            ok = 0;
        }
    }

    float expected_static[TOKENS * CHANNELS], actual_static[TOKENS * CHANNELS];
    ok = ok && ds4_inkling_state_reset(conv_slot) &&
        ds4_inkling_state_shortconv(
            conv_slot, DS4_INKLING_CONV_KEY, input, static_taps, NULL,
            expected_static, TOKENS, CHANNELS, 1, KERNEL);
    ds4_gpu_tensor *tstatic_history = ds4_gpu_tensor_alloc(
        DS4_INKLING_CONV_HISTORY * CHANNELS * sizeof(float));
    ds4_gpu_tensor *tstatic_out = ds4_gpu_tensor_alloc(sizeof(actual_static));
    ok = ok && tstatic_history && tstatic_out &&
        ds4_gpu_tensor_fill_f32(tstatic_history, -3.0f,
                                DS4_INKLING_CONV_HISTORY * CHANNELS) &&
        ds4_gpu_inkling2_shortconv_tensor(
            tstatic_out, tstatic_history, tx, tt, NULL, TOKENS, CHANNELS, 1,
            KERNEL, 0) &&
        ds4_gpu_tensor_read(tstatic_out, 0, actual_static, sizeof(actual_static));
    for (uint32_t i = 0; ok && i < TOKENS * CHANNELS; i++) {
        if (!near_record(actual_static[i], expected_static[i], 3e-5f,
                         &g_m4_static_conv_max_error)) {
            fprintf(stderr, "Inkling2 static shortconv mismatch i=%u\n", i);
            ok = 0;
        }
    }
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(tt);
    ds4_gpu_tensor_free(td);
    ds4_gpu_tensor_free(th);
    ds4_gpu_tensor_free(tout);
    ds4_gpu_tensor_free(tchunk_history);
    ds4_gpu_tensor_free(tstatic_history);
    ds4_gpu_tensor_free(tstatic_out);
    ds4_inkling_state_session_free(&session);
    return ok;
}

int main(void) {
    int ok = 1;
    if (!ds4_gpu_init()) {
        fprintf(stderr, "Inkling2 Metal test could not initialize Metal\n");
        return 1;
    }
    ds4_inkling_plan operator_plan;
    make_small_plan(&operator_plan);
    if (!ds4_inkling_gpu_context_init(&g_operator_context, &operator_plan, 128u)) {
        fprintf(stderr, "Inkling2 reusable operator workspace setup failed\n");
        ds4_gpu_cleanup();
        return 1;
    }
    CHECK(test_rms_norm());
    CHECK(test_gate_update());
    CHECK(test_l2_clip());
    CHECK(test_logit_softcap());
    CHECK(test_router_topk());
    CHECK(test_quantized_matvec());
    CHECK(test_expert_accumulation());
    CHECK(test_situ_moe_metal());
    CHECK(test_expert_grouped_iq4());
    CHECK(test_expert_route_map_512());
    CHECK(test_expert_grouped_iq4_full_shape());
    CHECK(test_qk_and_relative_bias());
    CHECK(test_attention_output_gate());
    CHECK(test_attention_prefill_matches_decode());
    CHECK(test_attention_and_shortconv());
    CHECK(test_loop_mix_and_control_metal());
    CHECK(test_gpu_context_owns_independent_visit_state());
    fprintf(stderr,
            "Inkling2 M4 max absolute error: QK/relative %.8g, attention %.8g, "
            "full-1025 %.8g, gate %.8g, dynamic-conv %.8g, chunk-delta %.8g, "
            "static-conv %.8g\n",
            g_m4_qk_relative_max_error, g_m4_attention_max_error,
            g_m4_full_attention_max_error, g_m4_gate_max_error,
            g_m4_conv_max_error, g_m4_conv_chunk_max_error,
            g_m4_static_conv_max_error);
    fprintf(stderr, "Inkling2 Metal operators: ok\n");
cleanup:
    ds4_inkling_gpu_context_free(&g_operator_context);
    ds4_gpu_cleanup();
    free(g_synthetic_model_map);
    free(g_synthetic_moe_map);
    free(g_full_shape_iq4_map);
    free(g_synthetic_loop_map);
    return ok ? 0 : 1;
}
