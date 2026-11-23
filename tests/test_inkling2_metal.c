#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void *g_synthetic_model_map;
static uint64_t g_synthetic_model_size;

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

int main(void) {
    int ok = 1;
    if (!ds4_gpu_init()) {
        fprintf(stderr, "Inkling2 Metal test could not initialize Metal\n");
        return 1;
    }
    CHECK(test_rms_norm());
    CHECK(test_gate_update());
    CHECK(test_l2_clip());
    CHECK(test_logit_softcap());
    CHECK(test_router_topk());
    CHECK(test_quantized_matvec());
    CHECK(test_expert_accumulation());
    fprintf(stderr, "Inkling2 Metal operators: ok\n");
cleanup:
    ds4_gpu_cleanup();
    free(g_synthetic_model_map);
    return ok ? 0 : 1;
}
