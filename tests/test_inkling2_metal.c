#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

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
    fprintf(stderr, "Inkling2 Metal operators: ok\n");
cleanup:
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
