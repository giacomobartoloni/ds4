#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum {
    QUERY_HEADS = 24,
    KV_HEADS = 8,
    HEAD_DIM = 128,
    SLIDING_WINDOW = 1024,
    CONTEXT = 8192,
    WARMUP = 2,
    RUNS = 3,
    REPETITIONS = 3,
};

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static double monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int run_case(const char *name, ds4_gpu_tensor *out,
                    const ds4_gpu_tensor *query,
                    const ds4_gpu_tensor *keys,
                    const ds4_gpu_tensor *values,
                    const ds4_gpu_tensor *bias,
                    const ds4_gpu_tensor *sinks,
                    uint32_t capacity, uint32_t count,
                    uint64_t oldest_position, uint64_t query_position,
                    uint32_t sliding_window) {
    for (uint32_t i = 0; i < WARMUP; i++) {
        if (!ds4_gpu_inkling2_attention_decode_tensor(
                out, query, keys, values, bias, sinks,
                QUERY_HEADS, KV_HEADS, HEAD_DIM, capacity, count,
                oldest_position, query_position, sliding_window,
                SLIDING_WINDOW)) return 0;
    }
    for (uint32_t run = 0; run < RUNS; run++) {
        const double start = monotonic_ms();
        for (uint32_t i = 0; i < REPETITIONS; i++) {
            if (!ds4_gpu_inkling2_attention_decode_tensor(
                    out, query, keys, values, bias, sinks,
                    QUERY_HEADS, KV_HEADS, HEAD_DIM, capacity, count,
                    oldest_position, query_position, sliding_window,
                    SLIDING_WINDOW)) return 0;
        }
        const double elapsed = monotonic_ms() - start;
        fprintf(stderr, "Inkling2 synthetic %s run %u: %.3f ms/op\n",
                name, run + 1u, elapsed / REPETITIONS);
    }
    return 1;
}

int main(void) {
    const uint64_t kv_width = (uint64_t)KV_HEADS * HEAD_DIM;
    const uint64_t kv_elements = (uint64_t)CONTEXT * kv_width;
    uint16_t *keys_host = calloc((size_t)kv_elements, sizeof(uint16_t));
    uint16_t *values_host = malloc((size_t)kv_elements * sizeof(uint16_t));
    float *query_host = calloc((size_t)QUERY_HEADS * HEAD_DIM, sizeof(float));
    float *bias_host = calloc((size_t)QUERY_HEADS * SLIDING_WINDOW, sizeof(float));
    float *sink_host = calloc(QUERY_HEADS, sizeof(float));
    if (!keys_host || !values_host || !query_host || !bias_host || !sink_host) {
        fprintf(stderr, "Inkling2 M4 benchmark host allocation failed\n");
        free(keys_host);
        free(values_host);
        free(query_host);
        free(bias_host);
        free(sink_host);
        return 1;
    }
    for (uint64_t i = 0; i < kv_elements; i++) values_host[i] = 0x3c00u;

    ds4_gpu_tensor *query = ds4_gpu_tensor_alloc(
        (uint64_t)QUERY_HEADS * HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *keys = ds4_gpu_tensor_alloc(kv_elements * sizeof(uint16_t));
    ds4_gpu_tensor *values = ds4_gpu_tensor_alloc(kv_elements * sizeof(uint16_t));
    ds4_gpu_tensor *bias = ds4_gpu_tensor_alloc(
        (uint64_t)QUERY_HEADS * SLIDING_WINDOW * sizeof(float));
    ds4_gpu_tensor *sinks = ds4_gpu_tensor_alloc(QUERY_HEADS * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
        (uint64_t)QUERY_HEADS * HEAD_DIM * sizeof(float));
    int ok = query && keys && values && bias && sinks && out && ds4_gpu_init() &&
        ds4_gpu_tensor_write(query, 0, query_host,
            (uint64_t)QUERY_HEADS * HEAD_DIM * sizeof(float)) &&
        ds4_gpu_tensor_write(keys, 0, keys_host, kv_elements * sizeof(uint16_t)) &&
        ds4_gpu_tensor_write(values, 0, values_host, kv_elements * sizeof(uint16_t)) &&
        ds4_gpu_tensor_write(bias, 0, bias_host,
            (uint64_t)QUERY_HEADS * SLIDING_WINDOW * sizeof(float)) &&
        ds4_gpu_tensor_write(sinks, 0, sink_host, QUERY_HEADS * sizeof(float));
    if (ok) {
        fprintf(stderr,
                "Inkling2 synthetic M4 attention benchmark: %u query heads, "
                "%u KV heads x %u dim, %u warmups, %u runs x %u synchronous ops; "
                "no model weights\n",
                QUERY_HEADS, KV_HEADS, HEAD_DIM, WARMUP, RUNS, REPETITIONS);
        ok = run_case("SWA-1024", out, query, keys, values, bias, sinks,
                      SLIDING_WINDOW, SLIDING_WINDOW,
                      CONTEXT - SLIDING_WINDOW, CONTEXT - 1u,
                      SLIDING_WINDOW) &&
             run_case("full-8192", out, query, keys, values, bias, sinks,
                      CONTEXT, CONTEXT, 0, CONTEXT - 1u, 0);
    }
    if (!ok) fprintf(stderr, "Inkling2 M4 attention benchmark failed\n");
    ds4_gpu_tensor_free(query);
    ds4_gpu_tensor_free(keys);
    ds4_gpu_tensor_free(values);
    ds4_gpu_tensor_free(bias);
    ds4_gpu_tensor_free(sinks);
    ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    free(keys_host);
    free(values_host);
    free(query_host);
    free(bias_host);
    free(sink_host);
    return ok ? 0 : 1;
}
