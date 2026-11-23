#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    IN_DIM = 2304,
    OUT_DIM = 1152,
    IQ4_BLOCK_BYTES = 18,
    Q8_BLOCK_BYTES = 34,
    WARMUP = 3,
    RUNS = 3,
    REPETITIONS = 10,
};

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static double monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void fill_weights(uint8_t *map, uint64_t iq4_offset, uint64_t q8_offset) {
    const uint32_t blocks = IN_DIM / 32u;
    const uint32_t iq4_row_bytes = blocks * IQ4_BLOCK_BYTES;
    const uint32_t q8_row_bytes = blocks * Q8_BLOCK_BYTES;
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        for (uint32_t block = 0; block < blocks; block++) {
            uint8_t *iq4 = map + iq4_offset +
                (uint64_t)row * iq4_row_bytes + (uint64_t)block * IQ4_BLOCK_BYTES;
            const uint16_t iq4_scale = 0x3c00u;
            memcpy(iq4, &iq4_scale, sizeof(iq4_scale));
            for (uint32_t i = 0; i < 16; i++) {
                const uint8_t lo = (uint8_t)((row + block + i * 3u) & 15u);
                const uint8_t hi = (uint8_t)((row * 3u + block + i * 5u + 7u) & 15u);
                iq4[2u + i] = (uint8_t)(lo | (hi << 4u));
            }

            uint8_t *q8 = map + q8_offset +
                (uint64_t)row * q8_row_bytes + (uint64_t)block * Q8_BLOCK_BYTES;
            const uint16_t q8_scale = 0x3c00u;
            memcpy(q8, &q8_scale, sizeof(q8_scale));
            for (uint32_t i = 0; i < 32; i++) {
                q8[2u + i] = (uint8_t)(int8_t)
                    ((int32_t)((row * 7u + block * 11u + i * 13u) % 255u) - 127);
            }
        }
    }
}

typedef int (*bench_fn)(ds4_gpu_tensor *, const void *, uint64_t, uint64_t,
                        uint64_t, uint64_t, const ds4_gpu_tensor *);

static int bench_iq4(ds4_gpu_tensor *out, const void *map, uint64_t map_size,
                     uint64_t offset, uint64_t in_dim, uint64_t out_dim,
                     const ds4_gpu_tensor *x) {
    return ds4_gpu_inkling2_gemv_iq4_nl_tensor(
        out, map, map_size, offset, in_dim, out_dim, x);
}

static int bench_q8(ds4_gpu_tensor *out, const void *map, uint64_t map_size,
                    uint64_t offset, uint64_t in_dim, uint64_t out_dim,
                    const ds4_gpu_tensor *x) {
    return ds4_gpu_matmul_q8_0_tensor(
        out, map, map_size, offset, in_dim, out_dim, x, 1u);
}

static int run_benchmark(const char *name, bench_fn fn, ds4_gpu_tensor *out,
                         const void *map, uint64_t map_size, uint64_t offset,
                         const ds4_gpu_tensor *x) {
    for (uint32_t i = 0; i < WARMUP; i++) {
        if (!fn(out, map, map_size, offset, IN_DIM, OUT_DIM, x)) return 0;
    }
    for (uint32_t run = 0; run < RUNS; run++) {
        const double start = monotonic_ms();
        for (uint32_t i = 0; i < REPETITIONS; i++) {
            if (!fn(out, map, map_size, offset, IN_DIM, OUT_DIM, x)) return 0;
        }
        const double elapsed = monotonic_ms() - start;
        fprintf(stderr, "Inkling2 synthetic %s run %u: %.3f us/op\n",
                name, run + 1u, elapsed * 1000.0 / REPETITIONS);
    }
    return 1;
}

int main(void) {
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) return 1;
    const uint64_t iq4_bytes = (uint64_t)OUT_DIM * (IN_DIM / 32u) * IQ4_BLOCK_BYTES;
    const uint64_t q8_bytes = (uint64_t)OUT_DIM * (IN_DIM / 32u) * Q8_BLOCK_BYTES;
    const uint64_t iq4_offset = 256u;
    const uint64_t q8_offset = align_up(iq4_offset + iq4_bytes, 256u);
    const uint64_t model_size = q8_offset + q8_bytes;
    const uint64_t allocation = align_up(model_size, (uint64_t)page);
    void *model_map = NULL;
    if (allocation > SIZE_MAX || posix_memalign(&model_map, (size_t)page,
                                                 (size_t)allocation) != 0) {
        fprintf(stderr, "Inkling2 benchmark model allocation failed\n");
        return 1;
    }
    memset(model_map, 0, (size_t)allocation);
    fill_weights(model_map, iq4_offset, q8_offset);

    float *x_host = malloc(IN_DIM * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(IN_DIM * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(OUT_DIM * sizeof(float));
    if (!x_host || !x || !out) {
        fprintf(stderr, "Inkling2 benchmark activation allocation failed\n");
        free(x_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(model_map);
        return 1;
    }
    for (uint32_t i = 0; i < IN_DIM; i++) {
        x_host[i] = (float)((int32_t)((i * 17u) % 127u) - 63) / 64.0f;
    }

    int ok = ds4_gpu_init() &&
        ds4_gpu_set_model_map(model_map, model_size) &&
        ds4_gpu_tensor_write(x, 0, x_host, IN_DIM * sizeof(float));
    if (ok) {
        fprintf(stderr,
                "Inkling2 synthetic Metal microbenchmark: %u x %u, %u warmups, "
                "%u runs x %u synchronous operations; no model weights\n",
                OUT_DIM, IN_DIM, WARMUP, RUNS, REPETITIONS);
        ok = run_benchmark("IQ4_NL", bench_iq4, out, model_map, model_size,
                           iq4_offset, x) &&
             run_benchmark("Q8_0", bench_q8, out, model_map, model_size,
                           q8_offset, x);
    }
    if (!ok) fprintf(stderr, "Inkling2 Metal microbenchmark failed\n");
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(x_host);
    ds4_gpu_cleanup();
    free(model_map);
    return ok ? 0 : 1;
}
