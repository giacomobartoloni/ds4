#define _DARWIN_C_SOURCE

#include "ds4_inkling_moe.h"
#include "ds4_inkling_gpu_state.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    HIDDEN_DIM = 2304,
    EXPERT_DIM = 1152,
    N_EXPERTS = 128,
    N_USED = 8,
    IQ4_BLOCK_BYTES = 18,
    WARMUP = 2,
    RUNS = 3,
    REPETITIONS = 3,
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

static void fill_experts(uint8_t *dst, uint32_t rows, uint32_t in_dim,
                         uint32_t seed) {
    const uint32_t blocks_per_row = in_dim / 32u;
    const uint32_t row_bytes = blocks_per_row * IQ4_BLOCK_BYTES;
    const uint64_t expert_bytes = (uint64_t)rows * row_bytes;
    static const uint16_t scales[] = { 0x3400u, 0x3800u, 0x3a00u };
    for (uint32_t expert = 0; expert < N_EXPERTS; expert++) {
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t block = 0; block < blocks_per_row; block++) {
                uint8_t *packed = dst + (uint64_t)expert * expert_bytes +
                    (uint64_t)row * row_bytes + (uint64_t)block * IQ4_BLOCK_BYTES;
                const uint16_t scale = scales[(expert + row + seed) % 3u];
                memcpy(packed, &scale, sizeof(scale));
                for (uint32_t i = 0; i < 16; i++) {
                    const uint8_t lo = (uint8_t)((expert + row * 3u +
                                                  block + i * 7u + seed) & 15u);
                    const uint8_t hi = (uint8_t)((expert * 5u + row +
                                                  block * 3u + i * 11u + seed) & 15u);
                    packed[2u + i] = (uint8_t)(lo | (hi << 4u));
                }
            }
        }
    }
}

static int run_batch(ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
                     const ds4_gpu_tensor *bias, const void *model_map,
                     uint64_t model_size, uint64_t router_offset,
                     uint64_t gate_up_offset, uint64_t down_offset,
                     const ds4_inkling_gpu_moe_workspace *workspace,
                     uint32_t repetitions) {
    if (!ds4_gpu_begin_commands()) return 0;
    int ok = 1;
    for (uint32_t i = 0; i < repetitions; i++) {
        if (!ds4_inkling_moe_situ_metal_with_workspace(
                out, input, bias, model_map, model_size,
                router_offset, gate_up_offset, down_offset,
                HIDDEN_DIM, EXPERT_DIM, N_EXPERTS, N_USED, 20u,
                8.0f, 1.0f, 1.0f, workspace)) {
            ok = 0;
            break;
        }
    }
    const int commands_ok = ds4_gpu_end_commands();
    return ok && commands_ok;
}

static int run_benchmark(ds4_gpu_tensor *out, const ds4_gpu_tensor *input,
                         const ds4_gpu_tensor *bias, const void *model_map,
                         uint64_t model_size, uint64_t router_offset,
                         uint64_t gate_up_offset, uint64_t down_offset,
                         const ds4_inkling_gpu_moe_workspace *workspace) {
    for (uint32_t i = 0; i < WARMUP; i++) {
        if (!run_batch(out, input, bias, model_map, model_size, router_offset,
                       gate_up_offset, down_offset, workspace, 1u)) return 0;
    }
    for (uint32_t run = 0; run < RUNS; run++) {
        const double start = monotonic_ms();
        if (!run_batch(out, input, bias, model_map, model_size, router_offset,
                       gate_up_offset, down_offset, workspace, REPETITIONS)) return 0;
        const double elapsed = monotonic_ms() - start;
        fprintf(stderr, "Inkling2 synthetic IQ4_NL routed FFN run %u: %.3f ms/op\n",
                run + 1u, elapsed / REPETITIONS);
    }
    return 1;
}

int main(void) {
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) return 1;
    const uint64_t router_offset = 256u;
    const uint64_t router_bytes =
        (uint64_t)HIDDEN_DIM * N_EXPERTS * sizeof(float);
    const uint32_t gate_up_row_bytes = HIDDEN_DIM / 32u * IQ4_BLOCK_BYTES;
    const uint64_t gate_up_bytes =
        (uint64_t)N_EXPERTS * 2u * EXPERT_DIM * gate_up_row_bytes;
    const uint32_t down_row_bytes = EXPERT_DIM / 32u * IQ4_BLOCK_BYTES;
    const uint64_t down_bytes =
        (uint64_t)N_EXPERTS * HIDDEN_DIM * down_row_bytes;
    const uint64_t gate_up_offset = align_up(router_offset + router_bytes, 256u);
    const uint64_t down_offset = align_up(gate_up_offset + gate_up_bytes, 256u);
    const uint64_t model_size = down_offset + down_bytes;
    const uint64_t allocation = align_up(model_size, (uint64_t)page);
    void *model_map = NULL;
    if (allocation > SIZE_MAX || posix_memalign(&model_map, (size_t)page,
                                                (size_t)allocation) != 0) {
        fprintf(stderr, "Inkling2 MoE benchmark model allocation failed\n");
        return 1;
    }
    memset(model_map, 0, (size_t)allocation);

    float *router = (float *)(void *)((uint8_t *)model_map + router_offset);
    for (uint32_t expert = 0; expert < N_EXPERTS; expert++) {
        router[(uint64_t)expert * HIDDEN_DIM] =
            ((int32_t)expert - 64) * 0.03f;
    }
    fill_experts((uint8_t *)model_map + gate_up_offset,
                 2u * EXPERT_DIM, HIDDEN_DIM, 1u);
    fill_experts((uint8_t *)model_map + down_offset,
                 HIDDEN_DIM, EXPERT_DIM, 4u);

    float *input_host = malloc((size_t)HIDDEN_DIM * sizeof(float));
    float *bias_host = calloc(N_EXPERTS, sizeof(float));
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(
        (uint64_t)HIDDEN_DIM * sizeof(float));
    ds4_gpu_tensor *bias = ds4_gpu_tensor_alloc(
        (uint64_t)N_EXPERTS * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(
        (uint64_t)HIDDEN_DIM * sizeof(float));
    ds4_inkling_gpu_moe_workspace workspace = {
        .logits = ds4_gpu_tensor_alloc((uint64_t)N_EXPERTS * sizeof(float)),
        .ids = ds4_gpu_tensor_alloc((uint64_t)N_USED * sizeof(int32_t)),
        .weights = ds4_gpu_tensor_alloc((uint64_t)N_USED * sizeof(float)),
        .gate_up = ds4_gpu_tensor_alloc(
            (uint64_t)N_USED * 2u * EXPERT_DIM * sizeof(float)),
        .activation = ds4_gpu_tensor_alloc(
            (uint64_t)N_USED * EXPERT_DIM * sizeof(float)),
        .expert_outputs = ds4_gpu_tensor_alloc(
            (uint64_t)N_USED * HIDDEN_DIM * sizeof(float)),
        .zero_bias = ds4_gpu_tensor_alloc((uint64_t)N_EXPERTS * sizeof(float)),
    };
    if (!input_host || !bias_host || !input || !bias || !out ||
        !workspace.logits || !workspace.ids || !workspace.weights ||
        !workspace.gate_up || !workspace.activation ||
        !workspace.expert_outputs || !workspace.zero_bias) {
        fprintf(stderr, "Inkling2 MoE benchmark activation allocation failed\n");
        free(input_host);
        free(bias_host);
        ds4_gpu_tensor_free(input);
        ds4_gpu_tensor_free(bias);
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(workspace.logits);
        ds4_gpu_tensor_free(workspace.ids);
        ds4_gpu_tensor_free(workspace.weights);
        ds4_gpu_tensor_free(workspace.gate_up);
        ds4_gpu_tensor_free(workspace.activation);
        ds4_gpu_tensor_free(workspace.expert_outputs);
        ds4_gpu_tensor_free(workspace.zero_bias);
        free(model_map);
        return 1;
    }
    for (uint32_t i = 0; i < HIDDEN_DIM; i++) {
        input_host[i] = (float)((int32_t)((i * 17u) % 127u) - 63) / 128.0f;
    }

    int ok = ds4_gpu_init() &&
        ds4_gpu_set_model_map(model_map, model_size) &&
        ds4_gpu_tensor_write(input, 0, input_host,
                             (uint64_t)HIDDEN_DIM * sizeof(float)) &&
        ds4_gpu_tensor_write(bias, 0, bias_host,
                             (uint64_t)N_EXPERTS * sizeof(float));
    if (ok) {
        fprintf(stderr,
                "Inkling2 synthetic M5 routed FFN benchmark: %u hidden, %u expert, "
                "%u/%u experts, %.1f MiB IQ4_NL weights, %u warmups, "
                "%u runs x %u calls per command buffer; one sync per run; "
                "no model weights\n",
                HIDDEN_DIM, EXPERT_DIM, N_USED, N_EXPERTS,
                (double)(gate_up_bytes + down_bytes) / (1024.0 * 1024.0),
                WARMUP, RUNS, REPETITIONS);
        ok = run_benchmark(out, input, bias, model_map, model_size,
                           router_offset, gate_up_offset, down_offset,
                           &workspace);
    }
    if (!ok) fprintf(stderr, "Inkling2 Metal routed FFN benchmark failed\n");
    ds4_gpu_tensor_free(input);
    ds4_gpu_tensor_free(bias);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(workspace.logits);
    ds4_gpu_tensor_free(workspace.ids);
    ds4_gpu_tensor_free(workspace.weights);
    ds4_gpu_tensor_free(workspace.gate_up);
    ds4_gpu_tensor_free(workspace.activation);
    ds4_gpu_tensor_free(workspace.expert_outputs);
    ds4_gpu_tensor_free(workspace.zero_bias);
    ds4_gpu_cleanup();
    free(input_host);
    free(bias_host);
    free(model_map);
    return ok ? 0 : 1;
}
