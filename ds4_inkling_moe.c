#include "ds4_inkling_moe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int ds4_inkling_moe_situ_metal(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *correction_bias,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              router_offset,
        uint64_t              gate_up_offset,
        uint64_t              down_offset,
        uint32_t              hidden_dim,
        uint32_t              expert_dim,
        uint32_t              n_experts,
        uint32_t              n_used,
        uint32_t              expert_weight_type,
        float                 routed_scale,
        float                 situ_beta,
        float                 situ_linear_beta) {
    if (!out || !input || !model_map || hidden_dim == 0 || expert_dim == 0 ||
        n_experts == 0 || n_experts > 128u || n_used == 0 || n_used > 8u ||
        n_used > n_experts || (hidden_dim % 32u) != 0 ||
        expert_dim > UINT32_MAX / 2u || (expert_dim % 32u) != 0 ||
        (expert_weight_type != DS4_INKLING_GGML_TYPE_Q8_0 &&
         expert_weight_type != DS4_INKLING_GGML_TYPE_IQ4_NL) ||
        !isfinite(routed_scale) || !isfinite(situ_beta) || situ_beta <= 0.0f ||
        !isfinite(situ_linear_beta) || situ_linear_beta <= 0.0f) return 0;

    const uint64_t expert_rows = (uint64_t)n_used * expert_dim;
    if (expert_rows > UINT64_MAX / (2u * sizeof(float))) return 0;

    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc((uint64_t)n_experts * sizeof(float));
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc((uint64_t)n_used * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)n_used * sizeof(float));
    ds4_gpu_tensor *gate_up = ds4_gpu_tensor_alloc(expert_rows * 2u * sizeof(float));
    ds4_gpu_tensor *activation = ds4_gpu_tensor_alloc(expert_rows * sizeof(float));
    ds4_gpu_tensor *expert_outputs = ds4_gpu_tensor_alloc(
        (uint64_t)n_used * hidden_dim * sizeof(float));
    ds4_gpu_tensor *zero_bias = NULL;
    if (!correction_bias) {
        zero_bias = ds4_gpu_tensor_alloc((uint64_t)n_experts * sizeof(float));
        if (zero_bias && !ds4_gpu_tensor_fill_f32(zero_bias, 0.0f, n_experts)) {
            ds4_gpu_tensor_free(zero_bias);
            zero_bias = NULL;
        }
    }

    const ds4_gpu_tensor *bias = correction_bias ? correction_bias : zero_bias;
    int ok = logits && ids && weights && gate_up && activation && expert_outputs && bias;
    const char *failed_stage = ok ? NULL : "scratch allocation";
    if (ok && !ds4_gpu_matmul_f32_tensor(logits, model_map, model_size,
                                          router_offset, hidden_dim, n_experts,
                                          input, 1u)) {
        ok = 0;
        failed_stage = "router projection";
    }
    if (ok && !ds4_gpu_inkling2_router_topk_tensor(ids, weights, logits, bias,
                                                   1u, n_experts, n_used,
                                                   routed_scale)) {
        ok = 0;
        failed_stage = "router selection";
    }
    if (ok && !ds4_gpu_inkling2_expert_gemv_tensor(
                  gate_up, model_map, model_size, gate_up_offset,
                  expert_weight_type, hidden_dim, 2u * expert_dim,
                  n_experts, n_used, 0, ids, input)) {
        ok = 0;
        failed_stage = "gate/up projection";
    }
    if (ok && !ds4_gpu_inkling2_situ_tensor(activation, gate_up, n_used,
                                             expert_dim, situ_beta,
                                             situ_linear_beta)) {
        ok = 0;
        failed_stage = "SiTU activation";
    }
    if (ok && !ds4_gpu_inkling2_expert_gemv_tensor(
                  expert_outputs, model_map, model_size, down_offset,
                  expert_weight_type, expert_dim, hidden_dim,
                  n_experts, n_used, 1, ids, activation)) {
        ok = 0;
        failed_stage = "down projection";
    }
    if (ok && !ds4_gpu_inkling2_expert_accumulate_tensor(
                  out, expert_outputs, weights, hidden_dim, n_used)) {
        ok = 0;
        failed_stage = "expert accumulation";
    }
    if (!ok) fprintf(stderr, "ds4: Inkling2 SITU MoE failed at %s\n", failed_stage);

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(ids);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(gate_up);
    ds4_gpu_tensor_free(activation);
    ds4_gpu_tensor_free(expert_outputs);
    ds4_gpu_tensor_free(zero_bias);
    return ok;
}
