#include "ds4_inkling_moe.h"
#include "ds4_inkling_gpu_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int ds4_inkling_moe_situ_metal_with_workspace(
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
        float                 situ_linear_beta,
        const ds4_inkling_gpu_moe_workspace *workspace) {
    if (!out || !input || !model_map || hidden_dim == 0 || expert_dim == 0 ||
        n_experts == 0 || n_experts > 128u || n_used == 0 || n_used > 8u ||
        n_used > n_experts || (hidden_dim % 32u) != 0 ||
        expert_dim > UINT32_MAX / 2u || (expert_dim % 32u) != 0 ||
        (expert_weight_type != DS4_INKLING_GGML_TYPE_Q8_0 &&
         expert_weight_type != DS4_INKLING_GGML_TYPE_IQ4_NL) ||
        !isfinite(routed_scale) || !isfinite(situ_beta) || situ_beta <= 0.0f ||
        !isfinite(situ_linear_beta) || situ_linear_beta <= 0.0f) return 0;

    const uint64_t expert_rows = (uint64_t)n_used * expert_dim;
    const uint64_t expert_output_elements = (uint64_t)n_used * hidden_dim;
    if (expert_rows > UINT64_MAX / (2u * sizeof(float)) ||
        expert_output_elements > UINT64_MAX / sizeof(float) || !workspace ||
        !workspace->logits || !workspace->ids || !workspace->weights ||
        !workspace->gate_up || !workspace->activation ||
        !workspace->expert_outputs || !workspace->zero_bias) return 0;

    const uint64_t logits_bytes = (uint64_t)n_experts * sizeof(float);
    const uint64_t ids_bytes = (uint64_t)n_used * sizeof(int32_t);
    const uint64_t route_bytes = (uint64_t)n_used * sizeof(float);
    const uint64_t gate_up_bytes = expert_rows * 2u * sizeof(float);
    const uint64_t activation_bytes = expert_rows * sizeof(float);
    const uint64_t expert_bytes = expert_output_elements * sizeof(float);
    if (ds4_gpu_tensor_bytes(workspace->logits) < logits_bytes ||
        ds4_gpu_tensor_bytes(workspace->ids) < ids_bytes ||
        ds4_gpu_tensor_bytes(workspace->weights) < route_bytes ||
        ds4_gpu_tensor_bytes(workspace->gate_up) < gate_up_bytes ||
        ds4_gpu_tensor_bytes(workspace->activation) < activation_bytes ||
        ds4_gpu_tensor_bytes(workspace->expert_outputs) < expert_bytes ||
        (!correction_bias && ds4_gpu_tensor_bytes(workspace->zero_bias) < logits_bytes))
        return 0;

    if (!correction_bias &&
        !ds4_gpu_tensor_fill_f32(workspace->zero_bias, 0.0f, n_experts)) return 0;
    const ds4_gpu_tensor *bias = correction_bias ? correction_bias : workspace->zero_bias;
    int ok = 1;
    const char *failed_stage = ok ? NULL : "scratch allocation";
    if (ok && !ds4_gpu_matmul_f32_tensor(workspace->logits, model_map,
                                          model_size, router_offset,
                                          hidden_dim, n_experts, input, 1u)) {
        ok = 0;
        failed_stage = "router projection";
    }
    if (ok && !ds4_gpu_inkling2_router_topk_tensor(workspace->ids,
                                                   workspace->weights,
                                                   workspace->logits, bias,
                                                   1u, n_experts, n_used,
                                                   routed_scale)) {
        ok = 0;
        failed_stage = "router selection";
    }
    if (ok && !ds4_gpu_inkling2_expert_gemv_tensor(
                  workspace->gate_up, model_map, model_size, gate_up_offset,
                  expert_weight_type, hidden_dim, 2u * expert_dim,
                  n_experts, n_used, 0, workspace->ids, input)) {
        ok = 0;
        failed_stage = "gate/up projection";
    }
    if (ok && !ds4_gpu_inkling2_situ_tensor(workspace->activation,
                                             workspace->gate_up, n_used,
                                             expert_dim, situ_beta,
                                             situ_linear_beta)) {
        ok = 0;
        failed_stage = "SiTU activation";
    }
    if (ok && !ds4_gpu_inkling2_expert_gemv_tensor(
                  workspace->expert_outputs, model_map, model_size, down_offset,
                  expert_weight_type, expert_dim, hidden_dim,
                  n_experts, n_used, 1, workspace->ids, workspace->activation)) {
        ok = 0;
        failed_stage = "down projection";
    }
    if (ok && !ds4_gpu_inkling2_expert_accumulate_tensor(
                  out, workspace->expert_outputs, workspace->weights,
                  hidden_dim, n_used)) {
        ok = 0;
        failed_stage = "expert accumulation";
    }
    if (!ok) fprintf(stderr, "ds4: Inkling2 SITU MoE failed at %s\n", failed_stage);

    return ok;
}

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
    if (n_experts == 0 || n_experts > 128u || n_used == 0 || n_used > 8u ||
        n_used > n_experts || expert_dim == 0 ||
        (uint64_t)n_used * expert_dim > UINT64_MAX / (2u * sizeof(float)))
        return 0;
    const uint64_t expert_rows = (uint64_t)n_used * expert_dim;
    ds4_inkling_gpu_moe_workspace workspace = {0};
    workspace.logits = ds4_gpu_tensor_alloc((uint64_t)n_experts * sizeof(float));
    workspace.ids = ds4_gpu_tensor_alloc((uint64_t)n_used * sizeof(int32_t));
    workspace.weights = ds4_gpu_tensor_alloc((uint64_t)n_used * sizeof(float));
    workspace.gate_up = ds4_gpu_tensor_alloc(expert_rows * 2u * sizeof(float));
    workspace.activation = ds4_gpu_tensor_alloc(expert_rows * sizeof(float));
    workspace.expert_outputs = ds4_gpu_tensor_alloc(
        (uint64_t)n_used * hidden_dim * sizeof(float));
    workspace.zero_bias = ds4_gpu_tensor_alloc((uint64_t)n_experts * sizeof(float));
    const int allocated = workspace.logits && workspace.ids && workspace.weights &&
        workspace.gate_up && workspace.activation && workspace.expert_outputs &&
        workspace.zero_bias;
    const int ok = allocated && ds4_inkling_moe_situ_metal_with_workspace(
        out, input, correction_bias, model_map, model_size, router_offset,
        gate_up_offset, down_offset, hidden_dim, expert_dim, n_experts, n_used,
        expert_weight_type, routed_scale, situ_beta, situ_linear_beta,
        &workspace);
    ds4_gpu_tensor_free(workspace.logits);
    ds4_gpu_tensor_free(workspace.ids);
    ds4_gpu_tensor_free(workspace.weights);
    ds4_gpu_tensor_free(workspace.gate_up);
    ds4_gpu_tensor_free(workspace.activation);
    ds4_gpu_tensor_free(workspace.expert_outputs);
    ds4_gpu_tensor_free(workspace.zero_bias);
    return ok;
}
