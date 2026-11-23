#ifndef DS4_INKLING_MOE_H
#define DS4_INKLING_MOE_H

#include "ds4_gpu.h"

enum {
    DS4_INKLING_GGML_TYPE_Q8_0 = 8u,
    DS4_INKLING_GGML_TYPE_IQ4_NL = 20u,
};

/* Execute one token's routed Inkling SITU FFN entirely on Metal. Router IDs
 * and weights stay on the device. The active DS4 Metal model views must cover
 * the supplied model map's router and expert ranges. Expert weights use
 * [expert][output row][quantized input block] order. */
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
        float                 situ_linear_beta);

#endif /* DS4_INKLING_MOE_H */
