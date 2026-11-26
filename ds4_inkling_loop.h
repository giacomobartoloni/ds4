#ifndef DS4_INKLING_LOOP_H
#define DS4_INKLING_LOOP_H

#include "ds4_gpu.h"

#include <stdint.h>

struct ds4_inkling_gpu_control_workspace;

int ds4_inkling_mix_ref(
        float       *out,
        const float *states,
        const float *queries,
        uint32_t     state_count,
        uint32_t     hidden_dim,
        uint32_t     query_count,
        float        epsilon);

int ds4_inkling_control_ref(
        float       *out,
        const float *current,
        const float *message,
        const float *ctrl_down,
        const float *ctrl_gain,
        const float *ctrl_forget,
        uint32_t     hidden_dim,
        uint32_t     rank,
        float        epsilon);

int ds4_inkling_mix_metal(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *states,
        const ds4_gpu_tensor *queries,
        uint32_t              state_count,
        uint32_t              hidden_dim,
        uint32_t              query_count,
        float                 epsilon);

int ds4_inkling_control_metal(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *current,
        const ds4_gpu_tensor *message,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              ctrl_down_offset,
        uint64_t              ctrl_gain_offset,
        uint64_t              ctrl_forget_offset,
        uint32_t              hidden_dim,
        uint32_t              rank,
        float                 epsilon);

/* Runtime variant using session-owned scratch, with no per-call allocation. */
int ds4_inkling_control_metal_with_workspace(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *current,
        const ds4_gpu_tensor *message,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              ctrl_down_offset,
        uint64_t              ctrl_gain_offset,
        uint64_t              ctrl_forget_offset,
        uint32_t              hidden_dim,
        uint32_t              rank,
        float                 epsilon,
        const struct ds4_inkling_gpu_control_workspace *workspace);

/* Prefill variant with contiguous [token][hidden] current/message rows. */
int ds4_inkling_control_metal_batch_with_workspace(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *current,
        const ds4_gpu_tensor *message,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              ctrl_down_offset,
        uint64_t              ctrl_gain_offset,
        uint64_t              ctrl_forget_offset,
        uint32_t              hidden_dim,
        uint32_t              rank,
        uint32_t              n_tokens,
        float                 epsilon,
        const struct ds4_inkling_gpu_control_workspace *workspace);

#endif /* DS4_INKLING_LOOP_H */
