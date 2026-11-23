#include "ds4_inkling_loop.h"
#include "ds4_inkling_gpu_state.h"

#include <math.h>
#include <stdio.h>

#define DS4_INKLING_MAX_HIDDEN 2304u
#define DS4_INKLING_MAX_CTRL_RANK 64u
#define DS4_INKLING_MAX_MIX_STATES 6u
#define DS4_INKLING_MAX_MIX_QUERIES 4u

static int ds4_inkling_valid_epsilon(float epsilon) {
    return isfinite(epsilon) && epsilon > 0.0f;
}

static float ds4_inkling_rms_inverse(const float *input, uint32_t width,
                                    float epsilon) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < width; i++) sum += input[i] * input[i];
    return 1.0f / sqrtf(sum / (float)width + epsilon);
}

int ds4_inkling_mix_ref(
        float       *out,
        const float *states,
        const float *queries,
        uint32_t     state_count,
        uint32_t     hidden_dim,
        uint32_t     query_count,
        float        epsilon) {
    if (!out || !states || !queries || state_count == 0 ||
        state_count > DS4_INKLING_MAX_MIX_STATES || hidden_dim == 0 ||
        query_count == 0 || query_count > DS4_INKLING_MAX_MIX_QUERIES ||
        !ds4_inkling_valid_epsilon(epsilon)) return 0;

    float inverse_norm[DS4_INKLING_MAX_MIX_STATES];
    float scores[DS4_INKLING_MAX_MIX_STATES];
    float probabilities[DS4_INKLING_MAX_MIX_STATES];
    for (uint32_t state = 0; state < state_count; state++) {
        inverse_norm[state] = ds4_inkling_rms_inverse(
            states + (uint64_t)state * hidden_dim, hidden_dim, epsilon);
    }

    const float score_scale = 1.0f / sqrtf((float)hidden_dim);
    for (uint32_t query = 0; query < query_count; query++) {
        float max_score = -INFINITY;
        for (uint32_t state = 0; state < state_count; state++) {
            const float *x = states + (uint64_t)state * hidden_dim;
            const float *q = queries + (uint64_t)query * hidden_dim;
            float dot = 0.0f;
            for (uint32_t i = 0; i < hidden_dim; i++) {
                dot += x[i] * inverse_norm[state] * q[i];
            }
            scores[state] = dot * score_scale;
            if (scores[state] > max_score) max_score = scores[state];
        }
        float denominator = 0.0f;
        for (uint32_t state = 0; state < state_count; state++) {
            probabilities[state] = expf(scores[state] - max_score);
            denominator += probabilities[state];
        }
        if (!(denominator > 0.0f) || !isfinite(denominator)) return 0;
        const float inverse_denominator = 1.0f / denominator;
        for (uint32_t state = 0; state < state_count; state++)
            probabilities[state] *= inverse_denominator;

        float *y = out + (uint64_t)query * hidden_dim;
        for (uint32_t i = 0; i < hidden_dim; i++) {
            float sum = 0.0f;
            for (uint32_t state = 0; state < state_count; state++) {
                sum += probabilities[state] * inverse_norm[state] *
                    states[(uint64_t)state * hidden_dim + i];
            }
            y[i] = sum;
        }
    }
    return 1;
}

int ds4_inkling_control_ref(
        float       *out,
        const float *current,
        const float *message,
        const float *ctrl_down,
        const float *ctrl_gain,
        const float *ctrl_forget,
        uint32_t     hidden_dim,
        uint32_t     rank,
        float        epsilon) {
    if (!out || !current || !message || !ctrl_down || !ctrl_gain ||
        !ctrl_forget || hidden_dim == 0 ||
        hidden_dim > DS4_INKLING_MAX_HIDDEN || rank == 0 ||
        rank > DS4_INKLING_MAX_CTRL_RANK ||
        !ds4_inkling_valid_epsilon(epsilon)) return 0;

    float normalized[2u * DS4_INKLING_MAX_HIDDEN];
    float bottleneck[DS4_INKLING_MAX_CTRL_RANK];
    float gain[DS4_INKLING_MAX_HIDDEN];
    float forget[DS4_INKLING_MAX_HIDDEN];
    const float cur_inv = ds4_inkling_rms_inverse(current, hidden_dim, epsilon);
    const float msg_inv = ds4_inkling_rms_inverse(message, hidden_dim, epsilon);
    for (uint32_t i = 0; i < hidden_dim; i++) {
        normalized[i] = current[i] * cur_inv;
        normalized[hidden_dim + i] = message[i] * msg_inv;
    }
    for (uint32_t row = 0; row < rank; row++) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < 2u * hidden_dim; col++) {
            sum += ctrl_down[(uint64_t)row * 2u * hidden_dim + col] * normalized[col];
        }
        bottleneck[row] = sum;
    }
    for (uint32_t row = 0; row < hidden_dim; row++) {
        float gain_sum = 0.0f;
        float forget_sum = 0.0f;
        for (uint32_t col = 0; col < rank; col++) {
            const float weight = ctrl_gain[(uint64_t)row * rank + col];
            const float forget_weight = ctrl_forget[(uint64_t)row * rank + col];
            gain_sum += weight * bottleneck[col];
            forget_sum += forget_weight * bottleneck[col];
        }
        gain[row] = tanhf(gain_sum);
        forget[row] = tanhf(forget_sum);
        out[row] = current[row] - forget[row] * current[row] +
                   gain[row] * message[row];
    }
    return 1;
}

int ds4_inkling_mix_metal(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *states,
        const ds4_gpu_tensor *queries,
        uint32_t              state_count,
        uint32_t              hidden_dim,
        uint32_t              query_count,
        float                 epsilon) {
    return ds4_gpu_inkling2_mix_tensor(out, states, queries, state_count,
                                       hidden_dim, query_count, epsilon);
}

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
        const ds4_inkling_gpu_control_workspace *workspace) {
    if (!out || !current || !message || !model_map || hidden_dim == 0 ||
        hidden_dim > DS4_INKLING_MAX_HIDDEN || rank == 0 ||
        rank > DS4_INKLING_MAX_CTRL_RANK ||
        !ds4_inkling_valid_epsilon(epsilon) || !workspace ||
        !workspace->normalized || !workspace->bottleneck ||
        !workspace->gain || !workspace->forget) return 0;

    const uint64_t hidden_bytes = (uint64_t)hidden_dim * sizeof(float);
    int ok = ds4_gpu_tensor_bytes(workspace->normalized) >= 2u * hidden_bytes &&
             ds4_gpu_tensor_bytes(workspace->bottleneck) >=
                 (uint64_t)rank * sizeof(float) &&
             ds4_gpu_tensor_bytes(workspace->gain) >= hidden_bytes &&
             ds4_gpu_tensor_bytes(workspace->forget) >= hidden_bytes;
    const char *failed_stage = ok ? NULL : "scratch allocation";
    if (ok && !ds4_gpu_inkling2_control_prepare_tensor(
                  workspace->normalized, current, message, hidden_dim, epsilon)) {
        ok = 0;
        failed_stage = "input normalization";
    }
    if (ok && !ds4_gpu_matmul_f32_tensor(
                  workspace->bottleneck, model_map, model_size, ctrl_down_offset,
                  2u * hidden_dim, rank, workspace->normalized, 1u)) {
        ok = 0;
        failed_stage = "controller down projection";
    }
    if (ok && !ds4_gpu_matmul_f32_tensor(
                  workspace->gain, model_map, model_size, ctrl_gain_offset,
                  rank, hidden_dim, workspace->bottleneck, 1u)) {
        ok = 0;
        failed_stage = "controller gain projection";
    }
    if (ok && !ds4_gpu_matmul_f32_tensor(
                  workspace->forget, model_map, model_size, ctrl_forget_offset,
                  rank, hidden_dim, workspace->bottleneck, 1u)) {
        ok = 0;
        failed_stage = "controller forget projection";
    }
    if (ok && !ds4_gpu_inkling2_control_update_tensor(
                  out, current, message, workspace->gain,
                  workspace->forget, hidden_dim)) {
        ok = 0;
        failed_stage = "controller gated update";
    }
    if (!ok) fprintf(stderr, "ds4: Inkling2 loop controller failed at %s\n", failed_stage);
    return ok;
}

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
        float                 epsilon) {
    ds4_inkling_gpu_control_workspace workspace = {0};
    workspace.normalized = ds4_gpu_tensor_alloc(
        (uint64_t)2u * hidden_dim * sizeof(float));
    workspace.bottleneck = ds4_gpu_tensor_alloc((uint64_t)rank * sizeof(float));
    workspace.gain = ds4_gpu_tensor_alloc((uint64_t)hidden_dim * sizeof(float));
    workspace.forget = ds4_gpu_tensor_alloc((uint64_t)hidden_dim * sizeof(float));
    const int ok = workspace.normalized && workspace.bottleneck &&
        workspace.gain && workspace.forget &&
        ds4_inkling_control_metal_with_workspace(
            out, current, message, model_map, model_size,
            ctrl_down_offset, ctrl_gain_offset, ctrl_forget_offset,
            hidden_dim, rank, epsilon, &workspace);
    ds4_gpu_tensor_free(workspace.normalized);
    ds4_gpu_tensor_free(workspace.bottleneck);
    ds4_gpu_tensor_free(workspace.gain);
    ds4_gpu_tensor_free(workspace.forget);
    return ok;
}
