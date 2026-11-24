#ifndef DS4_INKLING_GPU_STATE_H
#define DS4_INKLING_GPU_STATE_H

#include "ds4_gpu.h"
#include "ds4_inkling_state.h"

typedef struct {
    ds4_gpu_tensor *key;
    ds4_gpu_tensor *value;
    ds4_gpu_tensor *conv[DS4_INKLING_CONV_GROUP_COUNT];
    uint32_t capacity;
    uint32_t length;
    uint64_t next_position;
    uint32_t conv_length[DS4_INKLING_CONV_GROUP_COUNT];
    uint64_t key_offset;
    uint64_t value_offset;
    uint64_t conv_offset[DS4_INKLING_CONV_GROUP_COUNT];
} ds4_inkling_gpu_slot;

typedef struct ds4_inkling_gpu_control_workspace {
    ds4_gpu_tensor *normalized;
    ds4_gpu_tensor *bottleneck;
    ds4_gpu_tensor *gain;
    ds4_gpu_tensor *forget;
} ds4_inkling_gpu_control_workspace;

typedef struct ds4_inkling_gpu_moe_workspace {
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *ids;
    ds4_gpu_tensor *weights;
    ds4_gpu_tensor *gate_up;
    ds4_gpu_tensor *activation;
    ds4_gpu_tensor *expert_outputs;
    ds4_gpu_tensor *zero_bias;
} ds4_inkling_gpu_moe_workspace;

typedef struct {
    ds4_gpu_tensor *attn_q_clip;
    ds4_gpu_tensor *attn_rel_bias;
    ds4_gpu_tensor *attn_rel_clip;
    ds4_gpu_tensor *attn_k_conv;
    ds4_gpu_tensor *attn_v_conv;
    ds4_gpu_tensor *attn_post_conv;
    ds4_gpu_tensor *ffn_post_conv;
    ds4_gpu_tensor *attn_res_query;
    ds4_gpu_tensor *ffn_res_query;
    ds4_gpu_tensor *router_bias;
    ds4_gpu_tensor *loop_msg_query;
} ds4_inkling_gpu_layer_parameters;

typedef struct {
    ds4_gpu_tensor *output_res_query;
    ds4_inkling_gpu_layer_parameters layer[DS4_INKLING_PHYSICAL_LAYERS];
} ds4_inkling_gpu_parameters;

/* Scratch views are allocated with the session and reused by every visit. */
typedef struct {
    ds4_gpu_tensor *current;
    ds4_gpu_tensor *next;
    ds4_gpu_tensor *mix_states;
    ds4_gpu_tensor *mix_outputs;
    ds4_gpu_tensor *mix_output_rows[4];
    ds4_gpu_tensor *controlled_inputs;
    ds4_gpu_tensor *controlled_input_rows[3];
    ds4_gpu_tensor *query;
    ds4_gpu_tensor *prepared_query;
    ds4_gpu_tensor *key;
    ds4_gpu_tensor *prepared_key;
    ds4_gpu_tensor *value;
    ds4_gpu_tensor *prepared_value;
    ds4_gpu_tensor *relative;
    ds4_gpu_tensor *prepared_relative;
    ds4_gpu_tensor *sink_logits;
    ds4_gpu_tensor *relative_bias;
    ds4_gpu_tensor *relative_distances;
    ds4_gpu_tensor *attention_gate;
    ds4_gpu_tensor *attention_context;
    ds4_gpu_tensor *attention_gated;
    ds4_gpu_tensor *attention_output;
    ds4_gpu_tensor *ffn_input;
    ds4_gpu_tensor *ffn_mixed;
    ds4_gpu_tensor *messages;
    ds4_gpu_tensor *message_rows[4];
    ds4_gpu_tensor *previous_loop_states;
    ds4_gpu_tensor *dynamic_k;
    ds4_gpu_tensor *dynamic_v;
} ds4_inkling_gpu_layer_workspace;

typedef struct {
    ds4_gpu_tensor *current_storage;
    ds4_gpu_tensor *next_storage;
    ds4_gpu_tensor *prelude_storage;
    ds4_gpu_tensor *current_blocks_storage;
    ds4_gpu_tensor *previous_blocks_storage;
    ds4_gpu_tensor *ffn_input_storage;
    ds4_gpu_tensor *moe_input_storage;
    ds4_gpu_tensor *moe_output_storage;
    ds4_gpu_tensor *attn_norm_storage[3];
    ds4_gpu_tensor *query_storage;
    ds4_gpu_tensor *key_storage;
    ds4_gpu_tensor *value_storage;
    ds4_gpu_tensor *relative_storage;
    ds4_gpu_tensor *dynamic_k_storage;
    ds4_gpu_tensor *dynamic_v_storage;
    ds4_gpu_tensor *sink_logits_storage;
    ds4_gpu_tensor *attention_gate_storage;
    ds4_gpu_tensor *attention_gated_storage;
    ds4_gpu_tensor *attention_output_storage;
    ds4_gpu_tensor *attn_norm[3][DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *query[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *key[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *value[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *relative[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *dynamic_k[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *dynamic_v[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *sink_logits[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *attention_gate[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *attention_gated[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *attention_output[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *current[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *next[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *prelude[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *current_blocks[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *previous_blocks[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *ffn_input[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *moe_input[DS4_INKLING_GPU_MOE_MAX_TOKENS];
    ds4_gpu_tensor *moe_output[DS4_INKLING_GPU_MOE_MAX_TOKENS];
} ds4_inkling_gpu_batch_workspace;

/* Per-session Metal state. Weight owners refer to the engine's 24 bound
 * physical layers; every visit has its own KV and convolution views. */
typedef struct {
    ds4_inkling_plan plan;
    uint32_t weight_owner_count;
    uint32_t weight_owner[DS4_INKLING_PHYSICAL_LAYERS];
    uint32_t context_capacity;
    uint32_t position;
    uint64_t allocated_bytes;
    ds4_gpu_tensor *key_storage;
    ds4_gpu_tensor *value_storage;
    ds4_gpu_tensor *conv_storage;
    ds4_gpu_tensor *work_storage;
    ds4_gpu_tensor *current_blocks;
    ds4_gpu_tensor *previous_blocks;
    ds4_gpu_tensor *prelude_output;
    ds4_gpu_tensor *input_token;
    ds4_gpu_tensor *embedding;
    ds4_gpu_tensor *activation;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *scratch;
    ds4_gpu_tensor *weight_storage;
    ds4_inkling_gpu_parameters parameters;
    ds4_inkling_gpu_control_workspace control;
    ds4_inkling_gpu_moe_workspace moe;
    ds4_inkling_gpu_layer_workspace layer;
    ds4_inkling_gpu_batch_workspace batch;
    ds4_inkling_gpu_slot slots[DS4_INKLING_CACHE_SLOTS];
} ds4_inkling_gpu_context;

bool ds4_inkling_gpu_context_init(
        ds4_inkling_gpu_context *context,
        const ds4_inkling_plan *plan,
        uint32_t context_capacity);
bool ds4_inkling_gpu_context_required_bytes(
        const ds4_inkling_plan *plan,
        uint32_t context_capacity,
        uint64_t *required_bytes);
void ds4_inkling_gpu_context_free(ds4_inkling_gpu_context *context);
void ds4_inkling_gpu_context_reset(ds4_inkling_gpu_context *context);

#endif
