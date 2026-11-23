#ifndef DS4_GGUF_H
#define DS4_GGUF_H

#include <stdint.h>

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

/* GGML type ids and canonical block sizes used by DS4's mmap GGUF directory. */
static const gguf_type_info gguf_types[] = {
    [0]  = {"f32",      1,   4},
    [1]  = {"f16",      1,   2},
    [2]  = {"q4_0",    32,  18},
    [3]  = {"q4_1",    32,  20},
    [6]  = {"q5_0",    32,  22},
    [7]  = {"q5_1",    32,  24},
    [8]  = {"q8_0",    32,  34},
    [9]  = {"q8_1",    32,  40},
    [10] = {"q2_k",   256,  84},
    [11] = {"q3_k",   256, 110},
    [12] = {"q4_k",   256, 144},
    [13] = {"q5_k",   256, 176},
    [14] = {"q6_k",   256, 210},
    [15] = {"q8_k",   256, 292},
    [16] = {"iq2_xxs",256,  66},
    [17] = {"iq2_xs", 256,  74},
    [18] = {"iq3_xxs",256,  98},
    [19] = {"iq1_s",  256,  50},
    [20] = {"iq4_nl",  32,  18},
    [21] = {"iq3_s",  256, 110},
    [22] = {"iq2_s",  256,  82},
    [23] = {"iq4_xs", 256, 136},
    [24] = {"i8",       1,   1},
    [25] = {"i16",      1,   2},
    [26] = {"i32",      1,   4},
    [27] = {"i64",      1,   8},
    [28] = {"f64",      1,   8},
    [29] = {"iq1_m",  256,  56},
    [30] = {"bf16",     1,   2},
    [39] = {"mxfp4",   32,  17},
};

enum {
    DS4_TENSOR_F32      = 0,
    DS4_TENSOR_F16      = 1,
    DS4_TENSOR_Q4_0     = 2,
    DS4_TENSOR_Q4_1     = 3,
    DS4_TENSOR_Q8_0     = 8,
    DS4_TENSOR_Q2_K     = 10,
    DS4_TENSOR_Q4_K     = 12,
    DS4_TENSOR_Q5_K     = 13,
    DS4_TENSOR_Q6_K     = 14,
    DS4_TENSOR_Q8_K     = 15,
    DS4_TENSOR_IQ2_XXS  = 16,
    DS4_TENSOR_IQ4_NL   = 20,
    DS4_TENSOR_I8       = 24,
    DS4_TENSOR_I32      = 26,
    DS4_TENSOR_BF16     = 30,
    DS4_TENSOR_MXFP4    = 39,
};

#endif /* DS4_GGUF_H */
