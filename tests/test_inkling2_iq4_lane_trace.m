/* Standalone M4 IQ4_NL diagnostic: compile the real kernels, with and without
 * instrumentation, and replay identical inputs through a 2x2 experiment.
 * No ds4 runtime, model allocation, or production math mode is changed. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LANES = 256, SIMD = 32, GROUPS = LANES / SIMD, MAX_ROUTES = 32,
       MAX_TRACE_DIM = 2304, TRACE_STRIDE = MAX_TRACE_DIM + LANES + GROUPS };
typedef struct {
    uint32_t in_dim, out_dim, n_experts, n_used, weight_type, blocks_per_row;
    uint32_t input_per_expert, n_tokens, simd_width, pad;
    uint64_t expert_bytes;
} gemv_args;
typedef struct {
    uint32_t in_dim, out_dim, n_experts, n_used, n_tokens, blocks_per_row;
    uint32_t input_per_expert, route_stride;
    uint64_t expert_bytes;
} gemm_args;
_Static_assert(sizeof(gemv_args) == 48, "GEMV Metal ABI");
_Static_assert(sizeof(gemm_args) == 40, "GEMM Metal ABI");

typedef struct {
    uint32_t out[MAX_ROUTES];
    uint32_t trace[MAX_ROUTES * TRACE_STRIDE];
} result;

static void fail(NSString *message, NSError *error) {
    fprintf(stderr, "%s: %s\n", message.UTF8String,
            error ? error.localizedDescription.UTF8String : "invalid input or device");
    exit(1);
}

static id<MTLLibrary> library(id<MTLDevice> device, NSString *source,
                              BOOL safe, BOOL trace, BOOL inl, BOOL simd) {
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (safe) {
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
    }
    NSMutableDictionary *macros = [NSMutableDictionary dictionary];
    if (trace) macros[@"DS4_INKLING2_IQ4_TRACE"] = @1;
    if (inl) macros[@"DS4_INKLING2_IQ4_DIAG_INLINE"] = @1;
    if (simd) macros[@"DS4_INKLING2_IQ4_DIAG_SIMD"] = @1;
#if defined(DS4_IQ4_TEST_SEPARATE_PRODUCT)
    if (!safe) macros[@"DS4_INKLING2_IQ4_SEPARATE_PRODUCT"] = @1;
#else
    if (!safe) macros[@"DS4_INKLING2_IQ4_CONTRACT_OFF"] = @1;
#endif
    options.preprocessorMacros = macros;
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:source options:options error:&error];
    if (!lib) fail(@"compile shader", error);
    return lib;
}

static id<MTLBuffer> buffer(id<MTLDevice> device, const void *data, NSUInteger size) {
    id<MTLBuffer> b = [device newBufferWithBytes:data length:size
                                        options:MTLResourceStorageModeShared];
    if (!b) fail(@"allocate buffer", nil);
    return b;
}

static void run(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> lib,
                id<MTLBuffer> weights, id<MTLBuffer> input, uint32_t dim,
                uint32_t routes, BOOL grouped, BOOL tracing, result *r) {
    NSString *name = grouped ? @"kernel_inkling2_expert_gemm_id_iq4_nl_scalar" :
                               @"kernel_inkling2_expert_gemv";
    NSError *error = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) fail(name, nil);
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!p) fail(name, error);
    if (p.threadExecutionWidth != SIMD || p.maxTotalThreadsPerThreadgroup < LANES)
        fail(@"requires 32-wide SIMD and 256 threads", nil);

    int32_t ids[MAX_ROUTES] = {0};
    uint32_t map[MAX_ROUTES];
    for (uint32_t i = 0; i < routes; i++) map[i] = i;
    id<MTLBuffer> ids_buf = buffer(device, ids, routes * sizeof(ids[0]));
    id<MTLBuffer> map_buf = buffer(device, map, routes * sizeof(map[0]));
    id<MTLBuffer> counts = buffer(device, &routes, sizeof(routes));
    memset(r, 0xff, sizeof(*r));
    id<MTLBuffer> out = buffer(device, r->out, routes * sizeof(uint32_t));
    id<MTLBuffer> trace = tracing ?
        buffer(device, r->trace, routes * TRACE_STRIDE * sizeof(uint32_t)) : nil;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!cb || !enc) fail(@"create encoder", nil);
    [enc setComputePipelineState:p];
    if (grouped) {
        gemm_args a = {dim, 1, 1, 1, routes, dim / 32, 0, routes, weights.length};
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:weights offset:0 atIndex:1];
        [enc setBuffer:map_buf offset:0 atIndex:2];
        [enc setBuffer:counts offset:0 atIndex:3];
        [enc setBuffer:input offset:0 atIndex:4];
        [enc setBuffer:out offset:0 atIndex:5];
        if (tracing) [enc setBuffer:trace offset:0 atIndex:6];
        [enc setThreadgroupMemoryLength:dim * sizeof(float) atIndex:0];
        [enc setThreadgroupMemoryLength:LANES * sizeof(float) atIndex:1];
    } else {
        gemv_args a = {dim, 1, 1, 1, 20, dim / 32, 0, routes, SIMD, 0, weights.length};
        [enc setBytes:&a length:sizeof(a) atIndex:0];
        [enc setBuffer:weights offset:0 atIndex:1];
        [enc setBuffer:ids_buf offset:0 atIndex:2];
        [enc setBuffer:input offset:0 atIndex:3];
        [enc setBuffer:out offset:0 atIndex:4];
        if (tracing) [enc setBuffer:trace offset:0 atIndex:5];
        [enc setThreadgroupMemoryLength:LANES * sizeof(float) atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(grouped ? 1 : routes, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(LANES, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) fail(name, cb.error);
    memcpy(r->out, out.contents, routes * sizeof(uint32_t));
    if (tracing) memcpy(r->trace, trace.contents, routes * TRACE_STRIDE * sizeof(uint32_t));
}

static float fp32(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint32_t ordered_bits(uint32_t bits) {
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

static uint64_t ulp_distance(uint32_t a, uint32_t b) {
    if (a == b) return 0;
    uint32_t ox = ordered_bits(a), oy = ordered_bits(b);
    return ox > oy ? (uint64_t)ox - oy : (uint64_t)oy - ox;
}

static unsigned compare(const char *label, const uint32_t *a, const uint32_t *b,
                        uint32_t routes, unsigned stride, unsigned offset, unsigned count) {
    unsigned bad = 0, nonfinite = 0;
    double max_abs = 0;
    uint64_t max_ulp = 0;
    for (uint32_t r = 0; r < routes; r++) {
        for (unsigned j = 0; j < count; j++) {
            unsigned i = r * stride + offset + j;
            float x = fp32(a[i]), y = fp32(b[i]);
            if (!isfinite(x) || !isfinite(y)) nonfinite++;
            else {
                max_abs = fmax(max_abs, fabs((double)x - y));
                uint64_t ulp = ulp_distance(a[i], b[i]);
                if (ulp > max_ulp) max_ulp = ulp;
            }
            if (a[i] != b[i]) {
                if (!bad) {
                    if (strcmp(label, "accumulator-step") == 0)
                        printf("  first %s mismatch: route=%u feature=%u lane=%u iteration=%u A=%08x B=%08x ULP=%llu\n",
                               label, r, j, j % LANES, j / LANES, a[i], b[i],
                               (unsigned long long)ulp_distance(a[i], b[i]));
                    else
                        printf("  first %s mismatch: route=%u index=%u A=%08x B=%08x ULP=%llu\n",
                               label, r, j, a[i], b[i],
                               (unsigned long long)ulp_distance(a[i], b[i]));
                    if (strcmp(label, "accumulator-step") == 0 && j >= LANES) {
                        unsigned previous = i - LANES;
                        printf("    prior accumulator: feature=%u A=%08x B=%08x\n",
                               j - LANES, a[previous], b[previous]);
                    }
                }
                bad++;
            }
        }
    }
    printf("  %s: bit_mismatch=%u/%u max_abs=%.9g max_ulp=%llu nonfinite=%u\n",
           label, bad, routes * count, max_abs, (unsigned long long)max_ulp, nonfinite);
    return bad + nonfinite;
}

static unsigned compare_trace(const result *a, const result *b, uint32_t routes, uint32_t dim) {
    unsigned stride = dim + LANES + GROUPS;
    unsigned bad = compare("accumulator-step", a->trace, b->trace, routes, stride, 0, dim);
    bad += compare("lane-final", a->trace, b->trace, routes, stride, dim, LANES);
    bad += compare("SIMD", a->trace, b->trace, routes, stride, dim + LANES, GROUPS);
    return bad + compare("output", a->out, b->out, routes, 1, 0, 1);
}

static int replay(id<MTLDevice> device, NSString *source, NSData *w, NSData *x,
                  uint32_t dim, BOOL safe) {
    uint32_t routes = (uint32_t)(x.length / (dim * sizeof(float)));
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) fail(@"create queue", nil);
    id<MTLBuffer> weights = buffer(device, w.bytes, w.length);
    id<MTLBuffer> input = buffer(device, x.bytes, x.length);
    result a, aa, at, att, b, bt, btt;
    unsigned invalid = 0, production_bad = 0;
    printf("shape=%u routes=%u math=%s\n", dim, routes, safe ? "safe" : "default");
    for (unsigned mode = 0; mode < 4; mode++) {
        @autoreleasepool {
            BOOL inl = (mode & 2u) != 0, simd = (mode & 1u) != 0;
            id<MTLLibrary> plain = library(device, source, safe, NO, inl, simd);
            id<MTLLibrary> traced = library(device, source, safe, YES, inl, simd);
            if (!mode) {
                puts("GEMV A/A (plain):");
                run(device, queue, plain, weights, input, dim, routes, NO, NO, &a);
                run(device, queue, plain, weights, input, dim, routes, NO, NO, &aa);
                invalid += compare("output", a.out, aa.out, routes, 1, 0, 1);
                run(device, queue, traced, weights, input, dim, routes, NO, YES, &at);
                run(device, queue, traced, weights, input, dim, routes, NO, YES, &att);
                puts("GEMV A/A (traced):");
                invalid += compare_trace(&at, &att, routes, dim);
                puts("GEMV plain/traced (instrumentation check):");
                invalid += compare("output", a.out, at.out, routes, 1, 0, 1);
            }
            printf("GEMM %s / %s\n", inl ? "inline" : "staged",
                   simd ? "two SIMD reductions" : "serial final reduction");
            run(device, queue, plain, weights, input, dim, routes, YES, NO, &b);
            run(device, queue, traced, weights, input, dim, routes, YES, YES, &bt);
            run(device, queue, traced, weights, input, dim, routes, YES, YES, &btt);
            puts("  GEMM plain/traced (instrumentation check):");
            invalid += compare("output", b.out, bt.out, routes, 1, 0, 1);
            puts("  GEMM traced repeat:");
            invalid += compare_trace(&bt, &btt, routes, dim);
            puts("  GEMV/GEMM plain:");
            unsigned diff = compare("output", a.out, b.out, routes, 1, 0, 1);
            puts("  GEMV/GEMM traced:");
            unsigned trace_diff = compare_trace(&at, &bt, routes, dim);
            if (simd) production_bad += diff + trace_diff;
        }
    }
    if (invalid) {
        puts("INVALID TRACE: nondeterminism, nonfinite values or instrumentation changed output.");
        return 1;
    }
    return production_bad ? 2 : 0;
}

static int synthetic(id<MTLDevice> device, NSString *source, uint32_t dim, BOOL safe) {
    NSMutableData *w = [NSMutableData dataWithLength:(dim / 32) * 18];
    uint8_t *bytes = w.mutableBytes;
    uint32_t seed = 0x12345678u;
    for (uint32_t b = 0; b < dim / 32; b++) {
        uint16_t half = (uint16_t)(0x1800u + (b % 9u) * 0x80u);
        memcpy(bytes + 18 * b, &half, 2);
        for (unsigned j = 2; j < 18; j++) {
            seed = seed * 1664525u + 1013904223u;
            bytes[18 * b + j] = (uint8_t)(seed >> 24);
        }
    }
    NSMutableData *x = [NSMutableData dataWithLength:MAX_ROUTES * dim * sizeof(float)];
    float *v = x.mutableBytes;
    for (uint32_t i = 0; i < MAX_ROUTES * dim; i++) {
        seed = seed * 1664525u + 1013904223u;
        v[i] = ((int32_t)(seed >> 8) - 8388608) / 8388608.0f;
    }
    return replay(device, source, w, x, dim, safe);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (ulp_distance(0x00000000u, 0x80000000u) != 1u ||
            ulp_distance(0x3f800000u, 0x3f800001u) != 1u) {
            fputs("ULP ordering self-check failed\n", stderr);
            return 1;
        }
        BOOL safe = argc > 1 && strcmp(argv[1], "--safe") == 0;
        int pos = safe ? 2 : 1;
        if (argc != pos && argc != pos + 2) {
            fprintf(stderr, "usage: %s [--safe] [packed-iq4-row.bin input-rows-f32.bin]\n", argv[0]);
            return 1;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) fail(@"no Metal device", nil);
        printf("device=%s\n", device.name.UTF8String);
        NSError *error = nil;
        NSString *kernel = [NSString stringWithContentsOfFile:@"metal/inkling2.metal"
                                                  encoding:NSUTF8StringEncoding error:&error];
        if (!kernel) fail(@"read metal/inkling2.metal from repository root", error);
        NSString *source = [@"#include <metal_stdlib>\nusing namespace metal;\n"
                             stringByAppendingString:kernel];
        if (argc == pos) {
            int a = synthetic(device, source, 2304, safe);
            int b = synthetic(device, source, 1152, safe);
            return (a == 1 || b == 1) ? 1 : (a || b ? 2 : 0);
        }
        NSData *w = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[pos]]];
        NSData *x = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[pos + 1]]];
        if (!w || !x || (w.length != 1296 && w.length != 648))
            fail(@"expected one packed 2304- or 1152-feature IQ4_NL row", nil);
        uint32_t dim = (uint32_t)(w.length / 18 * 32);
        NSUInteger row_bytes = dim * sizeof(float);
        if (x.length == 0 || x.length % row_bytes || x.length / row_bytes > MAX_ROUTES)
            fail(@"expected 1..32 contiguous input rows of matching FP32 width", nil);
        return replay(device, source, w, x, dim, safe);
    }
}
