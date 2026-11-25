# Inkling2 grouped IQ4_NL lane diagnostic (M4)

Status: **NOT READY — CORRECTNESS FAILURE**. This diagnostic changes no
default kernel arithmetic and does not authorize performance benchmarks.

Both original host dispatches use 256 threads; GEMV obtains SIMD width from
the pipeline and GEMM assumes 32. With finite FP16 scales, multiplying by the
IQ4_NL integer codebook (magnitude <=127) is exact in FP32. Thus storing the
decoded weight in threadgroup FP32 is not itself an additional rounding step.
Compiler reassociation, contraction, or reduction order remain hypotheses.

Run from repository root on the M4 (macOS 15 SDK or newer):

    xcrun clang -O2 -fobjc-arc -Wall -Wextra \
      -framework Foundation -framework Metal \
      tests/test_inkling2_iq4_lane_trace.m -o /tmp/inkling2-iq4-lane-trace
    /tmp/inkling2-iq4-lane-trace
    /tmp/inkling2-iq4-lane-trace --safe

The synthetic replay uses both real feature widths, 32 routes for one expert
and one output row. To replay the first divergent gate/up dot product, extract
the packed 1296-byte row for its expert and output feature from the mapped
GGUF weights (18 bytes per 32 features), and the corresponding normalized
FP32 input row(s), 9216 bytes per token. Use raw little-endian binary, no
header. For down use 648-byte weights and 4608-byte activation rows. Order
inputs by the expert's original route rank; put the offending route in the
replay even if testing a single input. Then run:

    /tmp/inkling2-iq4-lane-trace gate-row.bin input-rows.bin
    /tmp/inkling2-iq4-lane-trace --safe gate-row.bin input-rows.bin

This is a reduced replay of the real source kernels, not a full graph
capture. Expert and output row IDs are rebased to zero; it does not test
expert offsets, router decisions, full dispatch shape or attention. Match
the output bits against the original captured dot before attributing drift
to the isolated operation.

For each of four GEMM variants (staged/inline decode x serial/SIMD final
reduction), the program compares the original GEMV lane accumulators (256),
first SIMD reductions (8), and final output using raw FP32 bit patterns.
Plain dispatches are also compared to traced dispatches: extra stores may
change compiler optimization. A/A and repeat trace checks reject
nondeterminism, nonfinite values or a changed output due to instrumentation.
The diagnostic compiles a standalone library from the checked-in shader
file, so the full runtime compiler context may also differ.

Interpretation:
- Lane sums differ: investigate arithmetic/compiler lowering first.
- Lane sums match but SIMD sums differ: investigate the first reduction.
- Lane and SIMD sums match but output differs: final reduction ordering.
- Output matches in a candidate only: repeat on real captures, then validate
  full model routing/logits at 32 and 34 tokens before considering 128/512.
- Instrumentation changes output or results are nonfinite: discard trace as
  evidence, retain plain comparison, investigate compiler sensitivity.

Exit codes: 0 = original GEMM matches GEMV bitwise on this replay and traces
are valid; 1 = invalid/inconclusive trace or runtime failure; 2 = original
GEMM bit mismatch on an otherwise valid replay. No exit code certifies
whole-model correctness. The diagnostic macros are not set by ds4.
