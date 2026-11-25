# Inkling2 grouped IQ4_NL correctness diagnostic (M4)

Validated on Apple M4, 2026-11-25. The standalone harness replays the checked-in
Metal GEMV and grouped GEMM kernels and compares staged/inline decode and
serial/two-SIMD final reductions. It records every FP32 lane-accumulator step,
the 256 lane finals, the eight first-stage SIMD sums, output bits, ULP distance,
maximum absolute error and nonfinite values. A/A, repeat-trace and
plain-versus-traced checks reject nondeterminism or instrumentation effects.

Run from the repository root:

    xcrun clang -O2 -fobjc-arc -Wall -Wextra \
      -framework Foundation -framework Metal \
      tests/test_inkling2_iq4_lane_trace.m -o /tmp/inkling2-iq4-lane-trace
    /tmp/inkling2-iq4-lane-trace
    /tmp/inkling2-iq4-lane-trace --safe

The synthetic replay covers both real feature widths, 32 routes for one expert
and one output row. To replay a real dot product, extract its packed IQ4_NL row
and FP32 input rows. The packed row is 1296 bytes for gate/up or 648 bytes for
down; inputs are 9216 or 4608 bytes per row respectively. Use raw little-endian
binary without headers and keep input rows in the expert's route-rank order.

    /tmp/inkling2-iq4-lane-trace gate-row.bin input-rows.bin
    /tmp/inkling2-iq4-lane-trace --safe gate-row.bin input-rows.bin

The reduced replay rebases expert and output row IDs to zero. It validates the
source kernels and their arithmetic but not full dispatch offsets, routing or
the whole model. Match its output bits against the full-dispatch capture before
using it as evidence about a production divergence.

## First real divergence

The first divergent product is physical layer 0, visit 0, token 0, route slot 0,
expert 77, gate/up output feature 0 in
`blk.0.ffn_gate_up_exps.weight` (IQ4_NL, shape 2304x2304x128). The tensor begins
at GGUF offset 1,217,233,184; expert 77/output feature 0 begins at offset
1,447,153,952. The replay matches the original full-dispatch output bits.

In lane 2, feature 2 leaves accumulator bits `0xbb2ae377` in both kernels.
At feature 258, scale -0.000284910202 times IQ4_NL codebook value 89 gives
weight -0.025357008 (`0xbccfb980`); the input is -0.0790742561
(`0xbda1f1af`), and the rounded product is `0x3b0367c5`. GEMV's separate
multiply then add gives accumulator `0xba1deec8`; GEMM's fused multiply-add
gives `0xba1deec7`. This is the first differing accumulation step. The same
pattern occurs in output feature 516 at feature 256: separate arithmetic gives
`0xbb1bc442`, fused arithmetic `0xbb1bc441`.

The decoded weight is exact in FP32 for the finite FP16 scale and IQ4_NL integer
codebook. Staged and inline decoders produce identical per-step accumulators;
the staged threadgroup load and feature indexing are not the cause. The
divergence is FMA contraction, followed by a distinct final-reduction tree in
the serial variant.

## Correction and validation

In default math mode, `DS4_INKLING2_IQ4_CONTRACT_OFF` scopes
`#pragma clang fp contract(off)` to the grouped IQ4 lane-accumulator loop. This
preserves the GEMV sequence of a rounded FP32 product followed by an FP32 add,
without globally enabling strict math or changing the GEMV control. Production
uses the existing staged decoded-weight cache. `DS4_INKLING2_IQ4_TWO_SIMD_FINAL`
reduces the eight lane-group sums with a second SIMD reduction, matching the
reference's reduction tree. Safe math leaves the local contraction macro off.

For the two real captured output rows, both staged and inline variants have
zero bit mismatches across 13,824 per-feature accumulators, 1,536 lane finals,
48 first-stage SIMD sums and all six route outputs. A/A and traced/untraced
outputs match in standard and safe modes. Serial final reduction still differs
on 3/6 outputs (maximum absolute error 1.79e-7 or 2.38e-7); the production
two-SIMD reduction matches 6/6 bitwise.

The full-shape Metal test reports zero error for gate/up and down across
589,824 outputs each. Two independent full-model runs at 32 and 34 tokens are
byte-identical to each other and to the GEMV control logits: 200,058 logits per
frontier, zero nonfinite values, zero tolerance violations and identical
argmax. A synchronized route capture matches the control on all 256 route IDs
at visits 0–4 (0/1,280 mismatches), including visit 4, token 3, slot 7 where
the earlier GEMM selected expert 41 and GEMV selected expert 8.
Before correction, the compared route samples disagreed 56/96 with the original
serial reduction and 68/96 with the second-SIMD reduction.

## Performance measurements

Three sequential `ds4-bench` prefill-only runs used the same M4, model, prompt,
four CPU helper threads, allocation (8192 tokens) and 512-token prefill chunk.
The GEMV control uses the same 32-token batching change as the experiment. The
pre-fix GEMM is commit `cfc4e6a`; fixed GEMM uses the local contraction control.

| Context | GEMV control mean (range) | Fixed grouped GEMM mean (range) | Speedup vs GEMV | Pre-fix grouped GEMM mean (range) |
|---:|---:|---:|---:|---:|
| 128 | 7.27 (7.12–7.51) tok/s | 12.03 (11.81–12.18) tok/s | 65.4% | 11.84 (11.84–11.85) tok/s |
| 512 | 6.45 (6.42–6.48) tok/s | 10.26 (10.06–10.45) tok/s | 59.2% | 10.42 (10.24–10.56) tok/s |

The corrected GEMM is within run-to-run spread of the pre-fix grouped GEMM.
`/usr/bin/time -l` reported 1.28–1.31 GB process peak footprint at 128 and
1.30–1.33 GB at 512. A synchronized route-only capture recorded 49,152 route
assignments at 128 tokens and 196,608 at 512; all 128 experts were used in both
captures. At 128, per-expert counts ranged from 94 to 876; at 512 they ranged
from 411 to 3,783.

The benchmark exposes end-to-end prefill throughput, not per-kernel MoE,
gate/up, down or route-map time. GPU occupancy, bandwidth and device-only
memory were not available from the current benchmark or the measurements used
here. Route capture used a temporary synchronized readback hook, removed after
measurement; its runs were excluded from performance results.
