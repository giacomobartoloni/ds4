# Inkling 2 Mini on Apple Metal

This document tracks native DS4 support for the text-only Inkling 2 Mini GGUF. The first supported model format is the mixed IQ4_NL GGUF, with an 8192-token context and FP16 key/value state. DS4 loads the GGUF directly through its mmap loader and executes operators in its native C and Metal paths; it does not use llama.cpp or MLX as a runtime dependency.

## Milestone status

| Milestone | Status | Evidence |
| --- | --- | --- |
| M0: repository, references, and host baseline | Complete | Build and server results below; Metal regression baseline captured before implementation. |
| M1: GGUF inspector and 48-visit mapping | Complete for no-weight validation | Inspector and planner tests pass; live GGUF validation remains pending because weights are absent. |
| M2: native Metal control operators | Complete for isolated no-weight tests | Focused scalar-oracle tests pass on Apple M4; model integration is not part of this milestone. |
| M3: quantized token-vector primitives | Complete for synthetic block tests | IQ4_NL decode/GEMV, existing Q8_0 GEMV reuse, and top-8 output accumulation pass scalar-oracle tests. |
| M4: attention, convolution, and visit state | Partial, isolated operators | State, decode attention, Q/K preparation, relative bias, and shortconv pass synthetic state/oracle tests on Apple M4; prefill graph and model integration remain open. |
| M5: routed expert FFN | Partial, isolated Metal operation | Router, top-8 IQ4_NL/Q8_0 expert projections, SiTU, down projection, and routed accumulation pass synthetic scalar-oracle tests; DS4 layer/model wiring remains open. |
| M6+: loop, generation, numerical parity, and model performance | Not started | Model-dependent integration and validation require the GGUF to be available locally. |

## M0 baseline

- Repository: `giacomobartoloni/ds4`, branch `feature/inkling2-mini-metal-m4`, based on `b3f50180ec2c7324892be1f50bc83555ccb8d5e7`.
- At baseline, fork `main` and upstream `antirez/ds4` both pointed to `b3f50180ec2c7324892be1f50bc83555ccb8d5e7`; the checkout had no configured upstream remote.
- Host class: MacBook Pro with Apple M4, 32 GB unified memory, macOS 26.2. Metal reported a recommended maximum working set of 24.96 GiB.
- `make clean && make`: PASS.
- `make cpu`: PASS.
- `./ds4_test --server`: PASS (`server: OK`, `ds4 tests: ok`).
- `./ds4_test --metal-kernels`: FAIL on the unmodified baseline with 29 assertions. Four compressor pair state-store cases fail at `tests/ds4_test.c:1678` and the associated output comparisons at lines 1754–1758. Four gathered KV staging cases fail kernel selection at line 3315; raw and compressed output comparisons fail at lines 3365–3366. The failure log is preserved locally as `/tmp/ds4-inkling2-metal-baseline.log`; this is baseline evidence, not a result from Inkling code.
- Metal runtime source compilation and device initialization were reached on the M4 during the kernel suite. The standalone Xcode build tool and `metal` compiler are unavailable, so offline shader compilation is not available.
- The requested GGUF is not present. `make test` (which requires `ds4flash.gguf`), live GGUF inspection, model inference, numerical comparison against a reference runtime, and throughput/memory benchmarks have not been run. No model weights were downloaded.

## Pinned implementation references

These source revisions were inspected as format and semantic references; DS4 does not call them at runtime.

- llama.cpp: [`212c547fe908b288f6659d48741e6ffac1ff409c`](https://github.com/ggml-org/llama.cpp/tree/212c547fe908b288f6659d48741e6ffac1ff409c), including [`src/models/inkling2.cpp`](https://github.com/ggml-org/llama.cpp/blob/212c547fe908b288f6659d48741e6ffac1ff409c/src/models/inkling2.cpp) and [`src/models/inkling2-loop.cpp`](https://github.com/ggml-org/llama.cpp/blob/212c547fe908b288f6659d48741e6ffac1ff409c/src/models/inkling2-loop.cpp).
- Hugging Face Transformers: [`d9bdc606d543e1a85c0d7121af9379fbd2445bf3`](https://github.com/huggingface/transformers/tree/d9bdc606d543e1a85c0d7121af9379fbd2445bf3).
- Inkling 2 Mini model repository revision: [`7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`](https://huggingface.co/HuggingFaceTB/Inkling-2-Mini/tree/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6).
- IQ4_NL GGUF repository revision: [`e8a2cecf0e78914f07cc27b9be711a9b2c7561d9`](https://huggingface.co/HuggingFaceTB/Inkling-2-Mini-GGUF/tree/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9).

## Initial execution contract

- 24 physical layers, with loop core `[6,18)` executed in three passes. The resulting 48 logical visits reuse those 24 physical weight owners and each have an independent cache slot.
- The pinned model's 24-entry sliding-window pattern identifies physical full-attention layers 5, 11, 17, and 23. Across the 48 visits this yields 8 full-attention and 40 sliding-window cache slots.
- Initial inference scope: one text-only session, batch size 1, context 8192, IQ4_NL expert matrices, Q8_0 other matrices, F32 loop controller tensors, and FP16 KV. Unsupported model metadata must be rejected rather than silently assumed.
- A benchmark or numerical-parity claim requires the model file to be present and the exact tested file, prompt, context, and run count to be recorded.

## GGUF inspector

`gguf-tools/inkling2_inspect.py` reads GGUF v3 metadata and tensor descriptors. It validates tensor shapes, canonical byte sizes for the supported F32/F16/Q8_0/IQ4_NL tensor types, alignment, overlap, and file bounds without reading tensor payloads. Use `--summary`, `--tensors`, `--json`, or `--validate-inkling2`; all modes report that payload bytes were not read. The Inkling validator checks the loop metadata, 24-entry attention schedule, 48 logical visits, 156 F32 loop-controller tensors, and the IQ4_NL/Q8_0 expert tensor assignments.

Synthetic coverage: `make test-inkling2-inspect` passes 18 tests for valid descriptors, primitive/array metadata, truncation and size limits, duplicate keys and names, invalid shapes and types, offset alignment/overlap/bounds, quant block sizes, the expected Inkling layout, and payload-free inspection. The live GGUF is absent, so `--validate-inkling2` has not yet been run against model bytes.

## 48-visit plan and loader block sizes

`ds4_inkling.h` validates the supported 24-layer metadata and emits the execution order: prelude layers `[0,6)`, core `[6,18)` for three passes, then epilogue `[18,24)`. Each plan entry stores a physical layer index as its shared weight owner and a unique cache slot. Pass 0 uses cache slots 6–17, pass 1 uses 24–35, and pass 2 uses 36–47; prelude and epilogue use their physical indices. The planner validates the supplied attention pattern and rejects changed boundaries, pass counts, residual block size, or layer schedule. It describes the cache ownership needed by the runtime; inference buffers are not yet connected to it.

`ds4_gguf.h` is the shared type-size table used by the mmap loader and the C test. It corrects GGML type 19 IQ1_S to 256 elements / 50 bytes and type 20 IQ4_NL to 32 elements / 18 bytes. The canonical IQ4_NL layout is a 2-byte scale plus 16 packed bytes (`QK4_NL=32`) in the pinned llama.cpp source.

`make test-inkling2-mapping` passes: 48 ordered visits, 48 unique cache slots, 24 physical weight owners, 8 full-attention visits, 40 sliding-window visits, and canonical IQ1_S/IQ4_NL block sizes. The mapping is currently a tested planning primitive, not an end-to-end inference path.

## M2 native Metal control operators

`metal/inkling2.metal` is included in DS4's runtime Metal source library. The Objective-C wrappers use DS4 tensors, command buffers, and pipeline cache; RMSNorm reuses `ds4_gpu_rms_norm_plain_rows_tensor` instead of adding a second implementation.

- Recurrent gate update computes `(1 - forget) * current + gain * message` elementwise.
- Row-wise L2 clipping accumulates each row's squared norm in FP32 and scales rows above the positive finite limit. Rows shorter than a threadgroup and all-zero rows are covered.
- The router is fixed to the model's 128 experts and up to 8 selected experts. It ranks `sigmoid(logit) + correction_bias`, breaks ties by lower expert ID, then normalizes the un-biased sigmoid scores and applies the routed scale. NaN sigmoid inputs map to zero; NaN adjusted scores rank last. One GPU lane scans each token serially for this correctness-first implementation.
- Logit softcap computes `cap * tanh(logit / cap)` and explicitly maps positive/negative infinity to positive/negative `cap`; NaNs propagate.

`make test-inkling2-metal` passes on the Apple M4. The test compares these operators to scalar C references, including router ties, L2 clipping boundaries, zero/short rows, and softcap infinities/NaN. RMSNorm uses three independently compared rows, including a zero row.

After M2, `./ds4_test --metal-kernels` still reports exactly the 29 baseline assertion failures. Comparing captured assertion records against `/tmp/ds4-inkling2-metal-baseline.log` found no new or resolved records. Existing failures remain in compressor pair state-store and gathered KV staging/kernel-selection cases. No M2 throughput claim is made; the serial router and reductions have not been benchmarked or optimized.

These operators are not yet connected to a model execution path. The live Inkling GGUF remains absent, so end-to-end generation and same-weight numerical parity remain unverified.

## M3 quantized token-vector primitives

The IQ4_NL Metal GEMV reads the canonical 18-byte block layout directly: a little-endian FP16 scale followed by 16 packed bytes for 32 values, using the GGML `kvalues_iq4nl` lookup table. Low nibbles address values 0–15 and high nibbles address values 16–31. The kernel decodes each selected weight inline, accumulates in FP32 across a cooperative threadgroup, and does not materialize a dequantized matrix. Rows are contiguous with `18 * (in_dim / 32)` bytes; partial blocks are rejected.

`ds4_gpu_matmul_q8_0_tensor` is reused for Q8_0 rather than adding another copy of DS4's existing Metal GEMV. `ds4_gpu_inkling2_expert_accumulate_tensor` combines up to eight already-computed expert output vectors using routed weights; it does not dequantize or combine expert matrices. At the M3 checkpoint, gate/up SiTU, down projection, and expert-layer orchestration remained unimplemented; their isolated implementation is recorded under M5 below.

The extended `make test-inkling2-metal` fixture checks 96-element inputs and five output rows, multiple block scales, low/high nibble order, non-four-byte row strides (54-byte IQ4_NL rows and 102-byte Q8_0 rows), and rejection of partial blocks. Its scalar C oracle passes on the Apple M4. No model bytes are used.

The standalone `make inkling2-metal-quant-bench` was run on synthetic 1152 × 2304 matrices with three warmups and three serial runs of ten synchronous calls each. IQ4_NL took 537.6, 574.7, and 596.7 µs/op; reused Q8_0 took 331.5, 294.7, and 278.8 µs/op. These timings include DS4 dispatch and synchronization, use synthetic weights, and are not model throughput results.

After M3, `./ds4_test --metal-kernels` still reports the same 29 assertion records as M0; comparison found no new or resolved records. The model is absent, so no real checkpoint block, MoE output, full-model memory, or text-generation parity has been measured.

## M4 isolated attention, convolution, and visit state

`ds4_inkling_state.h` and `ds4_inkling_state.c` allocate one independent state slot for each planner cache id. The session derives each slot's type from the visit plan: sliding slots keep a 1024-row ring and full-attention slots keep the configured context capacity (8192 for the initial model path). Keys and values are stored as raw FP16 bits in `[position % capacity][kv_head][head_dim]` order. The session also owns separate length-three histories for key, value, attention output, and FFN output for every visit.

The state API rejects non-sequential appends and malformed visit maps. A transaction backs up only KV ring rows touched by its writes and snapshots the four short-convolution histories; rollback restores overwritten ring contents and positions, while commit retains the update. Deep clone copies the full cache and convolution histories into an independently allocated compatible slot, which supports reuse of a completed prefix state. `ds4_inkling_state_conv_export_history` converts the internal rolling history to the chronological, zero-padded layout consumed by the Metal short-convolution kernel.

The scalar C operators provide the comparison path for per-head learned scaling, Q/K L2 clipping, optional full-attention log scaling, projected relative-distance bias, causal attention with GQA and a zero-value attention sink, and static or input-dependent short convolution. The dynamic tap layout follows the reference: lag zero is the current token, static weights are `[kernel][channel]`, and dynamic weights are `[token][head][kernel]`. Sliding decode uses the local window; full decode walks every retained absolute position, including positions beyond 1024.

`metal/inkling2.metal` and `ds4_gpu_inkling2_*_tensor` add matching GPU operators for Q/K preparation, relative-bias projection, sliding/full decode attention, sigmoid attention output gating, and static/dynamic causal short convolution. Q/K preparation and attention keep FP32 accumulations; K/V remain FP16. Decode attention assigns a 128-lane threadgroup to each query head, reduces Q·K over head dimensions, and uses an online softmax so it does not materialize a score row. Causal shortconv currently assigns one lane to each channel and walks the token chunk serially.

`make test-inkling2-state` and `make test-inkling2-metal` pass on Apple M4. The state test checks independent pass-0/pass-1/pass-2 cache slots for the same physical layer, rejects malformed visit plans, checks local ring wrap, full-cache retention, commit/rollback including overwritten rows, reset, deep prefix clone, chronological convolution history, chunked versus monolithic static/dynamic convolution, Q/K scale-clip-log order, relative-distance masking, and full attention through position 1024. The Metal test compares Q/K preparation, relative bias, attention with sink and a wrapped sliding cache, full attention over 1025 keys, sigmoid output gating edge values, and static/dynamic convolution to scalar C references. GPU convolution chunk calls of lengths 2, 1, and 4 match one seven-token call, including nonzero stale history supplied with a zero history count.

The M4 Metal test reported maximum absolute errors of `0` for QK preparation/relative bias, `2.3841858e-07` for sliding attention with relative bias and sink, `0` for 1025-key full attention, `1.1350518e-43` for the tested attention-gate values, and `0` for dynamic convolution, chunked convolution delta, and static convolution. These are synthetic operator fixtures, not full-model parity.

The synthetic `make inkling2-metal-m4-bench` benchmark was run before and after changing attention from one serial lane per query head to the 128-lane online-softmax kernel. It uses 24 query heads, 8 KV heads × 128 dimensions, 2 warmups, and three serial runs of three synchronous operations per row on Apple M4. Keys and values are synthetic; these are operator timings with dispatch/synchronization, not model throughput.

| Decode cache | Before: ms/op, three runs | After: ms/op, three runs |
| --- | --- | --- |
| SWA, 1024 retained rows | 29.162 / 29.354 / 29.299 | 3.161 / 2.789 / 2.608 |
| Full attention, 8192 rows | 228.665 / 235.979 / 230.192 | 13.922 / 13.664 / 13.926 |

The optimized kernel passed the same scalar-oracle test suite before the after-run benchmark. This result only characterizes the attention operator and does not predict end-to-end Inkling tokens per second.

This milestone is still partial: DS4 does not yet schedule these operators as an Inkling decoder layer. Batched causal attention prefill, Q/K/V and gate/sink/relative projections, post-attention/post-FFN graph wiring, expert FFN, loop controller, tokenizer/template path, and same-GGUF parity remain open. No GGUF was present, so no real-model output or model throughput claim is made.

## M5 isolated routed expert FFN

`ds4_inkling_moe_situ_metal` composes a one-token routed FFN from DS4 Metal tensors and the existing F32 matvec path. Router logits and correction-biased top-8 selection stay on the GPU. Selected expert IDs index the shared physical matrices directly; gate/up projection, SiTU activation, per-expert down projection, and routed output accumulation never materialize a dense expert matrix or copy routing results to the CPU. IQ4_NL and Q8_0 are supported for expert tensors. The active DS4 Metal mapped views must cover the router and expert tensor ranges.

The SiTU kernel implements `beta * tanh(gate / beta) * sigmoid(gate) * linear_beta * tanh(up / linear_beta)`. Metal's finite large-input `tanh` behavior on M4 produced NaNs, so its implementation saturates inputs at ±10, where the FP32 result is within rounding of ±1. Routing still uses the un-biased sigmoid scores for normalized route weights after correction bias determines the selected experts.

`make test-inkling2-metal` now builds a synthetic 128-expert fixture with 8 selected experts and compares the full routed FFN against scalar references for both IQ4_NL and Q8_0, as well as checking routing, selected projections, SiTU, and down projection independently. It passes on Apple M4. This verifies the composed operator but does not bind it to DS4 model metadata or a decoder layer.

`make inkling2-metal-moe-bench` was run on the M4 with 2304 hidden width, 1152 expert width, 8 of 128 experts, and 546.8 MiB of synthetic IQ4_NL expert weights. Two warmups were followed by three runs of three synchronous calls. The measured routed-FFN latency was 6.355, 4.398, and 4.564 ms/op. The first run includes additional cold-page and pipeline effects; the latter two are 4.398 and 4.564 ms/op. These measurements include router, dispatch, synchronization, projections, SiTU, and accumulation. They are synthetic operator timing, not model tokens/sec.

M5 remains partial: native model loading/weight binding, attention and FFN residual wiring, 48-visit loop execution, prefill, tokenizer/template/EOS behavior, text generation, and same-GGUF numerical parity are open. The GGUF is absent, so real-model output and performance have not been measured.
