# Inkling 2 Mini on Apple Metal

This document tracks native DS4 support for the text-only Inkling 2 Mini GGUF. The first supported model format is the mixed IQ4_NL GGUF, with an 8192-token context and FP16 key/value state. DS4 loads the GGUF directly through its mmap loader and executes operators in its native C and Metal paths; it does not use llama.cpp or MLX as a runtime dependency.

## Handoff gate status

The numbered section headings later in this document preserve the incremental implementation slices used by the commit history. This table maps the current code to the canonical gates in the supplied handoff; an isolated operator or loader test does not count as end-to-end model support.

| Handoff gate | Status | Evidence |
| --- | --- | --- |
| M0: repository, references, and host baseline | Partial | Fork/base, M4 host, build and server results are recorded. Same-GGUF llama.cpp timing is unavailable because the checkpoint is not local. |
| M1: GGUF inspector, tensor table, and visit map | Complete for no-weight validation | Inspector/planner tests pass; the exact remote directory range was checked without reading tensor payloads; sparse fixtures validate binding and malformed descriptors. |
| M2: Metal plumbing and scalar-oracle operators | Complete for isolated control operators | Runtime Metal integration, RMSNorm, clipping, deterministic top-k routing, and logit softcap have focused M4 oracle coverage. The 29 pre-existing Metal-suite failures are unchanged from baseline. |
| M3: IQ4_NL/Q8_0 quantized products and expert routing | Complete for synthetic fixtures | Quantized GEMV and composed top-8 expert FFN fixtures pass against scalar references. No real checkpoint blocks were available for comparison. |
| M4: attention, convolution, and per-visit state | Partial | Decode attention, convolution, state transactions, and independent cache slots pass synthetic tests; architecture-level prefill/decode scheduling is not wired. |
| M5: loop core, output layer, and text generation | Not complete | Tensor binding, visit plan, loop operators, tokenizer, and prompt serialization are available as pieces; no native model graph, logits, or generated text is integrated. |
| M6: full-model M4 performance | Not run | Synthetic operator timings exist, but full-model profiling and tokens/sec require the checkpoint and a correct M5 graph. |
| M7: integration hardening and user workflow | Partial | Builds, focused suites, server regressions, GGUF validation, and this documentation pass. A model-backed completion demo and same-weight benchmark are unavailable. |
| M8: post-release formats and optional features | Not started | NVFP4/MXFP8, low-precision KV, vision/audio, and speculative decoding remain out of scope for this IQ4_NL path. |

## M0 baseline

- Repository: `giacomobartoloni/ds4`, branch `feature/inkling2-mini-metal-m4`, based on `b3f50180ec2c7324892be1f50bc83555ccb8d5e7`.
- At baseline, fork `main` and upstream `antirez/ds4` both pointed to `b3f50180ec2c7324892be1f50bc83555ccb8d5e7`; the checkout had no configured upstream remote.
- Host class: MacBook Pro with Apple M4, 32 GB unified memory, macOS 26.2. Metal reported a recommended maximum working set of 24.96 GiB.
- `make clean && make`: PASS.
- `make cpu`: PASS.
- `./ds4_test --server`: PASS (`server: OK`, `ds4 tests: ok`).
- `./ds4_test --metal-kernels`: FAIL on the unmodified baseline with 29 assertions. Four compressor pair state-store cases fail at `tests/ds4_test.c:1678` and the associated output comparisons at lines 1754–1758. Four gathered KV staging cases fail kernel selection at line 3315; raw and compressed output comparisons fail at lines 3365–3366. The failure log is preserved locally as `/tmp/ds4-inkling2-metal-baseline.log`; this is baseline evidence, not a result from Inkling code.
- Metal runtime source compilation and device initialization were reached on the M4 during the kernel suite. The standalone Xcode build tool and `metal` compiler are unavailable, so offline shader compilation is not available.
- The requested GGUF is not present locally. On 2026-11-23 the pinned remote file's directory-only byte range (`0..11,977,631`, exactly the GGUF `data_offset`) was parsed to validate metadata and descriptors; the exact range contained no tensor payload. A preliminary broader probe crossed the data offset by about 605 KB before the exact range was established; those bytes were transient in process memory and were not saved. No model file was saved. `make test` (which requires `ds4flash.gguf`), local model inference, numerical comparison against a reference runtime, and throughput/memory benchmarks have not been run.

## Pinned implementation references

These source revisions were inspected as format and semantic references; DS4 does not call them at runtime.

- llama.cpp: [`212c547fe908b288f6659d48741e6ffac1ff409c`](https://github.com/ggml-org/llama.cpp/tree/212c547fe908b288f6659d48741e6ffac1ff409c), including [`src/models/inkling2.cpp`](https://github.com/ggml-org/llama.cpp/blob/212c547fe908b288f6659d48741e6ffac1ff409c/src/models/inkling2.cpp) and [`src/models/inkling2-loop.cpp`](https://github.com/ggml-org/llama.cpp/blob/212c547fe908b288f6659d48741e6ffac1ff409c/src/models/inkling2-loop.cpp).
- Hugging Face Transformers: [`d9bdc606d543e1a85c0d7121af9379fbd2445bf3`](https://github.com/huggingface/transformers/tree/d9bdc606d543e1a85c0d7121af9379fbd2445bf3).
- Inkling 2 Mini model repository revision: [`7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6); official text chat template: [`chat_template.jinja` at `1fb4134b9a74f9681054f4ef82e4bf1f458a2337`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/1fb4134b9a74f9681054f4ef82e4bf1f458a2337/chat_template.jinja).
- IQ4_NL GGUF repository revision: [`e8a2cecf0e78914f07cc27b9be711a9b2c7561d9`](https://huggingface.co/HuggingFaceTB/Inkling-2-Mini-GGUF/tree/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9).

## Initial execution contract

- 24 physical layers, with loop core `[6,18)` executed in three passes. The resulting 48 logical visits reuse those 24 physical weight owners and each have an independent cache slot.
- The pinned model's 24-entry sliding-window pattern identifies physical full-attention layers 5, 11, 17, and 23. Across the 48 visits this yields 8 full-attention and 40 sliding-window cache slots.
- Initial inference scope: one text-only session, batch size 1, context 8192, IQ4_NL expert matrices, Q8_0 other matrices, F32 loop controller tensors, and FP16 KV. Unsupported model metadata must be rejected rather than silently assumed.
- A benchmark or numerical-parity claim requires the model file to be present and the exact tested file, prompt, context, and run count to be recorded.

## GGUF inspector

`gguf-tools/inkling2_inspect.py` reads GGUF v3 metadata and tensor descriptors. It validates tensor shapes, canonical byte sizes for the supported F32/F16/Q8_0/IQ4_NL tensor types, alignment, overlap, and file bounds without reading tensor payloads. Use `--summary`, `--tensors`, `--json`, or `--validate-inkling2`; all modes report that payload bytes were not read. The Inkling validator checks the loop metadata, 24-entry attention schedule, 48 logical visits, 156 F32 loop-controller tensors, and the IQ4_NL/Q8_0 expert tensor assignments.

Synthetic coverage: `make test-inkling2-inspect` passes 18 tests for valid descriptors, primitive/array metadata, truncation and size limits, duplicate keys and names, invalid shapes and types, offset alignment/overlap/bounds, quant block sizes, the expected Inkling layout, and payload-free inspection. A pinned remote range parse confirmed `data_offset=11,977,632`, 704 tensors, and counts F32=478/Q8_0=186/IQ4_NL=40; no payload bytes were part of that exact range and no local GGUF is present.

## 48-visit plan and loader block sizes

`ds4_inkling.h` validates the supported 24-layer metadata and emits the execution order: prelude layers `[0,6)`, core `[6,18)` for three passes, then epilogue `[18,24)`. Each plan entry stores a physical layer index as its shared weight owner and a unique cache slot. Pass 0 uses cache slots 6–17, pass 1 uses 24–35, and pass 2 uses 36–47; prelude and epilogue use their physical indices. The planner validates the supplied attention pattern and rejects changed boundaries, pass counts, residual block size, or layer schedule. It describes the cache ownership needed by the runtime; inference buffers are not yet connected to it.

`ds4_gguf.h` is the shared type-size table used by the mmap loader and the C test. It corrects GGML type 19 IQ1_S to 256 elements / 50 bytes and type 20 IQ4_NL to 32 elements / 18 bytes. The canonical IQ4_NL layout is a 2-byte scale plus 16 packed bytes (`QK4_NL=32`) in the pinned llama.cpp source.

## M7 native metadata validation and tensor binding

`config_validate_inkling2_model` accepts only the pinned text model contract: 24 physical layers, hidden 2304, padded/unpadded vocabulary 201024/200058, 24 query heads and 8 KV heads per layer, loop `[6,18)` for three passes, rank 64, six-layer residual blocks, 1024 sliding window, four-position short convolution, zero rotary dimension, and the recorded numeric parameters. It also checks Inkling's `gpt2` tokenizer metadata, EOS/padding IDs, and no implicit BOS/EOS.

`inkling_weights_bind` requires all 704 descriptors and validates every bound tensor's name, rank, dimensions, and type. The engine stores one array of 24 physical layer weights. Its loop plan contains 48 references to those physical owners with 48 unique cache IDs; it does not copy weights for repeated visits. Full-attention layers bind their Q/K/V norms and Q/K convolution projections, while sliding layers bind their single attention norm. Expert matrices use IQ4_NL in layers 0–19 and Q8_0 in layers 20–23.

`make test-inkling2-gguf` creates a sparse, payload-free 17.6 GB GGUF fixture from the live descriptor inventory, exercises the real mmap parser and metadata validator, checks the complete binder and engine inspect-only path, and verifies rejection of a missing descriptor, wrong tensor type, and wrong shape. This allocates only a sparse file and does not write model payloads. `make test-inkling2-model` checks valid and unsupported architecture metadata independently.

The normal engine-open path currently rejects Inkling after successful binding and plan creation with an explicit “native decoder is not integrated yet” error. It must not fall through to the DeepSeek generic tensor layout. Native prefill/decode scheduling, per-session 48-slot state wiring, greedy generation, and same-weight parity remain open.

After M7, `make` and `make cpu` pass, and `./ds4_test --server` passes on the M4 host. `make test-inkling2-model test-inkling2-gguf test-inkling2-mapping test-inkling2-inspect test-inkling2-state test-inkling2-metal` passes. `./ds4_test --metal-kernels` still reports 29 failures; the extracted assertion records are byte-for-byte identical to `/tmp/ds4-inkling2-metal-baseline.log` from M0, with no new or resolved records.

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

## M6 isolated loop-controller operators

`ds4_inkling_loop.c` adds scalar references for learned residual-state mixing and the low-rank loop controller. Mixing RMS-normalizes each candidate state with the model epsilon, scores each normalized state against its learned query at `1 / sqrt(hidden_dim)`, applies a stable softmax, and returns the weighted normalized-state sum. The controller RMS-normalizes current state and message separately, concatenates them, reuses DS4's F32 matmul primitive for the learned down/gain/forget projections, applies `tanh` to gain and forget vectors, and updates the residual state.

`metal/inkling2.metal` adds a cooperative query/state mix kernel, controller input normalization, and the gated update. `ds4_inkling_loop.c` composes these with DS4's existing F32 GPU matmul for the three controller projections. The implementation uses the pinned llama.cpp Inkling 2 graph semantics and does not add an external inference runtime.

`make test-inkling2-metal` compares the Metal path with scalar C at the target model dimensions: hidden 2304, controller rank 64, five candidate states, and four learned queries. It passes on Apple M4; maximum absolute errors were `4.02331352e-07` for residual mixing and `5.36441803e-07` for the controller update. The test uses synthetic inputs and controller matrices and covers multi-lane reductions above 256 elements.

M6 is still partial: these are independent operators, not a wired 48-visit decoder. The GGUF loader, per-layer tensor binding, pass-start prelude restore, controller message banks, causal prefill, tokenizer/template/EOS behavior, end-to-end generation, and same-GGUF parity remain open. No model weights were present, so these checks have not been claimed.

## Text tokenizer and chat prompt groundwork (M5 prerequisite)

Inkling uses the GPT4O BPE pre-tokenizer with `ignore_merges=true` and no implicit BOS. The C path now performs the GPT4O letter-case splits, contraction attachment, one-to-three digit groups, optional leading-space handling, punctuation/newline/slash grouping, and direct vocabulary lookup before falling back to the existing ranked BPE merges. Its special-token table binds the required role, text, thinking, end-message, and end-sampling IDs and rejects a GGUF whose core IDs differ from the pinned metadata. The pre-tokenizer behavior is checked against llama.cpp's pinned GPT4O expression; no external tokenizer runtime is linked.

The text-only prompt APIs render the model template's role markers and content blocks. Initial system messages come first, then the reasoning-effort system block is inserted before the first non-system message. Named values map to `none=0.0`, `minimal=0.1`, `low=0.2`, `medium=0.7`, `high=0.9`, and `xhigh=max=0.99`; decimal strings in `[0.0, 0.99]` are accepted through the shared reasoning-mode parser. Assistant history ends with both `end_message` and `content_model_end_sampling`; an open generation prompt ends with `message_model`. The CLI's one-shot and REPL prompt builders preserve this order. Tool schemas, image/audio payloads, and structured assistant tool-call generation are not implemented by this text-only path.

`make test-inkling2-tokenizer` checks GPT4O splitting for mixed/all-uppercase words, ASCII case boundaries around Unicode letters (`AéB`), contractions, Unicode letters, number groups, punctuation spanning newline/slash, direct-word lookup, ranked-merge fallback, exact effort strings, numeric-effort de-duplication, rendered special-token recognition, text chat message order, and the generation prefix. The `AéB` regression follows the backtracking behavior in the pinned llama.cpp GPT4O expression. The server regression suite additionally checks `minimal`, `xhigh`, and decimal reasoning levels. `make`, `make cpu`, the Inkling focused targets, and `./ds4_test --server` pass after this change on the Apple M4.

This milestone does not enable inference. The engine still refuses normal Inkling generation until the model graph connects embeddings, visit-specific KV/shortconv state, attention, routed experts, loop message banks, output projection, and sampling. The 17.6 GB model is not local, so no token stream, full-model numeric comparison, or end-to-end benchmark has been measured.
