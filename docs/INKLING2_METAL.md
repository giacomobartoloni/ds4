# Inkling 2 Mini on Apple Metal

This document tracks native DS4 support for the text-only Inkling 2 Mini GGUF. The first supported model format is the mixed IQ4_NL GGUF, with FP16 key/value state. Context length is selected per session, bounded by the GGUF metadata and Metal working-set budget. DS4 loads the GGUF directly through its mmap loader and executes operators in its native C and Metal paths; it does not use llama.cpp or MLX as a runtime dependency.

## Handoff gate status

The numbered sections preserve implementation snapshots from each milestone. Status statements inside those snapshots describe that milestone; the gate table and current model-backed checkpoint below are authoritative for the present branch. An isolated operator or loader test does not count as end-to-end model support.

| Handoff gate | Status | Evidence |
| --- | --- | --- |
| M0: repository, references, and host baseline | Complete for host and model inventory | The M4/32 GB host, exact GGUF hash, and pinned llama.cpp reference are recorded below. A stable performance baseline is still open. |
| M1: GGUF inspector, tensor table, and visit map | Complete for no-weight validation | Inspector/planner tests pass; the exact remote directory range was checked without reading tensor payloads; sparse fixtures validate binding and malformed descriptors. |
| M2: Metal plumbing and scalar-oracle operators | Complete for isolated control operators | Runtime Metal integration, RMSNorm, clipping, deterministic top-k routing, and logit softcap have focused M4 oracle coverage. The 29 pre-existing Metal-suite failures are unchanged from baseline. |
| M3: IQ4_NL/Q8_0 quantized products and expert routing | Complete for operators; model checked | Quantized GEMV and top-8 expert FFN tests pass against scalar references. Live model Q/K/V projections and final logits were compared with the same GGUF in pinned llama.cpp. |
| M4: attention, convolution, and per-visit state | Complete for single-session token execution | The native 48-visit session path decodes and prefills incrementally with independent KV/convolution state. The real model's first-layer intermediates and multi-token cache continuation were checked. |
| M5: loop core, output layer, and text generation | Operational for batch-1 text; validation partial | Normal engine/session generation emits real text. The first eight greedy token IDs match the pinned reference; long-context, EOS, and speculative rollback cases remain open. |
| M6: full-model M4 performance | Not complete | An encoder timeline now identifies expert GEMV as the main decode cost, and the SIMD reduction candidate is numerically checked. The timeline serializes encoder passes and the synthetic GEMV benchmark is variable, so stable throughput and resident-memory peak remain unmeasured. |
| M7: integration hardening and user workflow | Partial | OpenAI Chat tool round-trip and streamed Responses tool continuation ran on the real GGUF. Long-context and model-backed Anthropic validation remain open. |
| M8: post-release formats and optional features | Not started | NVFP4/MXFP8, low-precision KV, vision/audio, and speculative decoding remain out of scope for this IQ4_NL path. |

## Current model-backed checkpoint — 2026-11-24

The externally stored checkpoint is `/Users/gbartoloni/Models/Inkling-2-Mini/Inkling-2-Mini-IQ4_NL.gguf`, SHA-256 `cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4`. Its 704 tensor descriptors match the validated binder. No weights were downloaded or added to the repository. Numeric comparisons use llama.cpp commit [`212c547fe908b288f6659d48741e6ffac1ff409c`](https://github.com/ggml-org/llama.cpp/tree/212c547fe908b288f6659d48741e6ffac1ff409c), built locally with Metal; the reference is not a DS4 runtime dependency.

For raw prompt `Hello` (token ID 32269), context 8192, Metal, and greedy temperature 0, the SIMD-reduction build's untraced logits over all 200,058 vocabulary entries have the same argmax (token 11, `,`) and identical top 10 as the pinned reference. Measured errors are max absolute `9.6655e-4`, mean absolute `1.7648e-4`, RMSE `2.2101e-4`, reference RMS `7.59055`, and cosine similarity `0.9999999996`; the maximum occurs at token 131786. These are observed values, not an arbitrary acceptance tolerance. Relative to the saved pre-SIMD DS4 logits, the candidate differs by max absolute `2.66284e-4` and RMSE `1.61698e-5`, with the same argmax and top 10. This small change comes from floating-point accumulation order.

An intermediate trace isolated the first real divergence to static short-convolution tap order: llama.cpp indexes static taps oldest-to-current, while dynamic taps use current-to-oldest lag order. DS4 had used the dynamic order for both. The CPU reference and Metal kernel now reverse only the static tap index. Before that correction, the independent formula test failed at element 0 (`-0.207792` vs `-0.350649`); afterward it passes. The relative-bias fixture also checks the GGML `{relative_dim, extent}` layout (one contiguous vector per distance) and zero output for out-of-range buckets. On the first decoder layer, Q/K/V matched the pinned graph exactly, relative state differed by at most `1.43e-6`, and layer output differed by at most `1.51e-4` (NRMSE `2.08e-5`).

The normal CLI generated the same first eight greedy IDs as llama.cpp: `11, 68014, 40, 150087, 153221, 149882, 150054, 149876`, rendered as `,\n\nI am trying to use the`. `--decode-consistency 8` then compared the live decoded session with a fresh prefill of the resulting nine-token prefix: maximum logit error and RMS error were both exactly zero. Before the SIMD reduction, two 32-token runs reported 6.64 and 6.50 decode tokens/s; the current build reported 7.38 tokens/s on one 32-token run (prefill 4.12 tokens/s). These are smoke measurements, not a controlled throughput comparison. Model residency requests varied from 0.55 to 11.70 seconds, so they are not TTFT measurements.

The expert GEMV now uses an in-SIMD reduction followed by one threadgroup barrier and a reduction of SIMD-group partials. The host passes `threadExecutionWidth` from the selected pipeline. On the same raw `Hello` trace, this reduced the instrumented expert-GEMV sum from `240.107 ms` across two baseline evaluations (`192` encoders) to `109.146 ms` for one candidate evaluation (`96` encoders), about 9% less per evaluation. The total serialized command-buffer time was `152.116 ms` for the candidate versus a `161.931 ms` baseline average. The diagnostic intentionally adds pass boundaries and serializes work; these timings locate hotspots and are not direct model throughput. A workspace-reusing synthetic FFN benchmark now queues three calls per command buffer with one sync per run. Its current warmed medians varied from `3.08` to `3.35 ms/op`, so the synthetic measurement does not establish a stable speedup.

The eight focused Inkling targets (`state`, `inspect`, `model`, `mapping`, `metal`, `gguf`, `session`, and `tokenizer`) pass on this M4 build; the focused Metal test also passes after the SIMD reduction. `make ds4 cpu ds4_test`, a forced native Metal `make -B ds4`, and `./ds4_test --server` pass. A non-Apple-macro syntax check of `ds4.c` passes after guarding the Metal-only session field. The latest `./ds4_test --metal-kernels` run exits non-zero with exactly the same 29 assertion records as the captured M0 baseline; the assertion lists compare equal. The context allocator reports 454,785,284 bytes (433.72 MiB) at context 8192. A 500 ms `footprint` sample during a 32-token run reported process `phys_footprint_peak=585,418,792` bytes (558.3 MiB); this is not a complete Metal/device working-set peak measurement. The CLI's generic `KV 0.06 GiB` estimate is a different model-family metric and does not include this Inkling session allocation. The repository's default `ds4flash.gguf` is absent, so the full existing-model and SSD-streaming regression paths were not run; no CUDA compiler/device is available on this Mac. Inkling explicitly rejects SSD streaming and non-Metal execution.

The local GGUF declares `inkling2_loop.context_length=1,048,576`, matching the 1M-token maximum in the official [Inkling 2 Mini model card at the pinned revision](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6/README.md). That is a model limit, not a promise that a local device can allocate the corresponding cache. The M4 reports a recommended Metal working set of 26,800.60 MB (24.96 GiB); with this 17,627,940,768-byte GGUF and DS4's 2 GiB runtime reserve, the allocator estimates about 4.17 GiB of session state at context 131,072, leaving about 2.54 GB below the recommended working set. Context 200,000 estimates 6.28 GiB of state and leaves about 285 MB; treat it as near the device budget. Context 1,048,576 would need about 32.17 GiB for session state alone and cannot fit alongside the model. These are allocation estimates, not measured resident peaks. The runtime checks the actual Metal recommendation before allocating a session.

At the initial context 8192, the session allocator estimates 454,785,284 bytes (433.72 MiB), while DS4 reports 16,811.33 MiB mapped for the model. Peak resident memory was not measured. The current prefill path evaluates visit-major groups of up to eight rows: it batches Q/K/V and related projections, attention sink/gate/output projections, and routed MoE, while preserving causal per-row attention and convolution state. It computes and reads logits only at each group boundary. Long-context numerical parity, stable prompt-throughput measurements, peak resident memory, real EOS completion, and speculative rollback remain open; grouped/token-wise logits parity and interruption/resume are covered by the synthetic session test.

With the real GGUF on the M4, a session configured at `--ctx 131072` passed the
Metal budget check and generated `,\n\nI am trying to use the` from the
one-token raw prompt `Hello`. This verifies larger session capacity and normal
decode at that configuration; the prompt itself did not exceed 8192 tokens, so
it is not long-context numerical validation or a throughput benchmark.

## Visit-major grouped prefill — 2026-11-24

Each group processes up to eight prompt rows across the 48 logical visits.
The group keeps separate activations and residual banks per row, and advances
each visit's KV and convolution state in ascending token order. Q/K/V,
relative-state and dynamic-convolution projections, sink/gate projections,
attention-output projection, and top-8 MoE use the multi-row Metal path. The
attention and causal convolutions still execute row by row, preserving their
existing single-token kernels and cache semantics. Only the final group row
runs the vocabulary head and becomes the published checkpoint/logits. Progress
callbacks fire once per group, so a full group advances the server counter by
eight.

`make test-inkling2-metal` passes the three-row IQ4_NL/Q8_0 routed-MoE oracle
tests, including deterministic route IDs/weights and the no-correction-bias
case. `make test-inkling2-session` passes byte-identical grouped-vs-token-wise
synthetic logits, interruption after the first eight-row group, checkpoint
reuse, and resume. These session fixtures use synthetic weights and do not
establish model-backed logit equality.

On the M4 with the exact GGUF, context 8192, temperature 0, and short raw
prompts of 82 and 165 tokens, one comparison per length measured 6.57 vs
6.35 tokens/s and 6.46 vs 6.30 tokens/s respectively (grouped vs token-wise
baseline). For the 82-token prompt, eight generated tokens matched exactly as
text (`Question\n\nExplain how an Apple `); the 165-token prompt's sampled
output also matched (`Summar`). A separate short decode sample measured 6.49
vs 6.78 tokens/s, so this does not show a decode speedup. The prompt samples
suggest only a 2.5–3.5% prefill gain and are too short and few to establish a
stable rate. No full-model numeric logit comparison was repeated after batching.

The supplied OpenCode server log advances progress one token at a time. The
current source advances the callback by eight for each full group. No server
process was running when checked, so the executable that produced the supplied
log could not be identified; `ds4-server` has since been rebuilt from this
worktree. The 48,891-token prompt itself was not rerun. At the observed 6.4
tokens/s it takes about 2 h 7 min to prefill; the modest grouped-prefill gain
does not change that estimate materially. The next optimization target is the
remaining per-row attention and convolution dispatch work, measured against
repeatable real-prompt samples.

## Conversational API and function tools — 2026-11-23

The native server recognizes the `inkling-2-mini` model ID and selects the
Inkling GGUF chat template, parser, and SSE projection. OpenAI Chat Completions,
Responses, and Anthropic share the normal DS4 server API; the server returns
function calls for the client or harness to execute and does not execute tools
itself. The Responses endpoint accepts `tool_choice: "auto"` and `"none"`; it
rejects forced/required choices. Chat Completions accepts
`tool_choice: "required"`.

On the M4 with the GGUF above, an OpenAI Chat request with
`tool_choice: "required"` generated `weather({"city":"Rome"})`. Replaying that
assistant call with a harness-supplied tool result produced the plain final
answer “The weather in Rome is sunny, with a temperature of 18°C.” The same
cycle passed through streamed Responses: it emitted a `function_call`, then
accepted a follow-up containing `previous_response_id` and only the matching
`function_call_output`, and emitted the final text. Responses usage reported
165 cached input tokens out of 189 on that continuation. The weather result was
test input supplied by the caller, not a live weather lookup.

Responses `additional_tools` namespaces and custom tool input now pass through
the Inkling schema and SSE serializers. On the real GGUF, a streamed request
with the Codex-style `functions.exec` custom tool returned a
`custom_tool_call` with JavaScript input and the expected delta/done events. A
full-history follow-up with the caller-supplied tool result returned `TOOL_OK`.
The generated JavaScript was not executed by this test. Codex CLI 0.171.0 did
not complete with its full agent/tool prompt at context 8192. Larger session
contexts are accepted up to the GGUF limit and available Metal memory, but
Codex CLI has not yet been revalidated above 8192.

Inkling remains text-only, batch size 1, and one resident server session; do not
use `--batched-session`. Set `--ctx` within the GGUF limit and device memory
budget. Responses IDs are not stored durably.
Output-only tool continuation is bound by a `call_id` in the live server state;
after a process restart, replay the complete input history instead. The
OpenAI, Responses, and Anthropic Inkling SSE adapters and malformed-call
fail-closed behavior pass `./ds4_test --server`. The Anthropic path has unit
coverage but was not exercised against the real checkpoint in this run.

## M0 baseline

- Repository: `giacomobartoloni/ds4`, branch `feature/inkling2-mini-metal-m4`, based on `b3f50180ec2c7324892be1f50bc83555ccb8d5e7`.
- At baseline, fork `main` and upstream `antirez/ds4` both pointed to `b3f50180ec2c7324892be1f50bc83555ccb8d5e7`; the checkout had no configured upstream remote.
- Host class: MacBook Pro with Apple M4, 32 GB unified memory, macOS 26.2. Metal reported a recommended maximum working set of 24.96 GiB.
- `make clean && make`: PASS.
- `make cpu`: PASS.
- `./ds4_test --server`: PASS (`server: OK`, `ds4 tests: ok`).
- `./ds4_test --metal-kernels`: FAIL on the unmodified baseline with 29 assertions. Four compressor pair state-store cases fail at `tests/ds4_test.c:1678` and the associated output comparisons at lines 1754–1758. Four gathered KV staging cases fail kernel selection at line 3315; raw and compressed output comparisons fail at lines 3365–3366. The failure log is preserved locally as `/tmp/ds4-inkling2-metal-baseline.log`; this is baseline evidence, not a result from Inkling code.
- Metal runtime source compilation and device initialization were reached on the M4 during the kernel suite. The standalone Xcode build tool and `metal` compiler are unavailable, so offline shader compilation is not available.
- At the initial M0 checkpoint, the GGUF was not present locally. The directory-only range (`0..11,977,631`, exactly the GGUF `data_offset`) was parsed without tensor payloads; a broader preliminary probe crossed the offset by about 605 KB and was not saved. The model was later supplied at the external path and validated as recorded in the current checkpoint above. The generic `make test` target, which requires `ds4flash.gguf`, remains distinct from the focused Inkling checks.

## Pinned implementation references

These source revisions were inspected as format and semantic references; DS4 does not call them at runtime.

- llama.cpp: [`212c547fe908b288f6659d48741e6ffac1ff409c`](https://github.com/ggml-org/llama.cpp/tree/212c547fe908b288f6659d48741e6ffac1ff409c), including [`src/models/inkling2.cpp`](https://github.com/ggml-org/llama.cpp/blob/212c547fe908b288f6659d48741e6ffac1ff409c/src/models/inkling2.cpp) and [`src/models/inkling2-loop.cpp`](https://github.com/ggml-org/llama.cpp/blob/212c547fe908b288f6659d48741e6ffac1ff409c/src/models/inkling2-loop.cpp).
- Hugging Face Transformers: [`d9bdc606d543e1a85c0d7121af9379fbd2445bf3`](https://github.com/huggingface/transformers/tree/d9bdc606d543e1a85c0d7121af9379fbd2445bf3).
- Inkling 2 Mini model repository revision: [`7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6); official text chat template: [`chat_template.jinja` at `1fb4134b9a74f9681054f4ef82e4bf1f458a2337`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/1fb4134b9a74f9681054f4ef82e4bf1f458a2337/chat_template.jinja).
- IQ4_NL GGUF repository revision: [`e8a2cecf0e78914f07cc27b9be711a9b2c7561d9`](https://huggingface.co/HuggingFaceTB/Inkling-2-Mini-GGUF/tree/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9).

## Initial execution contract

- 24 physical layers, with loop core `[6,18)` executed in three passes. The resulting 48 logical visits reuse those 24 physical weight owners and each have an independent cache slot.
- The pinned model's 24-entry sliding-window pattern identifies physical full-attention layers 5, 11, 17, and 23. Across the 48 visits this yields 8 full-attention and 40 sliding-window cache slots.
- Initial inference scope: one text-only session, batch size 1, context 8192, IQ4_NL expert matrices, Q8_0 other matrices, F32 loop controller tensors, and FP16 KV. Context can now be selected per session up to the GGUF limit, subject to the Metal working-set check. Unsupported model metadata must be rejected rather than silently assumed.
- A benchmark or numerical-parity claim requires the model file to be present and the exact tested file, prompt, context, and run count to be recorded.

## Decoder integration inventory and GPU session state

The pinned llama.cpp layer order is the authority for this inventory. This table preserves the gaps identified before decoder wiring; its “Missing integration” column is historical. The normal DS4 session now connects these primitives into the batch-1 decoder and three-pass loop; the current checkpoint above records model-backed evidence and remaining validation limits.

| Decoder operation | Existing DS4 Metal pieces | Missing integration | Model tensors and persistent state | Temporary memory and synchronization |
| --- | --- | --- | --- | --- |
| Input embedding and norm | Q8_0 token embedding lookup; weighted RMSNorm | Seed the per-session current activation and carry token position through the graph | `token_embd.weight`, `token_embd_norm.weight`; one token ID and 2304-wide embedding | Input ID is a 4-byte GPU buffer; embedding stays in the session arena. No layer dispatch path yet. |
| Residual block mix and attention input | Inkling learned state mix kernel; weighted RMSNorm; `ds4_gpu_add_tensor` | Populate the block bank in reference order and choose one mixed input for SWA or separate Q/K/V inputs for full attention | `attn_res_query`, `attn_norm` or Q/K/V input norms; current block bank and its previous-pass copy | Up to four 2304-wide block rows plus the current vector. Copies must remain GPU-to-GPU and share the token command buffer. |
| Cross-pass Q/K/V controllers | Controller input norm, F32 matmul, tanh-gated update; four controller weight sets bind for each core layer | Build four messages from the prior pass, select Q/K/V/FFN controller weights, and invoke them only on core passes 1 and 2 | `loop_msg_query`, `loop_{q,k,v,ffn}_ctrl_{down,gain,forget}.weight`; previous-pass bank and four messages | Each current helper allocates four scratch tensors per call. The new session arena exists, but controller calls do not yet use it. |
| Q/K/V and relative projections | Q8_0 GPU matmul | Connect each selected/mixed normalized input to the correct layer projection and shape the results into heads | `attn_q.weight`, `attn_k.weight`, `attn_v.weight`, `attn_rel.weight` | Q: 3072 floats; K/V: 1024 each; relative state: 384. Keep all outputs in persistent session workspace. |
| K/V causal short convolution | Static/dynamic causal shortconv kernel; convolution histories now have four distinct GPU views per logical visit | Project dynamic taps on full-attention layers, feed the visit's history, and commit history only with the token | `attn_k_conv.weight`, `attn_v_conv.weight`, and on full attention `attn_{k,v}_conv_proj.weight`; three prior rows per group and visit | K/V histories are 1024 channels each. No host readback is needed. A single batched command buffer should carry the projection and convolution dispatches. |
| Q/K clipping, V norm, relative clipping | Per-row Q/K learned scaling and L2 clipping; plain row RMSNorm | Apply Q, K, V, and relative-state rules in reference order, including full-attention position scaling | `attn_q_clip.weight`, `attn_rel_clip.weight`; current position | Q/K/R use FP32 work rows; V norm is 8 rows × 128 values. |
| Relative bias, sink, and GQA attention | Relative-bias projection; decode attention with GQA, FP16 cache reads, sliding/full key ranges, relative bias, and sink | Generate causal relative-distance indices, project/gather the bias, append K/V to this visit's cache, and wire the sink convention exactly | `attn_rel_bias.weight`, `attn_sink_proj.weight`; separate K and V FP16 caches for each visit | At 8192 context the two KV banks allocate 416 MiB total; relative-bias scratch is at most 24 × 8192 FP32 values. Decode operator only: causal chunked prefill is missing. |
| Attention gate, output projection, and post-convolution | Sigmoid attention output gate; Q8_0 matmul; shortconv; residual add | Apply gate before projection, run post-attention convolution, then save/add the correct residual block input | `attn_gate.weight`, `attn_output.weight`, `attn_post_conv.weight`; visit's attention-output history | 3072-wide gated attention and 2304-wide output. All dispatches should remain in one token command buffer. |
| FFN input mix, controller, and norm | State mix, four-controller primitives, weighted RMSNorm | Reconstruct the exact block list after attention residual update and apply the FFN controller only for loop passes 1 and 2 | `ffn_res_query.weight`, `ffn_norm.weight`; current/previous block banks | Up to six 2304-wide states (55 KiB) in reusable scratch; current controller helper allocates per call. |
| Top-8 MoE and SiTU | Correction-biased sigmoid top-8 routing, IQ4_NL/Q8_0 expert GEMV, SiTU, routed accumulation | Bind the selected expert IDs to the physical owner tensors and sequence router, gate/up, SiTU, down, and residual update in one visit | `ffn_gate_inp.weight`, `exp_probs_b.bias`, shared physical `ffn_gate_up_exps` and `ffn_down_exps` | Selected IDs/weights and expert activations stay on GPU. Current MoE orchestration allocates scratch per call; it needs a session-workspace entry point. |
| Post-FFN convolution and layer output | Static causal shortconv and `ds4_gpu_add_tensor` | Commit the visit-specific FFN history and produce the next layer's current vector | `ffn_post_conv.weight`; visit's FFN-output history | 2304-wide input/output; no host/device transfer per visit. |
| Three-pass loop, final mix, and output head | 48-visit plan, learned mix/controller operators, Q8_0 matmul, weighted RMSNorm, softcap | Restore prelude output and blocks at each core pass; form prior-pass bank; run epilogue; crop padded vocabulary from 201024 to 200058 | `output_res_query.weight`, `output_norm.weight`, `output.weight`; prelude snapshot, two block banks, padded logits | One 8 MiB session work arena plus 0.77 MiB padded logits. Return logits through the existing session API after one final readback; no per-layer readback. |

The private `ds4_inkling_gpu_context` is owned by the normal Metal `ds4_session`: it contains 24 physical weight-owner IDs, 48 disjoint KV cache slots, 48 × 4 disjoint convolution-history views, the 48-visit plan, current/previous residual banks, a prelude snapshot, token/embedding/activation/logit buffers, and reusable scratch. The allocator test reports 454,785,284 bytes (433.72 MiB) at context 8192. This is allocated size, not measured resident working set. Session creation admits the model only when the model map, context allocation, and 2 GiB runtime reserve fit under Metal's recommended working set; one live session is supported.

Normal engine open accepts only the supported single-session Metal configuration and maps the GGUF for Metal. `ds4_session_sync`, `ds4_session_eval`, and `ds4_session_eval_argmax` use the native Inkling path; the generic DeepSeek graph is not used. Each token evaluation batches decoder work and reads back logits once, while KV and convolution history remain on the GPU. Prefill is correct but conservative: `ds4_session_sync` evaluates prompt tokens incrementally rather than using a block-prefill kernel.

## GGUF inspector

`gguf-tools/inkling2_inspect.py` reads GGUF v3 metadata and tensor descriptors. It validates tensor shapes, canonical byte sizes for the supported F32/F16/Q8_0/IQ4_NL tensor types, alignment, overlap, and file bounds without reading tensor payloads. Use `--summary`, `--tensors`, `--json`, or `--validate-inkling2`; all modes report that payload bytes were not read. The Inkling validator checks the loop metadata, 24-entry attention schedule, 48 logical visits, 156 F32 loop-controller tensors, and the IQ4_NL/Q8_0 expert tensor assignments.

Synthetic coverage: `make test-inkling2-inspect` passes 18 tests for valid descriptors, primitive/array metadata, truncation and size limits, duplicate keys and names, invalid shapes and types, offset alignment/overlap/bounds, quant block sizes, the expected Inkling layout, and payload-free inspection. A pinned remote range parse confirmed `data_offset=11,977,632`, 704 tensors, and counts F32=478/Q8_0=186/IQ4_NL=40; no payload bytes were part of that exact range and no local GGUF is present.

## 48-visit plan and loader block sizes

`ds4_inkling.h` validates the supported 24-layer metadata and emits the execution order: prelude layers `[0,6)`, core `[6,18)` for three passes, then epilogue `[18,24)`. Each plan entry stores a physical layer index as its shared weight owner and a unique cache slot. Pass 0 uses cache slots 6–17, pass 1 uses 24–35, and pass 2 uses 36–47; prelude and epilogue use their physical indices. The planner validates the attention pattern and rejects changed boundaries, pass counts, residual block size, or layer schedule. The session consumes this plan directly, shares the 24 physical weight owners, and keeps state independent for all 48 visits.

`ds4_gguf.h` is the shared type-size table used by the mmap loader and the C test. It corrects GGML type 19 IQ1_S to 256 elements / 50 bytes and type 20 IQ4_NL to 32 elements / 18 bytes. The canonical IQ4_NL layout is a 2-byte scale plus 16 packed bytes (`QK4_NL=32`) in the pinned llama.cpp source.

## M7 native metadata validation and tensor binding

`config_validate_inkling2_model` accepts only the pinned text model contract: 24 physical layers, hidden 2304, padded/unpadded vocabulary 201024/200058, 24 query heads and 8 KV heads per layer, loop `[6,18)` for three passes, rank 64, six-layer residual blocks, 1024 sliding window, four-position short convolution, zero rotary dimension, and the recorded numeric parameters. It also checks Inkling's `gpt2` tokenizer metadata, EOS/padding IDs, and no implicit BOS/EOS.

`inkling_weights_bind` requires all 704 descriptors and validates every bound tensor's name, rank, dimensions, and type. The engine stores one array of 24 physical layer weights. Its loop plan contains 48 references to those physical owners with 48 unique cache IDs; it does not copy weights for repeated visits. Full-attention layers bind their Q/K/V norms and Q/K convolution projections, while sliding layers bind their single attention norm. Expert matrices use IQ4_NL in layers 0–19 and Q8_0 in layers 20–23.

`make test-inkling2-gguf` creates a sparse, payload-free 17.6 GB GGUF fixture from the live descriptor inventory, exercises the real mmap parser and metadata validator, checks the complete binder and engine inspect-only path, and verifies rejection of a missing descriptor, wrong tensor type, and wrong shape. This allocates only a sparse file and does not write model payloads. `make test-inkling2-model` checks valid and unsupported architecture metadata independently.

Normal Metal engine open initializes the Inkling model map after successful binding and plan creation. It rejects CPU/CUDA, multi-session features, TP, distributed mode, streaming, and sidecar options explicitly. Session creation allocates the 48-slot GPU state through the standard API; normal generation now uses that session and produces real greedy tokens. Same-GGUF short-prompt parity is recorded above. Long-context validation, stable performance measurements, and EOS/rollback coverage remain open.

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

The Inkling engine-open branch now loads that vocabulary after GGUF validation and binding, including for `--inspect` sessions. The sparse GGUF fixture supplies the required special-token strings and verifies that the inspect-only engine converts role/content/end markers to their expected IDs.

The text-only prompt APIs render the model template's role markers and content blocks. Initial system messages come first, then the reasoning-effort system block is inserted before the first non-system message. Named values map to `none=0.0`, `minimal=0.1`, `low=0.2`, `medium=0.7`, `high=0.9`, and `xhigh=max=0.99`; decimal strings in `[0.0, 0.99]` are accepted through the shared reasoning-mode parser. Assistant history ends with both `end_message` and `content_model_end_sampling`; an open generation prompt ends with `message_model`. The CLI's one-shot and REPL prompt builders preserve this order. Tool schemas, image/audio payloads, and structured assistant tool-call generation are not implemented by this text-only path.

`make test-inkling2-tokenizer` checks GPT4O splitting for mixed/all-uppercase words, ASCII case boundaries around Unicode letters (`AéB`), contractions, Unicode letters, number groups, punctuation spanning newline/slash, direct-word lookup, ranked-merge fallback, exact effort strings, numeric-effort de-duplication, rendered special-token recognition, text chat message order, and the generation prefix. The `AéB` regression follows the backtracking behavior in the pinned llama.cpp GPT4O expression. `make test-inkling2-gguf` also opens the synthetic descriptor fixture in inspect-only mode and checks loaded marker IDs. The server regression suite additionally checks `minimal`, `xhigh`, and decimal reasoning levels. `make`, `make cpu`, the Inkling focused targets, and `./ds4_test --server` pass after this change on the Apple M4.

This milestone does not enable inference. The session API now owns the per-visit Metal state, but eval still fails explicitly until the graph connects embeddings, attention, routed experts, loop message banks, output projection, and sampling. The 17.6 GB model is not local, so no token stream, full-model numeric comparison, or end-to-end benchmark has been measured.
