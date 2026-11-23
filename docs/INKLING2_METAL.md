# Inkling 2 Mini on Apple Metal

This document tracks native DS4 support for the text-only Inkling 2 Mini GGUF. The first supported model format is the mixed IQ4_NL GGUF, with an 8192-token context and FP16 key/value state. DS4 loads the GGUF directly through its mmap loader and executes operators in its native C and Metal paths; it does not use llama.cpp or MLX as a runtime dependency.

## Milestone status

| Milestone | Status | Evidence |
| --- | --- | --- |
| M0: repository, references, and host baseline | Complete | Build and server results below; Metal regression baseline captured before implementation. |
| M1: GGUF inspector and 48-visit mapping | In progress | Read-only inspector and 18 synthetic tests are complete; the native visit planner and loader type-size correction are next. |
| M2+: native operators, generation, numeric parity, and performance | Not started | Model-dependent validation requires the GGUF to be available locally. |

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
