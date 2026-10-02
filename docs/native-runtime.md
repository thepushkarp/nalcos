# Native runtime validation

MiniLM INT8 on CPU is the alpha setup default. `release_qualification` remains `pending`. The checks below establish specific execution and numerical behavior; they do not establish retrieval quality or release performance.

The default ARM64 INT8 CPU smoke passes shape, finite normalization, token budget, and same-batch repeatability (cosine at least 0.9999). Dynamic activation quantization is sensitive to batch composition: the two batch-versus-singleton samples measured cosine 0.9943486 and 0.9927478. This diagnostic is separate from the stricter FP32/GGUF and automatic-device parity gates, which remain unchanged. Retrieval qualification is still pending.

The October 2, 2026 checks used an Apple M2 with 16 GB memory, the cached pinned artifacts listed in [configuration.md](configuration.md), `ort` 2.0.0-rc.10, and `llama-cpp-2` / `llama-cpp-sys-2` 0.1.158. Native Rust tests used the development build, four inference threads, and batch size eight. Linux CUDA/Vulkan and Intel Mac execution have not been exercised on this host.

| Native path | Reference | Minimum cosine across three queries and three documents | Result at 0.9999 |
| --- | --- | ---: | --- |
| MiniLM original ONNX, CPU, ONNX Runtime 1.23.2 | Saved original ONNX pilot | 0.99999995 | Passed |
| Jina v2 base code original ONNX, CPU, ONNX Runtime 1.23.2 | Saved original ONNX pilot | 0.99999994 | Passed |
| MiniLM original FP32 GGUF, CPU | Original ONNX pilot | 0.99999874 | Passed |
| MiniLM original FP32 GGUF, Metal | Original ONNX pilot | 0.99999863 | Passed |
| EmbeddingGemma Q8 GGUF, CPU | Saved llama-server CPU pilot | 0.99967258 | Failed |
| EmbeddingGemma Q8 GGUF, Metal | Saved llama-server CPU pilot | 0.99972283 | Failed |

The GGUF pilot used Homebrew llama.cpp build 10809, commit `5266f24da`; the Rust binding vendors a different source revision and build configuration. Its batch and context shapes also differ from the server. The Gemma results therefore identify unresolved compatibility differences, without isolating their cause. The same Rust build's Gemma CPU versus Metal minimum cosine was 0.99974386 for these queries and 0.99977865 for these documents. These differences have not been accepted as release parity.

Cached Gemma CPU and Metal smoke tests separately passed dimension, finite-vector, normalization, tokenizer chunking, every-sequence batch-versus-singleton consistency, and exact special-token-ID checks on small synthetic inputs. Metal device enumeration accepts the native backend name `MTL`. Those smoke checks do not supersede the failed broader reference comparison. Automatic selection requires every vector from three query samples and three document samples to match the same-artifact CPU reference at cosine 0.9999 before comparing performance. The calibration cache policy is versioned, so cached decisions from a weaker gate cannot be reused. Explicit device selection remains available with `qualified: false`.

Explicit device failures return `explicit_device_unavailable`, whether the choice comes from the CLI, runtime configuration, or model profile. Hybrid search cannot turn those failures into lexical fallback. Cancellation, deadlines, and invalid input or configuration retain their own error codes.

[A release CLI smoke run](benchmarks.md#acceleration-and-scale) rejected Metal at cosine 0.99983677 for indexing and 0.99981067 for queries, retained CPU, and reused its cached query calibration for a semantic search with verified evidence. Unchanged sync reported zero document embeddings; total embedding calls were not instrumented. This single-commit fixture leaves retrieval quality and default-model qualification pending.

The managed CPU runtime is ONNX Runtime **1.23.2**, installed in the user cache from official Microsoft archives with pinned SHA256 digests. The tested 1.22.0 macOS library completed inference but aborted during process shutdown in `OrtEnv` destruction, matching [upstream issue 25038](https://github.com/microsoft/onnxruntime/issues/25038). [The 1.23.2 environment implementation](https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/session/ort_env.cc) uses different ownership. Replacing only the runtime with 1.23.2 made the same release CLI status check exit successfully. The installer test loads the extracted library, and macOS overrides older than 1.23.2 are rejected before creating an environment. Runtime reporting includes the loaded version and build information.

GGUF tokenization recognizes registered literal special tokens as well as automatically added boundaries, matching the tested llama.cpp embedding endpoint. This policy has a separate document fingerprint version. Loading a persisted descriptor from an older contract returns `encoder_contract_changed` with a sync action before inference; old vectors cannot silently receive queries under the new policy. Query-prefix and device-only changes do not change document identity.

The normal test suite requires no model downloads. Real-model tests are ignored unless explicitly selected:

```sh
NALCOS_RUN_MODEL_TESTS=1 cargo test --locked --lib \
  embedding::tests::cached_embeddinggemma_metal_smoke -- --ignored --exact
```

The same opt-in requirement applies to `cached_minilm_int8_cpu_smoke`, `cached_minilm_cpu_smoke`, `cached_jina_cpu_smoke`, and `cached_embeddinggemma_cpu_smoke`. The exact pinned files must already exist in the shared Hugging Face cache; ONNX tests also need a compatible installed runtime. `cached_native_reference_parity` takes an independently generated JSON fixture through `NALCOS_EMBEDDING_REFERENCE_JSON` and never downloads. `managed_ort_install_smoke` is a separate, explicitly enabled installer check requiring `NALCOS_RUN_RUNTIME_INSTALL_TESTS=1` and an isolated `NALCOS_RUNTIME_CACHE`; it may download the pinned runtime pack.

ONNX inference supports cancellation through its native run options. GGUF cancellation is checked around model loading and complete inference batches; the Rust binding does not expose a safe abort callback for a running kernel. A currently executing GGUF batch can therefore exceed the requested deadline before control returns.

In-flight GGUF cancellation is tracked in [issue #60](https://github.com/thepushkarp/nalcos/issues/60).
