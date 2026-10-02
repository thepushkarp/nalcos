# Configuration

NaLCoS has no selected embedding model by default. Lexical search can bootstrap an index without one. Select a known model or an explicit profile when running `init` or `sync`; these commands apply model changes and may download the required assets.

## Configuration and data locations

`--config FILE` selects a TOML file. `NALCOS_CONFIG` supplies the same option when no CLI value is given. Without either, NaLCoS looks for `nalcos/config.toml` in the platform configuration directory: normally `~/Library/Application Support` on macOS and `$XDG_CONFIG_HOME` or `~/.config` on Linux. An absent default file uses built-in settings; an explicitly requested missing file is an error.

Configuration files use `version = 1`. Unknown fields and unsupported versions are errors. Start with the [example file](../examples/config.toml):

```sh
nalcos --config examples/config.toml status
nalcos --config examples/config.toml init --model profile:minilm
```

`init` and `sync` choose a model in this order:

1. An explicit `--model`.
2. A nonempty `default_model` whose value changed since the last applied setup.
3. The persisted pending selection, if a replacement generation is being built; otherwise the active selection.

A CLI model choice persists across later syncs. An unchanged configuration default cannot switch it back. Removing `default_model` also leaves the persisted selection in place. Edits to the selected named profile are applied by `init` or `sync`; `search` continues using the active generation until setup applies those edits.

The index-storage root is selected in this order:

1. `NALCOS_DATA_DIR`.
2. The top-level `data_dir` configuration value.
3. NaLCoS's platform-local data directory.

Each repository index is stored under `repos/<repository-identity>/index.sqlite3`. Identity is derived from the canonical common Git directory, so linked worktrees share stored history. The active/staging embedding selection is persisted in SQLite; NaLCoS does not write model choices into a repository TOML file.

Synchronization indexes new commits in the requested scope. It retains cached commits reachable from every current ref under `refs/` that resolves to a commit, from worktree HEADs, or from the explicitly requested scope; cached commits outside that set are pruned. The refs include custom namespaces and `refs/stash`. Reflog-only history is not retained unless explicitly requested. This retention set does not broaden the query scope or index every ref automatically. Evidence IDs for pruned commits are no longer available from the cache.

Identical document content can reuse active-model vectors before obsolete associations are pruned, including after a rebase. Changes to indexing policy rescan retained indexed history and the requested scope; discarded history does not need to remain available as Git objects.

## Select a model

The following recipes are registered explicitly. They are candidates for evaluation, not ranked recommendations or a qualified default:

| Model ID | Alias | Backend and registered artifact |
| --- | --- | --- |
| `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | `minilm` | ONNX, `onnx/model.onnx` |
| `jinaai/jina-embeddings-v2-base-code` | `jina-code`, `jina` | ONNX, `onnx/model.onnx` |
| `BAAI/bge-small-en-v1.5` | `bge-small` | ONNX, `onnx/model.onnx` |
| `ggml-org/embeddinggemma-300M-GGUF` | `embeddinggemma`, `gemma` | GGUF, `embeddinggemma-300M-Q8_0.gguf` |
| `Qwen/Qwen3-Embedding-0.6B-GGUF` | `qwen3` | GGUF, `Qwen3-Embedding-0.6B-Q8_0.gguf` |

Each registered recipe pins a model revision and defines its dimensions, token limit, pooling, and query/document formatting. A bare HF ID must have a known recipe. Other repositories require a custom `[profiles.NAME]` section; NaLCoS cannot infer an arbitrary model's embedding contract from its repository name.

For example, explicitly choose the MiniLM recipe and later switch to another candidate:

```sh
nalcos init --model sentence-transformers/multi-qa-MiniLM-L6-cos-v1
nalcos sync --model BAAI/bge-small-en-v1.5
nalcos status --check
```

`--revision REVISION` selects an explicit HF revision during setup. `--variant FILE` selects an artifact filename within that model repository, not a quality preset; the artifact must remain compatible with the profile's backend and embedding contract. Local model revisions are resolved to immutable commits before use.

An ordinary `sync` keeps that resolved commit even if the requested HF ref, such as `main`, has moved. Editing only a profile's query prefix or runtime settings also preserves the pin. To resolve a moving ref again, pass it explicitly, for example `nalcos sync --revision main`; selecting `--model` again also resolves the requested revision. Changing the desired model ID or revision in configuration requests a new resolution when `init` or `sync` applies it. `--reembed` alone rebuilds the currently pinned revision.

A model change builds a complete replacement embedding generation for retained indexed documents rather than mixing vectors from different models. Use `--reembed` to force a new generation for the same profile. Plain `sync` resumes an interrupted replacement before returning to ordinary active-model updates. Selecting a different model explicitly supersedes incompatible pending work; selecting the active model cancels an incompatible pending replacement. Check `status` before assuming that a requested model has become the active generation.

`--dry-run` plans setup or synchronization without creating an index, downloading a model/runtime, or probing a provider. `--offline` disallows downloads and provider network calls, including calls to a localhost embedding server. Existing local model assets can still be used offline.

### Shared model cache

HF model artifacts use the shared Hugging Face cache: `HF_HUB_CACHE` takes precedence; otherwise the hub cache is derived from `HF_HOME`, normally `~/.cache/huggingface/hub`. Cached snapshots are reused across repositories. Model weights are not copied into each NaLCoS index.

Explicit setup may download missing artifacts. Automatic search updates and `status` never download models or runtime libraries. HF credentials can be supplied through `HF_TOKEN` or the Hugging Face token cache; keep them out of the TOML file. See [HF cache documentation](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache) for shared-cache management.

## Index and retrieval settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `index.exclude_paths` | `[]` | Glob patterns excluded from historical indexing. |
| `index.max_blob_bytes` | `1048576` | Per-blob size limit. |
| `index.max_patch_bytes` | `2097152` | Cumulative patch-byte limit per commit. |
| `index.max_document_bytes` | `16384` | Maximum retrieval-document size; at least `256` bytes. |
| `index.chunk_overlap_tokens` | `32` | Overlap between adjacent tokenizer-bounded embedding chunks. |
| `search.candidate_limit` | `100` | Candidate retrieval limit before final result selection; valid range `1..=10000`. |
| `search.rrf_k` | `60.0` | Reciprocal-rank-fusion smoothing constant. |
| `search.lexical_weight` | `1.0` | Lexical contribution to hybrid ranking. |
| `search.semantic_weight` | `1.0` | Semantic contribution to hybrid ranking. |

Blob and patch limits must be positive; the document limit must be at least `256` bytes. Ranking weights and `rrf_k` must be finite and positive. Use `--mode lexical` or `--mode semantic` to select a single retrieval mode rather than setting a hybrid weight to zero. Search `--limit` is separate from `candidate_limit` and accepts `1..=100`.

Exclusions and size limits affect coverage. Inspect reported omissions before interpreting absent results. A search's repeatable `--path` accepts Git pathspecs and narrows that query; it is not a persistent indexing exclusion. Diff evidence must match the old or new path. Commit-message evidence remains eligible when the commit's first-parent changes touch a matching path; unrelated file diffs from the same commit are excluded.

## Runtime settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `runtime.query_device` | `"auto"` | Query device override; `auto` defers to the selected profile's `device`. |
| `runtime.index_device` | `"auto"` | Bulk-indexing device override; `auto` defers to the selected profile's `device`. |
| `runtime.threads` | Available parallelism, capped at `4` | Native inference thread count; valid range `1..=1024`. |
| `runtime.batch_size` | `8` | Indexing batch size; valid range `1..=1024`. Query embedding uses a batch size of one. |
| `runtime.ort_library` | Unset | Optional path to an existing ONNX Runtime shared library. |

Device selection uses the global `--device` option first, then a non-`auto` workload setting (`runtime.query_device` or `runtime.index_device`), then the selected profile's `device`. The profile defaults to `auto`. An explicit `--device auto` overrides even a strict profile device; use it with `init` to request automatic calibration. Search uses a matching cached calibration or falls back to CPU, as described below.

`sync` applies changed device, thread, batch, or runtime-library settings by probing both workloads and calibrating automatic device choices while preserving the generation and existing vectors; unchanged syncs skip those probes.

Runtime device names are `auto`, `cpu`, `metal`, `cuda`, and `vulkan`; CUDA and Vulkan may include a zero-based device index, such as `cuda:1`. Backend support and device availability are checked at runtime. Inspect the reported effective backend/device instead of assuming the requested accelerator was used.

ONNX uses a dynamically loaded runtime library. Explicit setup can install the pinned ONNX Runtime `1.23.2` CPU package when needed; search does not install it implicitly. On macOS, libraries older than `1.23.2` are rejected before runtime initialization because they can abort during process shutdown. A configured `runtime.ort_library` takes precedence over `NALCOS_ORT_LIBRARY`, then `ORT_DYLIB_PATH`; otherwise NaLCoS checks bundled/managed libraries. `NALCOS_RUNTIME_CACHE` can relocate the managed runtime cache. GGUF uses the bundled llama.cpp backend. Native Metal support is built for Apple Silicon. Linux accelerator builds are optional:

```sh
cargo install --path . --locked --features cuda
# Or, with the required Vulkan toolchain installed:
cargo install --path . --locked --features vulkan
```

The corresponding native SDK/toolchain is required. CUDA remains unqualified in this alpha. These build options do not qualify a Linux CUDA/Vulkan release or every model/device combination.

`status --check` performs an explicit encoder probe with synthetic input and reports runtime information. It does not download missing assets, repair an index, or send repository text to an API provider. A successful probe verifies that the encoder runs; retrieval quality needs a separate evaluation.

### Compare CPU and Metal

During `init`, an effective `auto` device measures the available devices against CPU using the same artifact. Query and indexing workloads have separate calibration because their batch sizes differ. Search reuses a matching calibration; without one, it selects CPU and reports why. ONNX `auto` uses CPU on macOS; Metal requires a compatible GGUF model.

After explicitly installing a GGUF candidate and building its index, inspect each device on Apple Silicon:

```sh
nalcos --offline --device cpu --json status --check
nalcos --offline --device metal --json status --check

time nalcos --offline --device cpu search "stop retrying cancelled requests" \
  --mode semantic --freshness cached
time nalcos --offline --device metal search "stop retrying cancelled requests" \
  --mode semantic --freshness cached
```

Explicit device requests fail if that device cannot run the selected model. Inspect the runtime's `selected_device`, `calibration`, and `fallback_reasons`. These timed commands include CLI startup and retrieval; repeat them on the same scope with other inference and build work stopped. They do not isolate embedding latency. See [benchmark methodology and results](benchmarks.md) for controlled comparisons.

Native cached-model smoke checks have passed on Apple Silicon for ONNX CPU and GGUF CPU/Metal. They check runtime selection, vector shape and finite normalization, repeated/batched consistency, and chunking. See [native runtime validation](native-runtime.md) for exact numerical comparisons and unresolved reference-parity differences. The alpha still reports `release_qualification: "pending"`: runtime checks do not establish retrieval quality or the product's benefit over existing tools. CUDA remains unqualified, and no default embedding model has passed the quality and product gates.

## Custom profiles

Profiles live under `[profiles.NAME]` and are selected with `--model profile:NAME` or `default_model = "profile:NAME"`.

| Field | Contract |
| --- | --- |
| `id` | HF repository ID for local backends; provider model ID for API backends. |
| `revision` | Model/deployment version. Local HF refs are resolved to a commit; API profiles require an explicit version other than `main` or `latest`. |
| `backend` | `onnx`, `gguf`, `open_ai`, or `ollama`. `openai` is accepted as an alias for `open_ai`. |
| `artifact` | Repository-relative model filename for ONNX/GGUF. |
| `dimensions` | Expected output-vector width, between `1` and `65536`. |
| `max_tokens` | Token limit including model formatting, between `8` and `131072`. |
| `pooling` | `mean`, `cls`, `last`, or `model`; default `mean`. Use the model's actual embedding contract. |
| `query_prefix`, `document_prefix` | Exact role-specific formatting prepended to inputs; default empty. |
| `tokenizer` | Repository-relative `tokenizer.json` for ONNX; absolute local tokenizer JSON path for API profiles. GGUF can use its embedded tokenizer. |
| `extra_files` | Additional repository-relative files required by a local model, such as external ONNX tensor data. |
| `output` | Optional ONNX output name. |
| `endpoint` | Explicit HTTP(S) API endpoint without embedded credentials, query, or fragment. |
| `api_key_env` | Name of the environment variable holding an API credential. |
| `device` | Profile device, default `auto`; used when the workload setting is `auto` and no CLI override is supplied. |

The [complete native example](../examples/config.toml) pins the MiniLM recipe. Device placement and credential-variable names are operational settings. The document-vector fingerprint covers the document embedding contract and resolved file contents; a query-prefix-only change does not require recomputing document vectors. Applying that change with `sync` still probes the encoder and, for API profiles, checks the provider's identity. Pending documents are embedded normally.

### API embedding profiles

API backends require explicit setup. Repository text is sent to the selected endpoint for indexing and query text is sent for search. A server running on localhost is still a network provider for `--offline`.

This configuration is a template: replace the model/deployment details, dimensions, context length, endpoint, and tokenizer path with the provider's actual embedding contract before using it.

```toml
[profiles.provider]
id = "your-embedding-model"
revision = "your-immutable-deployment-version"
backend = "open_ai"
dimensions = 768
max_tokens = 2048
pooling = "model"
tokenizer = "/absolute/path/to/tokenizer.json"
endpoint = "https://provider.example/v1"
api_key_env = "EMBEDDING_API_KEY"
```

Set the named credential in your environment, then run `nalcos init --model profile:provider`. NaLCoS stores the variable name, not its secret value. The OpenAI-compatible backend appends `/embeddings` to its base endpoint. An `ollama` profile commonly uses `http://localhost:11434` as its base endpoint and appends `/api/embed`; it follows the same explicit-version and tokenizer requirements and is not selected automatically because an Ollama server is running.

NaLCoS stores a synthetic document embedding as an identity probe and checks it before using an API encoder. A sufficiently different response produces `provider_changed`; pin the intended deployment and run `nalcos sync --reembed`. This check can catch changes behind a provider alias, but it cannot prove that every embedding remains unchanged. Keep the deployment version explicit. Provider probes also require network access, so `--offline` blocks them for both API backends, including Ollama on localhost.
