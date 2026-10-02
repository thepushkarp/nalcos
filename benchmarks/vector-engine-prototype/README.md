# Native vector-engine prototype

USearch 2.26.2 builds successfully on Apple Silicon with Rust/C++. The small local correctness run exercised persisted memory-mapped F32 indexes, scope predicates, per-commit aggregation, and exact reranking. See `results.json` for input provenance and recall. This is **not a qualified search backend**: the 300,000-vector synthetic construction was stopped during shared-host execution, and there is no complete scale recall or latency result. Production NaLCoS continues to use SQLite and exact vector scoring.

The 12 queries are preserved MiniLM FP32 ONNX reference outputs from the local model pilot. There were 111 commits and 1,282 actual vectors. At search expansion 128, all-scope mean commit Recall@10 was 99.17%, but one query had 90%; mean Recall@100 was 99.83%. The two narrower identifier-based scopes returned 100%. These scopes test predicate behavior, not Git graph correctness. Timings from the contended run are deliberately omitted.

The prototype pins USearch and its full dependency graph in `Cargo.lock`. It uses normalized F32 vectors with inner-product distance, connectivity 32, and construction expansion 128. No vector quantization is applied. Approximate graph traversal may still miss exact neighbors. The Apache-2.0 license is recorded from the downloaded crate metadata. Official API documentation: https://unum-cloud.github.io/USearch/rust/index.html.

## Running

Build this standalone crate with `cargo build --release --locked --manifest-path benchmarks/vector-engine-prototype/Cargo.toml`. The main NaLCoS crate does not depend on it. Set `CARGO_TARGET_DIR` to this prototype's `target` directory if your Cargo configuration overrides output placement; the Python evaluator expects that location.

Use the prototype binary's `build PREFIX INDEX_SQLITE` command to read an existing active generation without modifying it. This creates `PREFIX.usearch`, `PREFIX.vectors`, `PREFIX.commits`, and a provenance JSON beside the chosen prefix. Use a disposable directory outside the checkout. Construction reserves four native threads and requires memory for both the source vectors and the graph.

Supply a raw little-endian F32 query matrix with 384 columns to `python3 benchmarks/vector-engine-prototype/evaluate.py PREFIX --queries QUERIES_F32`. The pilot's `minilm-original-onnx-cpu-matched.npz` contains these vectors in `queries.npy`; strip its NumPy header before supplying it. Raw embeddings, repository content, local paths and serialized indexes are intentionally not included here. The evaluator records hashes, compares all/10%/1% commit scopes, and increases search expansion until mean Recall@10 and Recall@100 reach 99% or its configured sweep ends. This stopping rule is an exploratory check, not a release qualification rule.

The benchmark orders database vectors by document ID and chunk ordinal. A dense key maps each vector to its commit. The eligibility predicate executes during graph traversal. Returned vectors are reranked using F64 dot-product accumulation and aggregated by maximum chunk score per commit. Candidate count begins at four times the requested commit count and doubles until enough distinct eligible commits are available or results are exhausted. This avoids duplicate chunks filling the result list; it does not guarantee ANN recall.

The exact reference scans the same F32 vectors with the same predicate and aggregation. Engine timings exclude query inference, Git scope resolution, source verification and full CLI startup. OS caches are not flushed. On macOS a restrictive sandbox can deny NumKong's sysctl probes and silently select serial kernels; verify the recorded `hardware_acceleration` before comparing runs. The host execution context detected NEON.

## Integration constraints

SQLite should remain canonical. A production sidecar would require an immutable index and key mapping tied to an embedding fingerprint, corpus watermark and checksums. Persist and fsync temporary files, rename them, then activate the matching manifest in SQLite. Readers must pin the same SQLite snapshot and sidecar generation. Missing or corrupt files should fall back to exact canonical vectors; crash-orphaned files need safe reclamation.

Incremental vectors require an exact delta merged with the immutable base until a rebuild is worthwhile. Resolve live Git scope and document exclusions into a dense allowed-key bitmap before search; per-node callbacks must not query SQLite. Merge base and delta before commit aggregation, and refill after source evidence is rejected. Retired files must remain available to existing readers. This extra storage and lifecycle complexity needs an end-to-end measured benefit and robust recall evidence before production integration.
