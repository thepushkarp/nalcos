# Embedding and search benchmarks

The alpha default is **MiniLM INT8 on CPU**, chosen for its 22 MiB weights and fast
local indexing. This is a product choice for the alpha, not a claim that the model
passed broader quality, licensing, portability, or product qualification gates.
Profiles remain configurable; explicit setup downloads models into the shared
Hugging Face cache. This repository does not distribute weights.

## Model comparison

The exploratory pilot used an Apple M2 MacBook Air with 16 GiB RAM, four CPU
threads, batches of eight, three query warmups and 36 warm query samples.
NaLCoS snapshot `57bd49f174811f6bbaf4e541019f9a7e9f34a98e` supplied 111 commit
messages and 290 signed textual diff chunks. Twelve retrospective questions each
label one known commit; other relevant commits may be unlabeled. These are not
independent held-out questions.

| Model / runtime | Weights MiB | Documents/s | Warm query p50 ms | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniLM FP32 / ONNX CPU | 86.2 | 42.06 | 2.35 | 10/12 | .5810 |
| **MiniLM INT8 / ONNX CPU** | **22.0** | **81.70** | **1.14** | **11/12** | **.6250** |
| BGE-small FP32 / ONNX CPU | 126.9 | 19.43 | 5.95 | 10/12 | .5580 |
| Jina code INT8 / ONNX CPU, 1024 tokens | 154.4 | 9.68 | 7.15 | 12/12 | .7014 |
| EmbeddingGemma Q8 / llama.cpp CPU | 318.1 | 6.70 | 13.54 | 12/12 | .6508 |
| EmbeddingGemma Q8 / llama.cpp Metal | 318.1 | 35.31 | 12.65 | 12/12 | .6508 |
| Qwen3 0.6B Q8 / llama.cpp Metal | 609.5 | 5.40 | 34.92 | 12/12 | .6722 |

MiniLM is the smallest and fastest CPU candidate in this pilot. It trails the best
recall by 8.3 percentage points and MRR by .0764, outside the original five-point/.03
qualification tolerances. Its reviewed weight-license evidence remains incomplete
in the archived qualification report. The alpha selection does not certify those gates.

These semantic-only timings exclude Git, search and CLI startup; GGUF includes
loopback HTTP overhead. Caches were not cleared; thermal/load conditions were
uncontrolled. Runtime paths and input budgets differ. Weights are not peak RAM.
The basic FTS/RRF pilot is not a competent Git/rg agent baseline, and hybrid did
not consistently improve ranking.

## Acceleration and scale

Matched FP32 conversion used llama.cpp `5266f24da`, four threads, batches of eight,
the same 401 documents and three full indexing runs:

| MiniLM runtime | Median indexing seconds | Documents/s | Warm query p50 ms |
| --- | ---: | ---: | ---: |
| ONNX CPU | 11.10 | 36.13 | 2.46 |
| GGUF CPU | 7.18 | 55.84 | 15.04 |
| GGUF Metal | 3.08 | 130.08 | 3.85 |

All three produced Recall@10 10/12 and MRR .5810; minimum GGUF/ONNX cosine exceeded
.999996 with matching token IDs. Jina GGUF CPU tokenization matched but vector
parity failed (minimum cosine .0218), blocking GGUF timings. Separate native
EmbeddingGemma CPU/Metal calibration rejected Metal at cosine .99983677 for
indexing and .99981067 for queries and retained CPU. Qualification is specific to
the artifact/runtime; finite-vector smoke tests alone do not establish parity.

The native MiniLM INT8 NFT-history run used ONNX Runtime 1.23.2 CPU, 829 commits,
15,684 documents and 39,909 vector chunks. Its ten frozen retrospective questions
contained seven known positives and three unjudged questions:

| Mode | CLI p50 seconds | CLI p95 seconds | Known-gold Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| Lexical | .191 | .738 | 3/7 | .2381 |
| Semantic | .359 | .886 | 6/7 | .3333 |
| Hybrid | .422 | .649 | 5/7 | .4776 |

These are single observations per query, including startup and evidence checks,
with linearly interpolated percentiles. The frozen run predates a corrected source
coordinate bug and records nine verification warnings and 57 source omissions.
It has not been relabeled as a repaired-index evaluation. A separate repair smoke
preserved 15,683 source records and 39,907 vector chunks byte for byte, repaired
one source document, and repeated sync with zero embeddings. Complete history
coverage does not imply complete source-content coverage.

Synthetic 100,000-commit measurements on the M2/16 GiB host used 300,000 documents,
20 separate CLI invocations, cached freshness, limit 10 and a 16 KiB evidence budget:

| Search | CLI p50 seconds | CLI p95 seconds | Maximum seconds |
| --- | ---: | ---: | ---: |
| Lexical | .828 | 1.069 | 2.295 |
| Exact semantic, covering document index | 2.328 | 2.822 | 7.015 |

Nearest-rank p95 includes the first query; filesystem caches were not flushed.
Other development workloads paused during the query window. Semantic vectors are
deterministic synthetic 384-dimensional data, with a MiniLM FP32 query encoder:
this measures cost, not relevance. Exact search misses the two-second target.
The archived USearch prototype reached 99.17% mean commit Recall@10 on only 1,282
vectors/12 queries, with one query at 90%; the 300,000-vector build was incomplete.
It is not a production backend or an end-to-end speed result.

## Evidence archive

Detailed samples, diagnostics, conversion reports, historical harness sources,
corpus, prototype and the full methodology are in
[`nalcos-benchmarks-2026-10-02.tar.gz`](https://github.com/thepushkarp/nalcos/releases/download/v2.0.0-alpha.1/nalcos-benchmarks-2026-10-02.tar.gz),
distributed separately with the alpha release. The
[archive manifest](../benchmarks/archive-manifest.json) records the archive and
every member's SHA-256 and size for integrity verification.

Archive SHA-256:
`4e0fdb3746cb66fe0bce61fab3d6ae383d5289a4adecfeacfc87d37ee814fe33`.

Extract outside the checkout to preserve current documentation:

```sh
mkdir -p /tmp/nalcos-benchmark-archive
tar -xzf nalcos-benchmarks-2026-10-02.tar.gz -C /tmp/nalcos-benchmark-archive
python3 benchmarks/replay_pilot.py \
  --archive /tmp/nalcos-benchmark-archive \
  --output /tmp/nalcos-pilot-replay --repo .
```

The archive preserves historical labels, failures and limitations. It has no
model weights, raw vectors, private repository text or personal local paths.
Small [query/label fixtures](../benchmarks/fixtures/pilot/queries.jsonl) and the
[candidate manifest](../benchmarks/fixtures/pilot/candidates.json) remain in Git.

## Reproduction

Matched conversion uses these pinned HF revisions:

| Model | Revision |
| --- | --- |
| `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | `b207367332321f8e44f96e224ef15bc607f4dbf0` |
| `jinaai/jina-embeddings-v2-base-code` | `516f4baf13dec4ddddda8631e019b5737c8bc250` |

The corpus SHA-256 is
`190f64c5e38719221c6e0b84809ead5e1041a9e6ca1b004c0b264185cc5ab491`.
Set `NALCOS_SNAPSHOT` to the pinned shared-cache snapshot:

```sh
uv run --python 3.13 benchmarks/run.py --mode benchmark \
  --snapshot "$NALCOS_SNAPSHOT" \
  --model-id sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
  --revision b207367332321f8e44f96e224ef15bc607f4dbf0 \
  --corpus /tmp/nalcos-benchmark-archive/benchmarks/pilot/2026-10-02/corpus.json \
  --output /tmp/nalcos-bench --label minilm-fp32-cpu
python3 -m unittest discover -s benchmarks -p 'test_*.py'
```

uv may install pinned Python dependencies; the harness never downloads models or
executes model-supplied Python. Use `prepare_gguf.py --help` for conversion and
`run.py --mode check` before GGUF timing. Checks require identical token IDs, finite
normalized vectors, minimum cosine .9999, maximum component error .002 and batch
invariance. Reports bind artifact/revision/backend/input contract; quantization
requires separate evaluation. Keep inputs fixed, counterbalance runtime order,
record hardware and distinguish cold startup from warm queries.

`cli_matrix.py` runs lexical/semantic/hybrid against prepared indexes and a frozen
manifest, preserving scope, generation, coverage and evidence checks. It never
syncs or downloads. `evaluate.py metrics` scores the resulting rankings.
`scale.py` generates a lexical fixture; `vector_scale.py` copies it and adds synthetic
vectors. Default scale output goes into ignored `benchmarks/output/`.
`workflows.py` assesses collected paired agent trials; it does not run agents.
Each harness exposes `--help`; complete commands and schema contracts are preserved
in the archive's `docs/benchmarks.md`.

## Remaining qualification

The strict evaluator remains available:

```sh
python3 benchmarks/evaluate.py qualify \
  --manifest benchmarks/fixtures/pilot/queries.jsonl \
  --candidates benchmarks/fixtures/pilot/candidates.json
```

This fixture intentionally exits **3** (unqualified); invalid input exits **2**,
successful evaluation **0**. Qualification requires reviewed licensing,
correctness and macOS/Linux CPU portability; at most 500 MiB weights; development
Recall@10 within five points and MRR within .03 of the best; then fastest CPU
indexing, preferring the smaller artifact within a 10% timing tie. Confirmation
requires 30 genuine locked independent questions across three repositories,
separate from development. The current 30-question diagnostic has zero certified
independent holdout questions. Unknown labels stay unjudged; missing responses
stay incomplete. The alpha choice does not overwrite archived evidence labels.

Product qualification needs paired, isolated, counterbalanced agent trials
against competent Git/rg workflows, preserved task success, 30% less median time
to verified evidence and 14 elapsed days of reviewed use. No such result or
commitmux comparison has been collected. CUDA remains explicitly unqualified.
