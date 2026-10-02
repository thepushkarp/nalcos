# Embedding evaluation

No embedding default is release-qualified. The checked-in pilot contains 12
retrospective questions from one 111-commit repository. It is useful for finding
integration mistakes and forming hypotheses. It supplies no independent held-out
evidence, Linux confirmation, large-repository scaling measurement, or measured
advantage over an agent using Git and `rg`.

A separate three-repository diagnostic below measures the Rust CLI's lexical
indexing and search. It still supplies no independent held-out model qualification
or measured agent-workflow advantage.

## Preserved exploratory evidence

[`benchmarks/pilot/2026-10-02`](../benchmarks/pilot/2026-10-02/) contains the exact
corpus, sanitized results and source scripts, a JSONL query manifest, artifact
hashes, and a failing release-qualification report. The corpus is HEAD ancestry at
`57bd49f174811f6bbaf4e541019f9a7e9f34a98e`: 111 commit-message documents and 290
signed textual diff chunks. Merge diffs, binary bodies, and bundled model/cache
trees are excluded. The corpus SHA-256 is
`190f64c5e38719221c6e0b84809ead5e1041a9e6ca1b004c0b264185cc5ab491`.

The artifact manifest records original and sanitized hashes. Optional vectors,
server logs, and weights are omitted from Git; their identities are preserved by
hash. Personal absolute paths in archived scripts/results are placeholders.
`replay_pilot.py` exports the archived scripts to a chosen directory and substitutes
local paths. It preserves the exploratory method, including its documented
token-count metadata bug. The corrected counts are in the archived results.

```sh
python3 benchmarks/replay_pilot.py --output /tmp/nalcos-pilot-replay --repo .
python3 benchmarks/evaluate.py qualify \
  --manifest benchmarks/pilot/2026-10-02/queries.jsonl \
  --candidates benchmarks/pilot/2026-10-02/candidates.json
```

The second command intentionally exits **3**: no candidate qualifies. Invalid
inputs exit **2**; a successful evaluation exits **0**. Scripts print final JSON
to stdout and diagnostic progress to stderr.

The original measurements used an Apple M2 MacBook Air with 16 GiB RAM, four CPU
threads, document batches of eight, three query warmups, and 36 query samples.
This table reproduces the original message-plus-diff semantic retrieval results.
Recall is retrieval of one known commit per question; other relevant commits can
be unlabeled.

| Artifact/runtime | Weights MiB | Warm query p50 ms | Documents/s | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniLM FP32 / ONNX CPU | 86.2 | 2.35 | 42.06 | 10/12 | .5810 |
| MiniLM INT8 / ONNX CPU | 22.0 | 1.14 | 81.70 | 11/12 | .6250 |
| BGE-small FP32 / ONNX CPU | 126.9 | 5.95 | 19.43 | 10/12 | .5580 |
| Jina code FP32 / ONNX CPU, 512 tokens | 611.8 | 17.72 | 3.94 | 12/12 | .6807 |
| Jina code INT8 / ONNX CPU, 512 tokens | 154.4 | 10.46 | 5.80 | 12/12 | .6972 |
| Jina code INT8 / ONNX CPU, 1024 tokens | 154.4 | 7.15 | 9.68 | 12/12 | .7014 |
| Qwen3 0.6B Q8 / llama.cpp CPU | 609.5 | 49.68 | 1.54 | 12/12 | .6764 |
| Qwen3 0.6B Q8 / llama.cpp Metal | 609.5 | 34.92 | 5.40 | 12/12 | .6722 |
| EmbeddingGemma Q8 / llama.cpp CPU | 318.1 | 13.54 | 6.70 | 12/12 | .6508 |
| EmbeddingGemma Q8 / llama.cpp Metal | 318.1 | 12.65 | 35.31 | 12/12 | .6508 |

These are runtime-specific observations. ONNX and llama.cpp use different
execution paths; GGUF requests include loopback HTTP. Query time excludes Git
extraction, index lookup, ranking, and CLI startup. Weight size is neither installed
size nor peak RAM. The OS cache was not cleared. CPU RSS and Metal memory
accounting are different. Machine load and thermal state were not controlled;
the separate, faster Jina 1024-token run does not establish that increasing the
context limit improves performance. MiniLM/BGE truncate one pilot document at
512 tokens; Jina truncates eight. The 1024-token Jina run preserves them all.

Message-only embeddings, message-plus-diff embeddings, a basic SQLite FTS5
baseline, and equal-weight RRF are all recorded. Naive FTS/RRF results are not a
competent Git-using agent baseline. Ranking uses exact vector scans and the best
matching document per commit; these are not ANN recall measurements.

## Matched original-weight validation

Use the original pinned safetensors for MiniLM and Jina, convert to **F32 first**,
and compare with their matching upstream ONNX export. Both use masked mean
pooling, L2 normalization, and no query prefix. MiniLM has 384 dimensions and a
512-token limit. Jina has 768 dimensions and supports longer contexts; the matched
pilot fixes its limit at 512, with truncation recorded explicitly.

| Model | Hugging Face revision | Status |
| --- | --- | --- |
| `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | `b207367332321f8e44f96e224ef15bc607f4dbf0` | FP32 GGUF CPU and Metal tokenizer/vector checks pass |
| `jinaai/jina-embeddings-v2-base-code` | `516f4baf13dec4ddddda8631e019b5737c8bc250` | FP32 GGUF CPU tokenizer matches; vector parity fails; GGUF timings blocked |

Conversion used llama.cpp `5266f24da`, matching installed build 10809. Conversion
provenance records source-weight, tokenizer, converter-tree and output hashes.
Jina's original safetensors contain FP16 tensors; F32 conversion expands those
stored values and cannot recover unavailable precision. Its converted F32 weights
also exceed the 500 MiB default size budget.

The parity check tests empty/whitespace text, prose, code, Unicode, literal special
tokens, a truncation boundary, and corpus samples. It requires exact native token
IDs, finite nonzero normalized vectors, minimum cosine .9999, maximum absolute
component error .002, and batch invariance. These thresholds are explicit smoke
validation tolerances, not a guarantee of retrieval equivalence on every input.
The tool sends identical reference token IDs to the GGUF embedding endpoint to
separate graph/pooling differences from tokenizer differences. Native raw-text
tokenization must pass as well.

MiniLM achieved minimum cosine above .999998 on both backends. Jina's CPU check
had mean cosine .1503 and minimum .0218 against ONNX despite exact tokens. Do not
relax thresholds to admit that result, infer speed from it, or quantize it as a
qualified baseline. Further investigation must compare original-framework
outputs, exporter weights and native graph operations before attributing the
failure to one component.

The matched MiniLM measurements and conversion/parity reports are preserved in
[`benchmarks/matched/2026-10-02`](../benchmarks/matched/2026-10-02/). All three runs
used the same 401 documents, query inputs, 512-token limit, four threads and batches
of eight. Each indexed the full corpus three times:

| MiniLM original weights | Median indexing seconds | Documents/s | Warm query p50 ms |
| --- | ---: | ---: | ---: |
| FP32 GGUF / CPU | 7.18 | 55.84 | 15.04 |
| FP32 GGUF / Metal | 3.08 | 130.08 | 3.85 |
| FP32 ONNX / CPU | 11.10 | 36.13 | 2.46 |

All three retrieve the same known gold ranks: Recall@10 10/12, MRR .5810. Across
every document and query, minimum GGUF/ONNX cosine exceeds .999996. Metal has the
highest indexing throughput here; ONNX has the lowest warm single-query latency.
GGUF includes HTTP overhead. Backend order was fixed, and cache/thermal effects
were not instrumented, so repeat with counterbalanced order before making a
platform policy. This experiment is still exploratory and cannot qualify a default.

The native Rust smoke comparison is separately recorded in
[`native-gguf-diagnostic.json`](../benchmarks/matched/2026-10-02/native-gguf-diagnostic.json).
MiniLM CPU/Metal vectors pass the cosine threshold against ONNX. EmbeddingGemma Q8
does not pass the same threshold against the saved server reference; that
comparison changes llama.cpp build and batch/context shape, so it does not locate
the cause. The native test binary identity was not preserved. Treat these as
diagnostics requiring reproduction, not release qualification.

Licensing is a separate gate. Jina's pinned model card declares Apache-2.0. The
checked MiniLM card has no explicit model-license field or license file; the
license of the sentence-transformers library does not establish the license of
these weights. Record reviewed evidence before distributing any default weights.

### Reproduce locally

Use the standard shared Hugging Face cache. `hf download` prints the snapshot path;
do not copy weights into the repository or set a project-specific cache. Download
is an explicit preparation step. Benchmark and conversion commands run offline.

```sh
hf download sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
  model.safetensors config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json vocab.txt modules.json sentence_bert_config.json \
  1_Pooling/config.json onnx/model.onnx README.md \
  --revision b207367332321f8e44f96e224ef15bc607f4dbf0

hf download jinaai/jina-embeddings-v2-base-code \
  model.safetensors config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json vocab.json modules.json sentence_bert_config.json \
  1_Pooling/config.json onnx/model.onnx README.md \
  --revision 516f4baf13dec4ddddda8631e019b5737c8bc250
```

Obtain the reviewed llama.cpp source at the runtime's recorded commit. Install its
conversion dependencies in an isolated environment. The checked conversion used
Python 3.12, torch 2.11.0, transformers 4.57.6, NumPy 1.26.4, sentencepiece below
0.3, and protobuf below 5; the checkout supplies `gguf-py`. Set `NALCOS_SNAPSHOT`
to the printed pinned snapshot path and `NALCOS_LLAMA_SOURCE` to that source tree:

```sh
python benchmarks/prepare_gguf.py \
  --snapshot "$NALCOS_SNAPSHOT" \
  --model-id sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
  --revision b207367332321f8e44f96e224ef15bc607f4dbf0 \
  --llama-source "$NALCOS_LLAMA_SOURCE" --llama-revision 5266f24da \
  --output /tmp/nalcos-bench/minilm-f32.gguf

uv run --python 3.13 benchmarks/run.py --mode check \
  --snapshot "$NALCOS_SNAPSHOT" \
  --model-id sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
  --revision b207367332321f8e44f96e224ef15bc607f4dbf0 \
  --gguf /tmp/nalcos-bench/minilm-f32.gguf --backend cpu \
  --output /tmp/nalcos-bench --label minilm-f32-cpu-check

uv run --offline --python 3.13 benchmarks/run.py --mode benchmark \
  --snapshot "$NALCOS_SNAPSHOT" \
  --model-id sentence-transformers/multi-qa-MiniLM-L6-cos-v1 \
  --revision b207367332321f8e44f96e224ef15bc607f4dbf0 \
  --gguf /tmp/nalcos-bench/minilm-f32.gguf --backend cpu \
  --parity-report /tmp/nalcos-bench/minilm-f32-cpu-check.json \
  --output /tmp/nalcos-bench --label minilm-f32-cpu
```

The first `uv run` can prepare the script's pinned Python dependencies. The script
does not download models or execute model-supplied Python code. Use a fresh label
and `--backend metal` for the Metal check, then its passing report for Metal timing.
Omit `--gguf` and `--parity-report` for the matching ONNX CPU run. Keep corpus,
token limit, threads, batch size, prefixes and pooling fixed. A passing report is
bound to the artifact hash, revision, backend, token limit and pooling. Quantized
artifacts require their own parity tolerances and quality comparison after the
original FP32 path is correct; no quantized GGUF default is qualified here.

Run only one model process at a time. Pause compilers and other inference workloads.
Record AC/battery mode, thermal state, runtime build and accelerator placement.
Measure cold startup separately from warm requests, record every sample and
truncation count, and repeat full-corpus indexing at least three times. Alternate
backend/model order across fresh sessions to expose cache and thermal effects.
For a release decision add Linux CPU, sustained load, a large history, incremental
sync, and Rust CLI end-to-end measurements. Do not translate documents/second into
commits/second without measuring the repository's document expansion.

## Rust CLI history-scale diagnostic

[`benchmarks/scale/2026-10-02`](../benchmarks/scale/2026-10-02/) preserves aggregates
and content hashes. Raw queries, other repositories' source, gold adjudication,
local paths and CLI responses remain outside the checkout. The two larger local
histories were selected by repository inventory. Counts below use each frozen
HEAD's ancestry; they do not count unrelated refs.

| History | Commits | Indexed documents | Source omissions | Initial lexical sync seconds |
| --- | ---: | ---: | ---: | ---: |
| NaLCoS | 111 | 898 | 3 | 1.99 |
| vLLM | 14,193 | 257,525 | 1,461 | 204.50 |
| scaffold-eth NFT challenge | 829 | 15,684 | 57 | 29.67 |

These are single extraction/indexing runs on the same M2 machine, with no
embeddings. History coverage reports every eligible commit visited. Recorded
omissions mean some source bodies are unavailable or excluded; complete history
coverage is not complete source-content coverage. These production documents are
different from the 401-document exploratory model corpus.

Thirty questions were locked before matching diffs or model rankings were
inspected for this diagnostic. Twenty came from historical public issue reports;
ten came from repository documentation. The locked manifest SHA-256 is
`18af39c123e5185892b2bc01f99f01ee5512af7097e5b5deece48d9b40970237`.
Gold review established **21 positives, two scoped negatives and seven unjudged
questions**. Every positive commit is reachable from its frozen snapshot; parent,
path and patch hashes accompany local adjudication. Unmerged proposed fixes and
stale issue closures were not accepted as gold. Unjudged questions are excluded
from relevance denominators rather than counted as negative answers.

This remains retrospective diagnostic evidence. The same researcher wrote and
adjudicated questions, NaLCoS history had already been explored, historical issue
bodies may include later edits, and relevant commits beyond the known gold can be
unlabeled. The 20 development / 10 diagnostic-heldout assignment remains frozen,
but **zero questions are certified as genuine independent holdout**.

The initial release-binary lexical run used offline, cached searches, a fixed
snapshot ref, limit 10 and a 30-second timeout. All 30 returned verified evidence
without command failure. These timings include process startup, scope checks,
SQLite search and exact Git verification.

| History | Query p50 seconds | Query p95 seconds | Known-gold Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| NaLCoS | .143 | .162 | 4/8 | .1844 |
| vLLM | .882 | 3.275 | 0/6 | .0000 |
| scaffold-eth NFT challenge | .198 | .383 | 3/7 | .2381 |

Overall known-gold Recall@10 is 7/21 and MRR@10 is .1496. The two scoped negatives
received candidate lists, with no evidence-backed absence conclusion. Each
percentile uses linear interpolation over ten different queries, one observation
per query; it is not a repeated-run tail estimate. Cache, compilation load and
thermal conditions were not controlled. This baseline's binary SHA-256 is
`03130d74692dd33076e8fc0e6f496b9d64b8588667948cbfc6e8bbc9f0c2403c`.
Later optimization results must identify their own binary and retained inputs.

The SQL performance comparison alternated old/new binaries on two slow
**development** queries, five observations per query per binary. Median full CLI
times changed from .589 to .398 seconds and from 1.453 to .595 seconds. Every
commit, evidence identifier and score matched. The separate 30-query replay also
preserved all result identities and scores; its timing varied substantially with
cache/load, so it establishes functional equivalence rather than a stable speedup.
The sanitized paired samples and binary hashes are in
[`lexical-sql-comparison.json`](../benchmarks/scale/2026-10-02/lexical-sql-comparison.json).

A native ONNX Runtime 1.23.2 run covered all 898 NaLCoS documents using the pinned
MiniLM FP32 profile. All ten queries completed in each mode, with startup and Git
verification included:

| Mode | Query p50 seconds | Known-gold Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| Lexical | .224 | 4/8 | .1844 |
| Semantic | .359 | 5/8 | .2542 |
| Hybrid | .367 | 5/8 | .2052 |

This is one sample per query, one model and one small history. The eight positive
questions cover only five distinct gold commits; the other two are scoped
negatives. The results do not establish the best model or fusion policy. Preserve
the frozen diagnostic set and use development questions for further tuning.
[`nalcos-native-matrix.json`](../benchmarks/scale/2026-10-02/nalcos-native-matrix.json)
records model generation, runtime, coverage and binary identity.

The NFT history has a separate native **ARM64 INT8 ONNX** MiniLM run on all 15,684
documents / 39,909 vector chunks. Its ten locked questions contain seven positives
and three unjudged labels. All 30 searches returned ten commit IDs with verified
evidence, using the same active generation and ONNX Runtime 1.23.2 CPU backend.

| Mode | Query p50 seconds | Query p95 seconds | Known-gold Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| Lexical | .191 | .738 | 3/7 | .2381 |
| Semantic | .359 | .886 | 6/7 | .3333 |
| Hybrid | .422 | .649 | 5/7 | .4776 |

These are shared-host single observations, including startup and source checks.
They do not compare quantization fairly with the separate NaLCoS FP32 run.
Two hybrid responses clipped commit messages at 4 KiB; their ranked IDs and
verified evidence remained intact. The original assessment and raw responses are
preserved; the recorded re-assessment separates this presentation limit from
retrieval completeness without rerunning inference. The nine source-verification
warnings belong to the **frozen pre-fix binary and index** recorded in this matrix.
The source audit found one indexed hunk with incorrect line coordinates; its blobs
were available and verification correctly omitted that evidence. Corrected
extraction validates all 34 documents for the offending commit, but these scores
still retain the pre-fix limitation and 57 recorded source omissions. Existing
indexes need the [canonical repair through `nalcos sync`](migration.md) after
upgrading; this frozen matrix does not measure the repaired index.

[`nft-native-int8-matrix.json`](../benchmarks/scale/2026-10-02/nft-native-int8-matrix.json)
records the model artifact, binary, immutable label hashes and per-mode diagnostic
files. Raw questions, ranked commit identities and source content stay local.
[`native-full-corpus-lifecycle.json`](../benchmarks/scale/2026-10-02/native-full-corpus-lifecycle.json)
records resumable indexing and maintenance, including timeout and pause segments;
its indexing times were measured under shared development load.

The separate [canonical repair smoke](../benchmarks/scale/2026-10-02/canonical-repair-smoke.json)
replaces one malformed source record, preserves 15,683 other records and 39,907
vector chunks byte-for-byte, and verifies the repaired evidence through the
installed CLI. A repeat sync embeds no documents. This is a lifecycle check,
not a replacement retrieval-quality evaluation.

Run a matrix against already prepared indexes with a local mapping of `repo_key`
to `path`, `config`, and `data_dir`. The adapter never syncs an index or downloads
a model; it supports frozen `find` tasks. Regression experiments need an explicit
range-aware collection driver. Use a new output directory for every run:

```sh
python3 benchmarks/cli_matrix.py --binary target/release/nalcos \
  --manifest /tmp/nalcos-eval/questions.adjudicated.jsonl \
  --repositories /tmp/nalcos-eval/repositories.json \
  --output /tmp/nalcos-eval/run-1 --modes lexical semantic hybrid
python3 benchmarks/evaluate.py metrics \
  --manifest /tmp/nalcos-eval/questions.adjudicated.jsonl \
  --rankings /tmp/nalcos-eval/run-1/lexical.rankings.jsonl
```

Each raw response retains the active model generation, runtime, scope and separate
history/embedding coverage. Fallback, partial coverage, changed snapshot,
changed encoder generation/runtime, unverified returned evidence, timeout or an
unknown truncation flag makes retrieval incomplete. Output clipping is reported
separately and does not invalidate ranked commit IDs backed by verified evidence.
No-match retrieval never becomes a correct negative automatically.
Progress goes to stderr; final aggregate JSON goes to stdout. Multi-repository
semantic and hybrid qualification remains pending further measurements.

## Synthetic 100,000-commit lexical measurement

[`2026-10-02-lexical-100k.json`](../benchmarks/scale-results/2026-10-02-lexical-100k.json)
records a controlled query window on an Apple M2 with 16 GiB RAM, Git 2.55.0 and a
release build of `nalcos 2.0.0-alpha.1`. The generated history has 100,000 linear
commits and 128 rotating paths; each commit changes one small UTF-8 Rust file.
It contains no merges, renames, binaries or large patches. Indexing produced
300,000 documents, no source omissions and **zero embedding vectors**.

| Measurement | Observation |
| --- | ---: |
| Full CLI lexical search p50 | .828 seconds |
| Full CLI lexical search p95 | 1.069 seconds |
| Maximum, including the first search | 2.295 seconds |
| Search observations | 20 |
| Index storage | Approximately 576 MiB |
| Status command | .970 seconds |
| Initial indexing, with concurrent workloads | 1,484.80 seconds |

Every query used a new process with cached freshness, limit 10 and a 16 KiB
evidence budget. Timing includes startup, live Git scope resolution, SQLite
retrieval, source verification and JSON output. Queries mix ordinary terms and
exact generated identifiers, including some repeated terms; they are not an
independent natural-language relevance set. No response reported truncated output.
The p95 uses **nearest rank, `ceil(.95 × 20)`**, unlike the linear interpolation
in the three-repository diagnostic above.

Other agents paused builds, indexing and inference during the query window;
ordinary host activity was not controlled. Filesystem/page caches were not
flushed. The first query is included, there were no omitted warmups, and later
queries may benefit from cached data. The indexing run overlapped development
builds/tests and native MiniLM indexing in another repository, so its elapsed
time does not establish isolated ingestion throughput.

Index construction and queries used different release binaries; both hashes are
in the report. The query binary SHA-256 is
`0298b1390ec55a3f58da1d2f7f4ce2cb29d6e4b83ddf9aef81f71718d13fca11`.
This experiment establishes lexical scale behavior for this simple generated
history. It does not establish real-history retrieval quality, semantic/hybrid
latency, Linux portability, an agent-workflow advantage or default-model
qualification.

## Synthetic 100,000-commit exact semantic measurements

[`2026-10-02-semantic-100k.json`](../benchmarks/scale-results/2026-10-02-semantic-100k.json)
records three 20-invocation runs on copies of the generated history above. Its
300,000 documents receive deterministic normalized 384-dimensional vectors
(460.8 MB). **These vectors do not encode document meaning; no retrieval-quality
claim is valid.** Query encoding uses pinned MiniLM FP32, native ONNX Runtime
1.23.2, CPU and four threads on the same M2/16 GiB reference host. Blocked
subprocess hardware probes remain marked unknown in the reports.

| Exact-search implementation | Cached CLI p50 seconds | Observed p95 seconds | Maximum seconds |
| --- | ---: | ---: | ---: |
| SQL-ordered vectors | 3.875 | 11.280 | 12.568 |
| Unordered scan, borrowed-byte scoring and explicit ties | 3.445 | 11.182 | 11.800 |
| Same binary, covering document index and planner statistics | 2.328 | 2.822 | 7.015 |

Timing includes process/model startup, Git scope, retrieval, verified sources and
JSON output, with limit 10 and a 16 KiB evidence budget. Every ordered result,
score and coverage object matches across all 20 paired inputs. Reports preserve
binary/model identities and every sample. Real repository/model indexes were
opened read-only.

Runs were sequential, retained the first query and did not flush filesystem
caches. Other indexing, inference and builds were paused; ordinary host activity
and thermal conditions were uncontrolled. The nearest-rank p95 is observational,
not a causal speedup estimate. **The two-second target remains unmet.** Cached
results do not establish default automatic-freshness latency, and one vector per
short document underrepresents real histories requiring multiple chunks.

[`2026-10-02-semantic-sql-diagnostics.json`](../benchmarks/scale-results/2026-10-02-semantic-sql-diagnostics.json)
contains query plans, SQL comparisons and diagnostic phase profiles. An unchanged
history's pending check returned the same empty result in .124–.226 seconds with
ID-only lookup, versus .500–2.922 seconds scanning source records. In the final
three-invocation profile, automatic refresh took .212 seconds and the complete
auto search took 2.414 seconds, with unchanged results. These few observations
are not percentiles. Full embedding statistics changed the scan plan without a
material warm SQL gain; that experiment is not the production maintenance policy.
Power observations apply only at their recorded timestamps.

The resumable harness copies a completed lexical-only `benchmarks/scale.py`
fixture and uses an existing native MiniLM generation as query-encoder metadata:

```sh
python3 benchmarks/vector_scale.py \
  --fixture /tmp/nalcos-scale-100k --work /tmp/nalcos-vector-scale \
  --model-index "$NALCOS_MODEL_INDEX" --ort-library "$NALCOS_ORT_LIBRARY" \
  --binary target/release/nalcos --queries 20 --freshness cached \
  --output /tmp/nalcos-vector-scale/cached.json
```

Use `--resume --prepare-only` after interrupted preparation. Use `--measure-only`
with a new output filename for another run; `--freshness auto` measures that mode
separately. Synthetic indexes carry an explicit warning and are never evidence of
semantic relevance.


## Frozen evaluation manifests

Each JSONL manifest row contains `id`, public `repo` identifier, full `snapshot`
object ID, `query`, `split` (`exploratory`, `development`, `heldout`), `origin`
(`prospective`, `retrospective`), `frozen_before_tuning`, `task` (`find`, `regression`),
`answerable`, and `gold_commits`. Optional `gold_paths` aids review. A negative has
`answerable: false` and an empty gold list. An unresolved diagnostic uses
`label_status: "unjudged"`, `answerable: null`, and an empty gold list; it is excluded
from positive/negative metrics and blocks qualification when it belongs to the
declared evaluation cohort. Regression tasks additionally require
full `good` and `bad` object IDs. Record questions before investigating their
answers; freeze the development/holdout assignment before choosing a model,
prompt, chunker or fusion policy. The evaluator cannot prove these attestations.

`origin` describes collection timing; it does not certify independence. A genuine
historical user question can qualify when its original wording and provenance are
reviewed. A prompt authored from a known commit or answer cannot qualify. Each
qualifying held-out row requires a `provenance` evidence object with `passed`,
nonempty `references`, and these fields:

```json
{
  "source_kind": "historical_user_question",
  "answer_independent": true,
  "locked_before_answer_inspection": true,
  "locked_before_ranking": true,
  "isolation_reviewed": true
}
```

`source_kind` may also be `prospective_user_question`. The review references must
substantiate the original source, lock and isolation; for historical sources,
retain the original revision or timestamp and exclude later solution text. Each
row also needs a reviewed `label_review` object with `passed` and `references`.
Negatives additionally require `label_review.absence_reviewed: true`. Neither an
empty candidate list nor an unresolved search establishes a negative label.

Each rankings row contains `query_id`, `snapshot`, `query_sha256` (UTF-8 query bytes),
ordered unique full `commits`, `complete`, and `abstained`. The evaluator rejects
stale query/snapshot/range identities, duplicate IDs and duplicate ranked commits.
`complete` means the configured retrieval run completed successfully; it does not
claim exhaustive semantic relevance or that an empty candidate list proves
absence. Native CLI history/embedding coverage and output/ANN limits must be
checked before an adapter sets it true.

```sh
python3 benchmarks/evaluate.py metrics \
  --manifest benchmarks/pilot/2026-10-02/queries.jsonl \
  --rankings /tmp/nalcos-bench/minilm-f32-cpu.rankings.semantic_message_diff.jsonl \
  --split exploratory
python3 -m unittest discover -s benchmarks -p 'test_*.py'
```

Metrics are macro Recall@1/5/10 across all labeled gold commits, Hit@1/5/10 for any
gold, and MRR for the first relevant commit. Missing positive responses score zero
and are marked incomplete. Negatives are reported separately; a missing or partial
response is never a correct abstention. Empty output alone is not an abstention.

## Default selection and release gates

`evaluate.py qualify` consumes a manifest and a candidates JSON object containing
`candidates` and `evaluation_cohort`. Each candidate has `id`, `weight_bytes`, a relative `rankings` JSONL
path, `cpu_index_seconds` (at least three full-corpus samples), `timing_cohort`, and
`license`, `correctness`, `cpu_portability` evidence objects. An evidence object
has `passed: true` only after review and nonempty `references`; CPU portability
also lists tested `platforms`, including `darwin-*` and `linux-*`. Use a cohort
fingerprint covering the identical corpus, chunking, runtime family, CPU backend,
thread count, batching and timing method. A scalar flag is not independent proof;
the report states that attestations require review.

`evaluation_cohort` declares the exact development and held-out membership with
`query_ids`, `query_identities_sha256`, `frozen_before_tuning: true`,
`frozen_before_ranking: true`, and reviewed `passed`/`references` evidence.
`evaluate.cohort_digest(rows)` computes the digest from the selected rows' IDs,
repository identifiers, snapshots, question text, splits, task kinds and regression
ranges. Gold labels are adjudicated afterward and are not part of this lock.
An unknown or duplicate member is invalid; a missing review or mismatched digest
blocks qualification. Exploratory and unrelated unjudged rows can be excluded
only through this predeclared, reviewed membership. Removing difficult questions
after inspecting rankings invalidates the lock. Missing declaration never
silently narrows the cohort.

The selection rule is fixed before looking at holdout:

1. Require at most 500 MiB of weights and verified license, correctness and CPU
   portability on macOS and Linux.
2. Require development Recall@10 within five percentage points of the best
   development result, and MRR within .03 of the best. The quality frontier includes
   larger/slower measured candidates to avoid biasing it toward eligible defaults.
3. Among qualifying candidates, minimize median CPU indexing time. Candidates
   within 10% of the fastest time are tied; choose the smaller artifact.
4. Freeze that candidate and the strongest development quality reference. Confirm
   the choice on at least **30 genuine, reviewed, locked held-out questions across
   three repositories**, separate from development. Adjudicated positives and
   verified negatives count toward this quota. Recall/MRR use positive questions
   only and must be defined; negative outcomes are reported separately. Use the
   same quality tolerances for confirmation. A failed
   confirmation blocks release; it never selects a runner-up using holdout scores.

The schema-v2 report separates `development_choice`, `model_qualified`,
`product_qualified` and `release_qualified`. Model qualification requires the
model prerequisites, development selection and held-out confirmation above.
Product qualification uses the workflow and dogfood evidence below. Release
qualification requires both; each has explicit blockers. `qualify` exits 3 until
release qualification passes. Archived reports retain their recorded schema,
policy and labels and must not be overwritten or silently reclassified.

The product gate requires a reviewed, predeclared sample of paired held-out agent
tasks, the same agent/model and resource budgets, unchanged success rate or better, and
at least **30% lower median time to verified evidence** with NaLCoS plus Git/`rg`
than Git/`rg` alone. The workflow protocol declares its exact sample and explains
why that sample is meaningful; it does not inherit the retrieval cohort's
30-question quota. All declared pairs must be present and comparable. Existing
eligible held-out questions may be reused with isolated sessions.

Dogfood must span at least **two weeks (14 elapsed days)**, measured from the
earliest to latest observed trial start. Active UTC dates are reported separately;
daily activity is not required. The review must substantiate meaningful use over
that period; two timestamps alone are not proof. Supply the reviewed workflow
report as `workflow_evaluation`, including `passed`, `references`, `protocol`,
`same_agent`, `independent_heldout`, `paired_questions`, `success_rate_delta`,
`median_time_reduction`, `dogfood_span_days` and `dogfood_active_days`. Missing
product evidence does not erase a model qualification, but blocks release.

For the agent comparison, randomize task order and isolate runs so answers do not
leak between conditions. Allow a competent baseline `git log --grep`, `git log -S/-G`,
`git show`, `git blame` and `rg`; charge indexing separately and report both first-use
and amortized costs. Fix a time/token budget and count failures, timeouts and
incorrect evidence, instead of timing only successful runs. Review final evidence
blind to the tool condition. Measure tool calls and tokens as secondary outcomes.

`workflows.py` evaluates paired trial JSONL collected by a local agent harness or
recorded from audited agent sessions. It runs no agents, installs no tools and
makes no network calls. Each row has `query_id`, `snapshot`, `query_sha256`,
`condition` (`git_rg`, `nalcos_git_rg`, or another comparator), `agent_fingerprint`,
`isolation_id`, `budget_seconds`, `budget_tokens`, `elapsed_seconds`, `tokens_used`,
`outcome` (`completed`, `timeout`, `failed`), `order` (0 or 1 within the pair), and
timezone-qualified `started_at`. Regression rows also bind `good` and `bad`.

The `--protocol` JSON object contains `query_ids`, `query_identities_sha256`,
`frozen_before_trials: true`, a nonempty `sample_rationale`, and reviewed
`passed`/`references` evidence. It uses the same identity digest as the evaluation
cohort. Review must establish sample adequacy and that membership was fixed before
trial outcomes were observed. Its rows must belong to the requested split.
Without a reviewed protocol the adapter still reports diagnostic metrics, but
cannot pass product qualification.

An `evidence_review` object contains `passed`, `blind_to_condition`, `reviewer_id`,
`references`, and `verified_commits`. A correct negative additionally needs
`absence_reviewed: true`; empty tool output alone is insufficient. The agent
fingerprint must cover the exact agent/model/prompt configuration. Enforce budgets
in collection and keep transcripts for review. Matching metadata cannot prove
isolation, budget enforcement or independent judging.

The adapter rejects mismatched query identities, excludes incomparable pairs,
counts unverified or over-budget answers as failures, and charges failures at
least their full time budget. It reports the ratio of median charged times as
well as success rates and token usage. `dogfood_span_days` measures elapsed time;
`dogfood_active_days` counts distinct observed UTC dates. Order must be
counterbalanced, both conditions use the same agent and
budgets, and sessions must have different isolation IDs.
Do not reuse a session across questions; the adapter rejects repeated isolation
IDs spanning different questions.

```sh
python3 benchmarks/workflows.py \
  --manifest /tmp/nalcos-eval/questions.jsonl \
  --trials /tmp/nalcos-eval/agent-trials.jsonl \
  --protocol /tmp/nalcos-eval/workflow-protocol.json \
  --candidate nalcos_git_rg --output /tmp/nalcos-eval/workflow.json
```

Use a separate paired collection with `--candidate commitmux_git_rg` for commitmux.
No competent Git/`rg` agent baseline or commitmux comparison has been collected
here. Their checked-in status reports explicitly say unavailable and block the
workflow gate. The lexical table above supplies no evidence of a 30% agent
speedup. A third-party comparator needs its exact installation/version, indexing
scope and model setup recorded before comparison; an unavailable comparator is
not a measured loss.

For regressions, provide only the known-good/known-bad range and the symptom
observed before the fix. Exclude later fix messages, issue answers and future
commits from the indexed snapshot. Gold labels require a reproducible check or
independent review. Retrieval identifies candidate changes with exact Git evidence;
it does not establish causality. Include reverts, merge-parent ambiguity, renames,
unanswerable questions and plausible opposite changes. Keep retrieval evaluation
separate from agent explanation quality.

## Sources

- [MiniLM model and usage](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)
- [Jina code model and license declaration](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)
- [BGE-small model instructions](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Qwen3 embedding GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)
- [Shared Hugging Face cache](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache)
- [Pinned llama.cpp converter](https://github.com/ggml-org/llama.cpp/blob/5266f24da/convert_hf_to_gguf.py)
