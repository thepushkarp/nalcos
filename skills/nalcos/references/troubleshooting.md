# Troubleshooting

Errors arrive as `{"command":"error","error":{"code","message","action"}}` on stdout with
`--json`. Follow `action` when it is present. Rerun with `-v` to print the repository and index
paths on stderr.

## Error codes

| Code | Exit | Cause | Fix |
| --- | --- | --- | --- |
| `invalid_input` | 2 | A bad argument: empty query, `--limit` outside 1–100, a range other than `A..B`, an unknown revision, a malformed date, or `--since` later than `--until`. | Fix the argument named in `message`. Check revisions with `git rev-parse --verify REV`. |
| `invalid_config` | 2 | The configuration file is invalid, or `--model profile:NAME` names a missing profile. | Fix the TOML, or select another file with `--config`. |
| `unknown_model` | 2 | `--model` is not a registered alias, Hugging Face ID, or `profile:NAME`. | Use a registered alias such as `coderank` or `minilm`. |
| `unsupported_schema` | 2 | The index was written by an incompatible nalcos version. | Use the matching nalcos version. The index is derived data; with the user's approval, remove the index directory shown by `-v` and rebuild. |
| `unsupported_git` | 1 | Git is older than 2.45 or lacks `--no-lazy-fetch`. | Install Git 2.45 or later first on `PATH`. |
| `git_error` | 1 | Git failed, for example outside a Git checkout or on `safe.directory` ownership checks. | Read `message`; change into the checkout, pass `--repo PATH`, or fix the Git problem it names. |
| `semantic_unavailable` | 1 | `--mode semantic` or `status --check` without an active embedding model. | Use `--mode lexical`, or ask before running `nalcos init`. If `action` names a pending generation, run `nalcos sync`. |
| `model_required` | 1 | `--revision`, `--variant`, or `--reembed` without a model to modify. | Pass `--model ALIAS` too. |
| `model_not_cached` | 1 | Model files are missing, often under `--offline` or after the cache was cleared. | Run `nalcos init` or `nalcos sync` with network access (ask first). |
| `encoder_contract_changed` | 1 | Cached model files or the profile no longer match the index. | Run `nalcos sync` to build matching embeddings. |
| `provider_changed` | 1 | An API embedding provider now returns different vectors. | Pin the provider deployment, then `nalcos sync --reembed`. |
| `device_unavailable` | 1 | A requested accelerator cannot run the model. | Retry with `--device cpu` or `--device auto`. |
| `explicit_device_unavailable` | 1 | An explicit `--device` failed. | Use `--device auto` or `--device cpu`. |
| `unsupported_cpu` | 1 | The `minilm-int8` model needs ARM64 or x86_64 with AVX2. | Use `--model minilm`. |
| `offline` / `offline_provider` | 1 | `--offline` blocks an API embedding provider. | Drop `--offline` or search with `--mode lexical`. |
| `index_busy` | 1 | Another nalcos process holds the index writer lock. | Wait and retry, or search with `--freshness cached`. |
| `io_error` | 1 | A file operation failed, for example writing the index from a read-only sandbox with `--mode lexical`. | Search with `--freshness cached`, or set `NALCOS_DATA_DIR` to a writable directory. |
| `evidence_not_found` | 1 | The evidence ID is not in this repository's index. | Pass the exact `id` from a search of the same repository; IDs of pruned commits disappear after `sync`. |
| `source_unavailable` | 1 | The evidence's Git objects are no longer available locally. | Search again; history was rewritten or objects were removed. |
| `repository_changed` | 1 | Refs kept moving during the search. | Retry after the concurrent Git operation finishes. |
| `missing_object` | 1 | `HEAD` or a needed object is missing locally (partial or shallow clone). | Fetch or deepen the history with Git, then `nalcos sync`. |
| `timeout` | 124 | `--timeout` elapsed. | Rerun with a larger budget; completed batches are kept. |
| `interrupted` | 130 | Ctrl-C. | Rerun; completed batches are kept. |

## Symptoms without an error

| Symptom | Meaning | Fix |
| --- | --- | --- |
| `mode_used` is `lexical` although hybrid was requested | No usable embedding model; a warning says why. | Accept lexical results, or ask before `nalcos init`. |
| Warning `Incremental indexing is incomplete` | `auto` freshness ran out of its two-second budget. | `nalcos sync --timeout 10m`, or `--freshness wait --timeout 5m`. |
| Warning `Another nalcos process is indexing this repository` | Results come from the existing index while another process writes it. | Wait for it; `nalcos --json status` reports `writer_active`. |
| Warning `No index is available` | No index exists yet, typically with `--freshness cached`. | Search once with `auto` or `wait`, or run `nalcos init`. |
| Warning `Semantic indexing unavailable: …` | Index updates failed, for example on a read-only data directory. | Set `NALCOS_DATA_DIR` to a writable directory, or use `--freshness cached`. |
| Warning `History coverage is partial` | Some commits in scope are not indexed; results may be missing. | Finish indexing before concluding absence. |
| Warning `Embedding coverage is partial` | Semantic ranking saw only some documents. | `nalcos sync --timeout 10m`. |
| Warning `Source omissions across all paths …` | Some patch text was not indexed. | `nalcos --json omissions` and follow each reason's `action`. |
| Warning `Repository is shallow` | History beyond the shallow boundary is unknown. | `git fetch --unshallow` if the user agrees. |
| `output_truncated: true` | Excerpts or messages were clipped. | `nalcos --json show --evidence EVIDENCE_ID`, or raise `--max-bytes`. |
| Results from the wrong branch | The default scope is `HEAD` of the current worktree. | Pass `--ref`, `--range`, or `--all-refs`. |
