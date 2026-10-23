# Usage reference

Run commands inside a Git checkout or pass `--repo PATH`. All commands accept `--config`, `--json`, `--offline`, `--timeout DURATION`, `--device`, and `--verbose`. Use `nalcos COMMAND --help` for the full option list. Coding agents can follow the [nalcos agent skill](../skills/nalcos/SKILL.md).

## Scope and filters

The default scope is the ancestry of the current `HEAD`. Select a different scope with one of these mutually exclusive options:

| Option | Scope |
| --- | --- |
| `--ref REF` | Ancestry of a named ref; repeat to include several refs. |
| `--range A..B` | Commits reachable from `B` but not `A`. Only the two-dot form is accepted; `A...B` is rejected. |
| `--all-refs` | Every local ref resolving to a commit, plus worktree HEADs. Includes custom namespaces and `refs/stash`; excludes reflog-only commits. |

`--first-parent` narrows traversal through merges. Cached searches still validate the requested scope against local Git objects. NaLCoS uses Git's `--no-lazy-fetch` to prevent implicit downloads from partial clones; missing objects are reported and can leave evidence incomplete.

Search filters:

- `--path PATHSPEC`: repeatable Git pathspec. Diff evidence must match an old or new path; commit messages qualify when first-parent changes touch a matching path.
- `--author TEXT`: one case-insensitive literal substring of the author's `Name <email>`, such as `alice` or `@example.com`. Patterns and lists are not interpreted.
- `--since DATE`, `--until DATE`: inclusive committer timestamps, as `YYYY-MM-DD` (UTC) or RFC 3339 timestamps such as `2024-03-01T12:00:00+01:00`. A date-only `--until` includes the whole UTC day. `--since` must not be later than `--until`.

## Retrieval and freshness

Retrieval mode and index freshness are independent:

| Mode | Behavior |
| --- | --- |
| `--mode hybrid` | Default. Combine lexical and semantic retrieval; report lexical fallback if semantic retrieval is unavailable. |
| `--mode lexical` | Search without a model. |
| `--mode semantic` | Require usable embeddings; do not fall back to lexical results. |

Lexical retrieval splits the query on every character other than letters, digits, and `_`, ORs the resulting words, and ranks matching documents by BM25. Each word matches whole tokens case-insensitively, without stemming: `retry` matches `retry_limit` and `Retry`, but not `retries` or `retryLimit`. Phrase, boolean, regex, and wildcard syntax is not interpreted. Hybrid ranking fuses the lexical and semantic rankings by reciprocal rank, so result scores are not comparable across modes or queries.

| Freshness | Behavior |
| --- | --- |
| `--freshness auto` | Default. Update incrementally with a two-second indexing budget and report remaining work. |
| `--freshness cached` | Query without updating the index. |
| `--freshness wait` | Wait for indexing of the requested scope before searching. |

Search never downloads models or runtime libraries. Run `init` to install a model and enable semantic indexing. Use `sync` to finish or resume indexing. Model selection, persistence, and reembedding are covered in [configuration](configuration.md).

`init --dry-run` and `sync --dry-run` plan work without creating an index, downloading assets, or probing an encoder. `sync --watch` checks for changes every two seconds in the foreground and stops with Ctrl-C.

## Evidence and coverage

Each result identifies a commit and carries at most two evidence items, the commit's best-ranked documents. Indexed diffs compare each commit with its first parent. Expand a result's exact evidence ID with:

```sh
nalcos show --evidence EVIDENCE_ID
```

Inspect a file patch or a different merge parent with:

```sh
nalcos show COMMIT_SHA --path 'src/network/**' --parent 2 --context 5
```

Metadata-only matches are labeled and include a file-patch command in human output or `patch_lookup.args` in JSON. A separately verified hunk from the same file may accompany them as `supporting_code`.

`status` and `search` report history, embedding, and source coverage. Complete commit coverage does not imply complete source text. Source-omission counts cover all paths in the selected commits, including path-filtered searches. Inspect skipped sources and recovery guidance with:

```sh
nalcos omissions --limit 20
```

Use `--offset` to page through omissions; JSON supplies `next_offset`. `status` and `omissions` are read-only and never download assets. `status --check` explicitly probes the encoder with synthetic input.

During embedding work, `init` and `sync` report the actual device or remote provider. JSON exposes it as `indexing_runtime`.

## JSON and output limits

`--json` emits a single object with `schema_version: 1`. `sync --watch --json` emits one object per line for the initial synchronization and each changed update. Progress and diagnostics go to stderr.

Search output includes the resolved scope, requested/effective mode, coverage, active generation, runtime, results, warnings, and `output_truncated`. Check these fields before treating an empty result as evidence that a change does not exist. Coding agents use the CLI directly; there is no MCP server. The [agent skill](../skills/nalcos/SKILL.md) describes the workflow, and its [JSON reference](../skills/nalcos/references/json.md) lists the fields.

| Option | Default and limit |
| --- | --- |
| Search `--limit` | 10 results; accepts 1–100. |
| Search `--max-bytes` | 16 KiB shared across result excerpts. JSON envelope and commit metadata add overhead. |
| Show `--max-bytes` | 16 KiB patch budget. `--full` lifts the response budget; extraction safety limits still apply. |
| Show `--parent`, `--context` | Parent 1, three context lines. |

Use `-vv` for search-phase timings on stderr. `--timeout` accepts durations such as `30s` or `5m`. GGUF checks cancellation and deadlines between native batches, so a running batch can delay exit; see [runtime limitations](native-runtime.md).

An empty search result exits successfully. Invalid arguments, configuration, model names (`unknown_model`), and repository paths (`not_a_repository`) exit with code 2, other operation failures with 1, timeouts with 124, and interruption with 130. With `--json`, a failure prints `{"schema_version":1,"command":"error","error":{...}}` on stdout with a stable `code`, a `message`, and an `action` when a fix is known.

## Environment variables

| Variable | Effect |
| --- | --- |
| `NALCOS_CONFIG` | Configuration file used when `--config` is absent. |
| `NALCOS_DATA_DIR` | Index storage root; takes precedence over the `data_dir` setting. |
| `HF_HUB_CACHE` | Hugging Face model cache directory. |
| `HF_HOME` | Hugging Face home; without `HF_HUB_CACHE`, models are cached in `$HF_HOME/hub`, otherwise `$XDG_CACHE_HOME/huggingface/hub` or `~/.cache/huggingface/hub`. |
| `HF_ENDPOINT` | Hugging Face Hub base URL for model downloads, such as a mirror; default `https://huggingface.co`. Must be HTTP(S) without embedded credentials. |
| `HF_TOKEN` | Hugging Face access token; otherwise the Hugging Face token cache is used. |
| `NALCOS_ORT_LIBRARY` | ONNX Runtime shared library; `runtime.ort_library` takes precedence and `ORT_DYLIB_PATH` is the fallback. |
| `NALCOS_RUNTIME_CACHE` | Directory for the ONNX Runtime library that explicit setup installs. |

See [configuration](configuration.md) for the precedence of these values relative to configuration settings.
