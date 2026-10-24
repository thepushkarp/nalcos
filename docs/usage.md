# Usage reference

Run commands inside a Git checkout or pass `--repo PATH`. All commands accept `--config`, `--json`, `--offline`, `--timeout DURATION`, `--device`, and `--verbose`. Use `nalcos COMMAND --help` for the full option list.

A search query may start with `-`, as in `nalcos search "--no-verify flag"`. A query that is exactly an option name, such as `-v`, must follow `--`: `nalcos search -- -v`.

## Scope and filters

The default scope is the ancestry of the current `HEAD`. Select a different scope with one of these mutually exclusive options:

| Option | Scope |
| --- | --- |
| `--ref REF` | Ancestry of a named ref; repeat to include several refs. |
| `--range A..B` | Commits reachable from `B` but not `A`. |
| `--all-refs` | Every local ref resolving to a commit, plus worktree HEADs. Includes custom namespaces and `refs/stash`; excludes reflog-only commits. |

`--first-parent` narrows traversal through merges. Cached searches still validate the requested scope against local Git objects. NaLCoS uses Git's `--no-lazy-fetch` to prevent implicit downloads from partial clones; missing objects are reported and can leave evidence incomplete.

Search filters:

- `--path PATHSPEC`: repeatable Git pathspec. Diff evidence must match an old or new path; commit messages qualify when first-parent changes touch a matching path.
- `--author AUTHOR`: filter commits by author.
- `--since DATE`, `--until DATE`: inclusive committer timestamps. A date-only `--until` includes the whole UTC day; RFC 3339 timestamps are accepted.

## Retrieval and freshness

Retrieval mode and index freshness are independent:

| Mode | Behavior |
| --- | --- |
| `--mode hybrid` | Default. Combine lexical and semantic retrieval; report lexical fallback if semantic retrieval is unavailable. |
| `--mode lexical` | Search without a model. |
| `--mode semantic` | Require usable embeddings; do not fall back to lexical results. |

| Freshness | Behavior |
| --- | --- |
| `--freshness auto` | Default. Update incrementally with a two-second indexing budget and report remaining work. |
| `--freshness cached` | Query without updating the index. |
| `--freshness wait` | Wait for indexing of the requested scope before searching. |

Search never downloads models or runtime libraries. Run `init` to install a model and enable semantic indexing. Use `sync` to finish or resume indexing. Model selection, persistence, and reembedding are covered in [configuration](configuration.md).

`init --dry-run` and `sync --dry-run` plan work without creating an index, downloading assets, or probing an encoder. `sync --watch` checks for changes every two seconds in the foreground and stops with Ctrl-C.

## Evidence and coverage

Each result identifies a commit and parent comparison. Expand a result's exact evidence ID with:

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

Search output includes the resolved scope, requested/effective mode, coverage, active generation, runtime, results, warnings, and `output_truncated`. Check these fields before treating an empty result as evidence that a change does not exist. The CLI is the coding-agent interface; there is no MCP server.

| Option | Default and limit |
| --- | --- |
| Search `--limit` | 10 results; accepts 1–100. |
| Search `--max-bytes` | 16 KiB shared across result excerpts. JSON envelope and commit metadata add overhead. |
| Show `--max-bytes` | 16 KiB patch budget. `--full` lifts the response budget; extraction safety limits still apply. |
| Show `--parent`, `--context` | Parent 1, three context lines. |

Use `-vv` for search-phase timings on stderr. `--timeout` accepts durations such as `30s` or `5m`. GGUF checks cancellation and deadlines between native batches, so a running batch can delay exit; see [runtime limitations](native-runtime.md).

An empty search result exits successfully. Invalid arguments/configuration exit with code 2, operation failures with 1, timeouts with 124, and interruption with 130.
