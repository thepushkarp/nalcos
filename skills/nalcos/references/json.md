# JSON contract

Pass the global `--json` flag before or after the subcommand. Each command prints one JSON
object with `schema_version: 1` on stdout; `sync --watch --json` prints one object per line.
Progress, warnings for humans, and diagnostics go to stderr. Fields not listed here are
diagnostic and may change.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success, including a search with no results. |
| `2` | Invalid arguments or configuration, or an unknown `--model`. |
| `1` | Any other failure (Git, index, model, runtime, provider). |
| `124` | `--timeout` elapsed. Completed index batches are kept. |
| `130` | Interrupted with Ctrl-C. Completed index batches are kept. |

## Errors

```json
{"schema_version":1,"command":"error","error":{"code":"semantic_unavailable","message":"No active embedding generation is available","action":"Run nalcos init to configure an embedding model, select --model HF_ID, or use --mode lexical"}}
```

`code` is stable for programs; `message` is for people; `action`, when present, is the fix.
Argument errors from the parser also use this envelope with `code: "invalid_input"` when
`--json` is present. See [troubleshooting.md](troubleshooting.md) for codes.

## Shared fields

- `repository`: `id`, `work_dir`, `common_dir`, and `remote_freshness: "unchecked"` (nalcos
  never contacts remotes).
- `scope`: resolved `roots` and `excluded_roots` (`name`, `oid`), `first_parent`, `shallow`,
  `commits` (number of commits in scope), and `signature`.
- `history_coverage`: `total` and `indexed` commits, `complete`, and `omissions` (recorded
  source omissions in scope).
- `embedding_coverage`: `total` and `indexed` documents, and `complete`. It is never complete
  without an active embedding generation.
- `source_coverage`: `recorded_omissions`, `categories` (`expected`, `limited`, `unavailable`,
  `other`), and `reasons` (`reason`, `count`, `category`, `action`). Counts cover every path in
  the selected commits, even for path-filtered searches.
- `warnings`: array of strings. Treat any entry as a reason to qualify the conclusion.

## `status`

- `readiness`: `uninitialized`, `source_refresh_required`, `lexical_only`, `partial`, or
  `ready`, checked in that order. `ready` means every commit and document in scope is
  indexed and embedded; it does not mean that no source text was omitted.
- `source_refresh_required`: `true` when `nalcos sync` must refresh stored patches.
- `active_generation`: the embedding model in use (`model`, `revision`, `backend`,
  `dimensions`), or `null`.
- `staging_generation` and `staging_progress`: an unfinished model switch (`completed`,
  `total`, `remaining`, `resume_command`). `resume_command` is `null` while another process
  is indexing.
- `writer_active`: `true` while another nalcos process holds the index writer lock.
- `runtime`: the device used by the last indexing run, or `null`.
- `check`: result of `status --check`, which probes the encoder with synthetic text.

## `search`

- `mode_requested` and `mode_used`: `hybrid`, `lexical`, or `semantic`. Hybrid falls back to
  lexical, with a warning, when no usable embedding model exists.
- `freshness`: the policy used (`auto`, `cached`, or `wait`).
- `filters`: `paths`, `author`, `since`, `until` as given.
- `results`: at most `--limit` commits, best first. Each has:
  - `commit`: `oid`, `parents`, `author_name`, `author_email`, `authored_at`, `committed_at`
    (Unix seconds), and `message` (clipped to 4096 bytes).
  - `score`: ordering within this response only. Reciprocal-rank fusion in hybrid mode, BM25
    in lexical mode, cosine similarity in semantic mode. Not comparable across modes or
    queries, and not a probability.
  - `evidence`: at most two items, described below.
- `output_truncated`: `true` when any excerpt or commit message was clipped. `--max-bytes` is
  shared by all excerpts in the response. A total-output budget also sets this flag when it
  removes results or shortens excerpts.
- `source_refresh_required`, `active_generation`, and `runtime` as in `status`.

When no index exists yet (for example `--freshness cached` before any indexing), `results` is
empty, `mode_used` is `lexical`, and `warnings` explains why; `filters` and `freshness` are
omitted.

### Compact search (`--compact`)

With `--json search --compact`, `schema_version` stays `1` and `compact` is `true`.
The response retains `mode_requested`, `mode_used`, `freshness`, history and embedding coverage,
`warnings`, `output_truncated`, and `results`. `source_coverage` retains `recorded_omissions`
and `categories`; detailed reasons are available from `omissions`. Compact `freshness` is
included even when no index exists.

Each result carries:

- `rank`: 1-based retrieval order, best first.
- `commit`: full OID; `short`: its first 12 characters; `subject`: first message line.
- `committed_at`: RFC 3339 UTC timestamp; `author`: author name.
- `evidence`: `id`, `kind`, `role`, `excerpt`, `verified`, `excerpt_truncated`, and
  `metadata_only`, with the meanings below; `path` is a display path, `"old -> new"` for
  renames, or `null` for messages. Diff evidence includes `old_start`, `old_lines`,
  `new_start`, and `new_lines`.
- Each evidence item also has `show`: runnable argv
  `["nalcos", "--repo", PATH, "show", "--evidence", ID]`.

Hashes, blob/parent/tree OIDs, emails, full messages, scores, and duplicate path encodings
remain available from verbose search or `show --evidence`. Human output is unchanged.

### Total JSON budget (`--max-output-bytes N`)

`N` is at least 1024 and bounds JSON search stdout, including the trailing newline, with or
without `--compact`. Lowest-ranked results are dropped first; if the top result alone
overflows, its excerpts are shortened on UTF-8 boundaries and marked `excerpt_truncated`.
`dropped_results` counts removed commits, `output_bytes` reports stdout bytes, and any
trimming sets `output_truncated: true`. Output still exceeds `N` only when the top result's
metadata alone does.

### Evidence items

- `id`: stable evidence ID, `e1:` followed by 64 hex digits. `show --evidence` needs the exact
  string; prefixes do not match.
- `kind`: `message` (commit message) or `diff` (one file hunk or file header).
- `role`: `match` when retrieval found it; `supporting_code` when it is a verified hunk from
  the same file added beside a metadata-only match.
- `commit_oid`, `parent_oid`, `parent_index`: the comparison the hunk belongs to. Indexing
  compares each commit with its first parent; use `show COMMIT --parent 2` for the other side
  of a merge.
- `old_path`, `new_path`: `{display, bytes_hex}` or `null`. Use `display` for reading; it
  escapes control characters and invalid UTF-8.
- `old_start`, `old_lines`, `new_start`, `new_lines`: hunk coordinates.
- `excerpt`, `excerpt_byte_start`, `excerpt_byte_end`, `excerpt_truncated`: the returned slice
  of the document text.
- `truncated`: the indexed document itself was clipped by an indexing limit.
- `verified`: always `true`; evidence that no longer matches Git is dropped and counted in a
  warning.
- `metadata_only`: `true` for a file change without a textual hunk (rename, mode change,
  binary file). Such items carry `patch_lookup`.
- `patch_lookup.args`: arguments to pass to `nalcos` to read the patch, for example
  `["show", COMMIT, "--parent", "1", "--path", ":(literal)src/a.rs", "--full"]`.
  `patch_lookup.file_scoped` is `false` when the path could not be expressed as a string and
  the command shows the whole commit/parent patch.

## `show`

- With `--evidence ID`: `commit` and `evidence` (one evidence item with the full document text
  in `excerpt`, subject to `--max-bytes` unless `--full`), and `output_truncated`.
- With `COMMIT`: `commit`, `parent_oid`, `patch` (unified diff text), `omissions`, and
  `output_truncated`. `--parent N` picks the merge parent; `--path` restricts files.

## `omissions`

- `records`: one item per omitted source with `commit_oid`, `path`, `reason`, and `detail`.
- `limit`, `offset`, `next_offset`: repeat with `--offset NEXT_OFFSET` and the same scope
  options until `next_offset` is `null`.
- `history_coverage` and `source_coverage` as above.

## `init` and `sync`

- `commits_added`, `commits_pruned`, `documents_embedded`.
- `history_coverage`, `embedding_coverage`, `active_generation`, `staging_generation`.
- `indexing_runtime`: the device or provider that embedded documents in this run.
- With `--dry-run`: `commits_pending`, `model`, `model_profile`, `full_reembedding`, and
  `writes_performed: false`.
