---
name: nalcos
description: Search local Git history in natural language with the nalcos CLI and get ranked commits with source-verified patch evidence. Use when asked which commit changed, introduced, removed, or broke something; when or why code or behavior changed; to list regression candidates between two revisions; or when git log -S, git log -G, grep, and git blame miss because the history uses different words than the question. Works on local checkouts only and never fetches remote history.
license: MIT
compatibility: Requires the nalcos CLI on macOS (Apple Silicon) or Linux and Git 2.45 or later. Semantic search needs a one-time model download of about 274 MB; lexical mode needs no model.
---

# nalcos

nalcos indexes commit messages and patch hunks from local Git history and ranks commits for a
natural-language query. Every evidence item is re-verified against Git objects before it is
returned. Treat results as candidates to confirm with Git, not as answers.

Use plain Git instead when you already know the exact string (`git log -S`, `git log -G`), the
file and lines (`git blame`, `git log -L`), or need history that is not in the local clone.

## Workflow

### 1. Check the installation

```sh
nalcos --version
```

Expect `nalcos 2.x`. If the command is missing or prints `nalcos 0.x` (the retired Python CLI),
follow [install.md](references/install.md). Ask the user before installing software or
downloading a model.

### 2. Check readiness

Pass the same scope options (`--range`, `--ref`, `--all-refs`, `--first-parent`) that you plan
to search with; readiness and coverage are computed for that scope.

```sh
nalcos --json status
```

| `readiness` | Meaning | Do this |
| --- | --- | --- |
| `uninitialized` | No index for this repository. | Search with `--mode lexical --freshness wait --timeout 5m` (builds a lexical index), or ask before running `nalcos init` (downloads about 274 MB, then indexes). |
| `lexical_only` | Index exists without an embedding model; history may still be partial. | Search with `--mode lexical`, adding `--freshness wait --timeout 5m` while `history_coverage.complete` is `false`. `nalcos init` enables semantic search. |
| `partial` | Some commits or embeddings in scope are not indexed. | Run `nalcos sync --timeout 10m`, or search with `--freshness wait --timeout 5m`. |
| `source_refresh_required` | Index was built by an older extraction format. | Run `nalcos sync`; searches still work meanwhile. |
| `ready` | Every commit and document in scope is indexed and embedded. | Search. Still check `source_coverage`. |

### 3. Search

```sh
nalcos --json search "stop retrying requests after cancellation" --limit 5 --timeout 60s
```

- Describe the change itself: behavior, identifiers, error text, file or module names.
- Choose freshness deliberately:
  - default `auto` indexes new commits for up to two seconds, which writes the index and
    takes a writer lock;
  - `--freshness cached` never writes; use it in read-only or sandboxed environments and
    while another nalcos process is indexing;
  - `--freshness wait` indexes the whole scope first; always pair it with `--timeout`.
- Narrow the scope instead of raising `--limit` (see [Scope and filters](#scope-and-filters)).

Without an embedding model, or with `--mode lexical`, the query is split on every character
other than letters, digits, and `_`, and the words are ORed and ranked by BM25. Each word
matches whole tokens case-insensitively, without stemming: `retry` matches `retry_limit` and
`Retry`, but not `retries` or `retryLimit`. There are no phrases, AND, NOT, regex, or
wildcards. Put the distinctive identifiers in the query, and run a second query with
alternative wording rather than adding filler words.

### 4. Check the result before concluding

Read these fields of the search JSON every time:

- `mode_used`: hybrid search falls back to `lexical` with a warning when no model is usable.
  Compare it with `mode_requested`.
- `history_coverage.complete`: `false` means some commits in scope were not searched.
- `embedding_coverage.complete`: matters only when `mode_used` is not `lexical`.
- `source_coverage.recorded_omissions`: nonzero means some patch text was not indexed
  (binary files, size limits, exclusions, missing objects). Inspect with `nalcos omissions`.
- `warnings`: human-readable strings; read every one.
- `output_truncated`: excerpts were clipped to `--max-bytes`; expand evidence before quoting.

`score` orders results within one response only. It is reciprocal-rank fusion in hybrid mode,
BM25 in lexical mode, and cosine similarity in semantic mode, so scores are not comparable
across modes or queries and are not probabilities.

Each result is one commit with at most two evidence items. Evidence `kind` is `message` or
`diff`; `role` is `match` or `supporting_code`. Field details are in
[json.md](references/json.md).

### 5. Expand and confirm

```sh
nalcos --json show --evidence EVIDENCE_ID
nalcos --json show COMMIT_SHA --path 'src/network/**'
```

Pass evidence IDs exactly as returned (`e1:` followed by 64 hex digits); prefixes do not match.
When an evidence item has `metadata_only: true` (a rename, mode change, or binary file without a
textual hunk), run `nalcos` with its `patch_lookup.args` to read the file patch.

Then confirm with Git itself before you report:

```sh
git show --stat COMMIT_SHA
git log --oneline -n 5 COMMIT_SHA
```

### 6. Report

- Cite the full or abbreviated SHA, the subject line, and the evidence that supports the claim.
- Ranking is not causality. Say "candidate" until the patch or a test confirms the behavior.
- An empty result is not proof of absence. Claim absence only when history coverage is
  complete, `nalcos omissions` reports nothing relevant, and alternative wordings also return
  nothing.
- State the scope you searched and any coverage gaps or warnings.

## Scope and filters

The default scope is the ancestry of `HEAD`. Choose at most one of these:

| Option | Scope |
| --- | --- |
| `--ref REF` | Ancestry of `REF`; repeat for several refs. |
| `--range A..B` | Commits reachable from `B` but not `A`. Only the two-dot form; `A...B` is rejected. |
| `--all-refs` | Every local ref that resolves to a commit, worktree HEADs, and `refs/stash`. |

Combine with any of these:

- `--first-parent`: follow only first parents through merges.
- `--path PATHSPEC`: repeatable Git pathspec; quote globs such as `'src/**'`.
- `--author TEXT`: one case-insensitive substring of `Name <email>`; not a regex or list.
- `--since DATE`, `--until DATE`: committer dates as `YYYY-MM-DD` (UTC; `--until` includes the
  whole day) or RFC 3339 timestamps.
- `--repo PATH`: search another checkout without changing directory.

```sh
nalcos --json search "drop support for legacy tokens" --range v1.8.0..v2.0.0 --path 'src/auth/**' --limit 5 --timeout 60s
nalcos --json search "connection pool sizing" --author alice@example.com --since 2024-01-01 --until 2024-06-30 --freshness cached
```

Recipes for regressions and file archaeology are in [workflows.md](references/workflows.md).

## Output and errors

- JSON goes to stdout as one object with `schema_version: 1`; progress and diagnostics go to
  stderr. Parse stdout only.
- Exit codes: `0` success, including empty results; `2` invalid arguments, configuration, model
  name, or repository path; `1` other failures; `124` timeout; `130` interrupted.
- Failures print `{"schema_version":1,"command":"error","error":{"code":…,"message":…,"action":…}}`.
  Follow `action` when present. Codes and fixes are in
  [troubleshooting.md](references/troubleshooting.md).
- After a `124` timeout, completed index batches are kept; rerun the same command or
  `nalcos sync` to continue.

## Do not

- Do not run `nalcos init`, `nalcos sync --model`, or `--reembed` without the user's consent;
  they download models and can take a long time on large histories.
- Do not use `--freshness wait` or `nalcos sync --watch` without `--timeout`; `--watch` never
  exits on its own.
- Do not treat a high `score` or the first result as the answer without reading its evidence.
- Do not expect remote branches or pull requests that are not fetched locally; nalcos never
  fetches.
