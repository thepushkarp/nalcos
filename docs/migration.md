# Migrating from the Python CLI

NaLCoS 2 is an alpha Rust application. Build this checkout with `cargo install --path . --locked`; the older `pip install nalcos` package uses the Python interface. If both are installed, check `command -v nalcos` and `nalcos --version` to confirm which executable your shell resolves.

## Command mapping

| Python interface | Rust interface |
| --- | --- |
| `nalcos "query" /path/to/repo` | `nalcos --repo /path/to/repo search "query"` |
| `--branch main` | `search "query" --ref main` |
| `--n-matches 5` | `search "query" --limit 5` |
| `--look-past 100` | An explicit revision range, such as `--range BASE..HEAD`, when that is the intended scope. |
| `--github` with `owner/repo` | Clone or fetch the repository using Git, then point `--repo` at the local checkout. |
| `--verbose` to print a full commit message | Expand a result with `show COMMIT` or `show --evidence ID`. Global `--verbose` controls diagnostics. |
| `--show-score` | Machine-readable search results include ranking information in `--json` output. Scores are not confidence probabilities. |

The default history scope is all ancestry of the current `HEAD`, subject to reported indexing coverage. It is not a fixed 100-commit window. A revision range describes Git reachability; it is not an exact replacement for a commit-count limit on histories containing merges.

## Start with the local index

```sh
nalcos --repo /path/to/repo search "handle missing values" --mode lexical
nalcos --repo /path/to/repo status
```

Search creates the derived local index as needed. Automatic updates are bounded and may leave history partially indexed; use `--freshness wait` for an explicit indexing wait. Use `--freshness cached` to leave the index unchanged while querying.

Select an embedding model explicitly when enabling semantic search. The alpha has no benchmark-qualified default. `init --model` and `sync --model` select a model; a configuration profile can supply the same selection. See [configuration](configuration.md) for supported model formats and complete examples.

## Models and stored data

The Rust index is not compatible with any Python-era derived index or cache. NaLCoS reads Git objects to build its own index outside the working tree; it does not need a conversion of old query results.

An older Rust index can require a source-format refresh. `status`, search, and `sync --dry-run` report `source_refresh_required`; run `sync` to refresh retained indexed patches. Unchanged source records keep their evidence IDs and vectors. Searches continue to verify evidence against Git while a refresh is pending. Interrupted refreshes can be retried with `sync`.

Model files are shared through the Hugging Face cache. NaLCoS does not import the old `nalcos/models/Cache` directory as an index or write new weights into the repository. Existing Python installations and caches can be kept independently; verify the Rust setup before removing anything manually.

Changes to embedding weights, revision, representation, or document preprocessing require complete reembedding. Apply model changes through `init` or `sync`; search uses the active generation rather than silently applying a newly edited profile.

Model choices persist: plain `sync` resumes a pending replacement or uses the active selection. An unchanged `default_model` does not undo an explicit CLI switch. Setting a new configuration default or passing `--model` requests a switch; an explicit incompatible choice supersedes pending work.

Ordinary synchronization preserves the resolved HF commit even when a moving ref has advanced. Pass `--revision` explicitly to request a model upgrade. A query-prefix-only change preserves document vectors, although applying it can run synthetic encoder and provider-identity probes. `--offline` disallows API calls even to a localhost Ollama server.

The derived cache follows all current refs under `refs/` that resolve to commits, worktree HEADs, and explicitly requested history. This includes custom refs and `refs/stash`; reflog-only commits are excluded unless requested. Sync prunes cached commits outside that retention set. Keep a Git ref to history that should remain retained across later syncs.

## Agent integrations

Update shell scripts to call a verb explicitly and consume `--json`. Check `schema_version`, coverage, warnings, and `output_truncated`; do not parse terminal tables or treat a partial empty result as a definitive absence.

`--timeout` accepts a duration such as `30s` or `5m`. An empty search result is successful. Invalid arguments/configuration use exit code 2, operation failures use 1, timeouts use 124, and interruption uses 130.

The Rust CLI operates on refs already available locally. It does not perform GitHub API searches, fetch remote history, generate an answer, or run a regression bisect. Its agent interface is the CLI and JSON output.
