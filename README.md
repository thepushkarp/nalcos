# NaLCoS — Natural Language Commit Search

Find past changes and investigate regression candidates in local Git history. NaLCoS searches commit messages and patch evidence, then returns the commits and changed code for you or your coding agent to inspect.

**Version 2 is an alpha Rust CLI.** Fresh `nalcos init` uses MiniLM INT8 on CPU. This is the alpha product default chosen for its small download and fast local indexing; broader retrieval and release qualification remain pending. See [benchmark methodology and results](docs/benchmarks.md).

## Install

On Apple Silicon macOS, install the alpha from the [Homebrew tap](https://github.com/thepushkarp/homebrew-tap):

```sh
brew install thepushkarp/tap/nalcos
```

To build from source, use Rust 1.88 or newer, Git 2.45 or newer, a C/C++ compiler, and CMake:

```sh
cargo install --path . --locked
nalcos --help
```

NaLCoS uses Git's `--no-lazy-fetch` on every invocation to prevent implicit object downloads from partial clones. Missing local objects are reported and can leave evidence incomplete; NaLCoS does not fetch them.

The native embedding backends may need model assets and a compatible runtime library. Lexical search works without downloading a model. See [configuration](docs/configuration.md) for embedding profiles and runtime setup.

Users of the Python release should read the [migration guide](docs/migration.md). The Rust CLI has a new command interface and index format.

## Find a past change

Run inside a Git checkout, or select one with `--repo`:

```sh
nalcos search "stop retrying a request after cancellation"
nalcos --repo ../another-project search "add an authentication provider"
```

The first search creates a local index and performs a bounded incremental update. `hybrid` search uses lexical retrieval when a semantic index is unavailable and reports that fallback. Automatic search updates have a two-second indexing budget and do not download models or runtime libraries. Results can therefore cover only part of the requested history until indexing finishes.

Use `sync` for an explicit update, or `--freshness wait` when the query must wait for indexing. Run `nalcos init` to install the default embedding model and enable semantic indexing. Choose another model with `init --model`, `sync --model`, or a configuration profile. Model selection and supported formats are described in [configuration](docs/configuration.md).

To initialize MiniLM INT8 on CPU and search semantically:

```sh
nalcos init
nalcos search "stop retrying a request after cancellation" --mode semantic
```

For a model-free search with complete indexing of the requested scope:

```sh
nalcos search "stop retrying a request after cancellation" \
  --mode lexical --freshness wait --path 'src/network/**'

nalcos show COMMIT_SHA --path 'src/network/**'
```

A search result points to evidence in a particular commit and parent comparison. Expand a returned evidence identifier with `nalcos show --evidence EVIDENCE_ID`.

## Investigate a regression

Restrict candidate retrieval to the commits between a known-good revision and the failing revision:

```sh
nalcos search "requests are retried after the caller cancels" \
  --range v1.8.0..HEAD \
  --path 'src/network/**' \
  --freshness wait --limit 8
```

Ranking identifies changes worth inspecting. Verify the behavior with the patch, tests, and your existing debugging tools; similarity is not a probability that a commit caused the regression. NaLCoS does not generate answers or run bisection.

## Five commands

| Command | Purpose |
| --- | --- |
| `init` | Select an embedding model and initialize its index. Fresh setup defaults to MiniLM INT8 on CPU; existing model selections persist. |
| `sync` | Update the requested history scope and embeddings. Add `--watch` to keep updating in a foreground process. |
| `search QUERY` | Retrieve ranked commit and patch evidence. |
| `show COMMIT` | Read a commit's patch, or expand an exact result with `--evidence ID`. |
| `status` | Inspect index coverage and active model/runtime state. Add `--check` for an explicit encoder probe. |

All commands accept `--repo`, `--config`, `--json`, `--offline`, `--timeout DURATION`, `--device`, and `--verbose`. Use `nalcos COMMAND --help` for the complete option list.

Use `-vv` to print search-phase timings to stderr when diagnosing latency.

`init` and `sync` accept `--model`, `--revision`, `--variant`, `--reembed`, and `--dry-run`. Dry runs do not create an index, download assets, or probe an embedding provider. A model change rebuilds the embedding generation; `--reembed` requests that work even when the selected model is unchanged.

`sync --watch` stays in the foreground, checks for changes every two seconds, and stops with Ctrl-C. With `--json`, it emits one JSON object per line for the initial synchronization and each changed update; other commands return a single JSON object.

### Scope and freshness

The default scope is the ancestry of the current `HEAD`. Select a different scope with one of:

- Repeated `--ref REF` options to include the ancestry of named refs.
- `--range A..B` to include commits reachable from `B` but not `A`.
- `--all-refs` to include every current ref under `refs/` that resolves to a commit, plus worktree HEADs. This includes custom namespaces and `refs/stash`; reflog-only commits are excluded.

These selectors are mutually exclusive. `--first-parent` narrows traversal through merges. NaLCoS does not fetch remote refs; a remote-tracking ref must already exist locally. Cached searches still validate their requested scope against the local Git repository.

Search freshness is independent of its retrieval mode:

| Option | Behavior |
| --- | --- |
| `--freshness auto` | Default. Perform a bounded incremental update and report remaining work. |
| `--freshness cached` | Query the existing index without updating it. |
| `--freshness wait` | Wait for indexing of the requested scope before searching. |

`--mode hybrid` combines lexical and semantic retrieval when available. `--mode lexical` needs no model. `--mode semantic` requires usable embeddings; it does not silently return lexical results.

`--path` accepts repeatable Git pathspecs. It limits diff evidence to matching old or new paths and includes messages from commits whose first-parent changes touch a matching path. `--author`, `--since`, and `--until` narrow search results. Date filters use inclusive committer timestamps; date-only `--until` includes the whole UTC day, and RFC 3339 timestamps are accepted.

## Coding-agent use

Use `--json` for machine-readable results and `show` to expand selected evidence:

```sh
nalcos --json search "stop retrying cancelled requests" \
  --range v1.8.0..HEAD --limit 5 --max-bytes 16384

nalcos --json show --evidence EVIDENCE_ID --max-bytes 16384
```

The JSON envelope is versioned with `schema_version: 1`. It identifies the command, repository, resolved scope, requested and effective retrieval modes, history coverage, embedding coverage, active generation, runtime, results, warnings, and output truncation where applicable. Check coverage and warnings before treating an empty result as evidence that a change does not exist.

`search` defaults to 10 results and accepts at most 100. Both `search` and `show` default to a 16 KiB evidence/patch budget; the JSON envelope and commit metadata add overhead. `show` defaults to parent 1 and three context lines; use `--parent`, `--context`, and `--path` to focus it, or `--full` to lift the response byte budget. Extraction safety limits still apply. Progress and diagnostics are separate from JSON stdout.

The CLI is the agent interface in this release. There is no MCP server.

## Local data and model control

The index is derived from local Git objects and stored outside the working tree. Linked worktrees share an index through their common Git directory; each query still uses its requested revision scope. `status` is read-only and does not create an index or download assets.

Sync retains indexed history reachable from current refs, worktree HEADs, and the requested scope. It prunes other derived cache entries and reuses compatible vectors for identical content where possible. Queries still search only their requested scope.

Model choices persist. Plain `sync` resumes a pending model replacement or updates the active model; an unchanged configuration default cannot undo an explicit CLI choice. Searches use the active generation while `init` or `sync` applies model/profile changes. Ordinary syncs keep the resolved HF revision pinned; request a revision explicitly to upgrade it. Model artifacts use the shared Hugging Face cache rather than repository-local copies. API embedding profiles send selected text to the configured endpoint; `--offline` blocks provider calls, including localhost servers.

See the [configuration reference](docs/configuration.md) and [example configuration](examples/config.toml).

## Development and validation

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
```

CPU CI checks build and behavior. The optional Metal smoke test requires Apple Silicon and the pinned EmbeddingGemma GGUF snapshot in the shared cache; it reports a skip when those prerequisites are unavailable and never downloads a model. To run that test after explicitly installing its assets:

```sh
NALCOS_RUN_MODEL_TESTS=1 cargo test --locked --lib \
  embedding::tests::cached_embeddinggemma_metal_smoke -- --ignored --exact --nocapture
```

Native cached-model CPU and Metal smoke checks have passed on Apple Silicon; [native runtime validation](docs/native-runtime.md) records the exact evidence and unresolved reference-parity differences. See [CPU and Metal comparisons](docs/configuration.md#compare-cpu-and-metal) for explicit device probes and end-to-end timing. CUDA remains unqualified in this alpha; optional build features do not qualify CUDA or Vulkan releases. Model quality and product qualification remain pending, including the two-week dogfood evaluation.

## Contributing

Useful contributions include reproducible retrieval misses, Git-history edge cases, and measured indexing/runtime results. Include the command, effective configuration, scope, and relevant diagnostics; remove private repository content and credentials before sharing.

Use [Discussions](https://github.com/thepushkarp/nalcos/discussions) for product ideas, [issues](https://github.com/thepushkarp/nalcos/issues/new/choose) for bugs, or [pull requests](https://github.com/thepushkarp/nalcos/pulls) for changes. See [benchmarks](docs/benchmarks.md) before making accuracy or performance claims.

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://thepushkarp.com/"><img src="https://avatars.githubusercontent.com/u/42088801?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Pushkar Patel</b></sub></a><br /><a href="https://github.com/thepushkarp/nalcos/commits?author=thepushkarp" title="Code">💻</a> <a href="https://github.com/thepushkarp/nalcos/commits?author=thepushkarp" title="Documentation">📖</a> <a href="#maintenance-thepushkarp" title="Maintenance">🚧</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## License

This project is licensed under the terms of the MIT license.
