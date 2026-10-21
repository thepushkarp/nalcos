# NaLCoS — Natural Language Commit Search

[![skills.sh](https://skills.sh/b/thepushkarp/nalcos)](https://skills.sh/thepushkarp/nalcos)

Search local Git history with natural-language queries. Get ranked commits and verified patch evidence to inspect.

**Version 2 is an alpha Rust CLI.** Retrieval quality and broader platform qualification are still being evaluated.

## Install

Apple Silicon macOS:

```sh
brew install thepushkarp/tap/nalcos
```

From [crates.io](https://crates.io/crates/nalcos):

```sh
cargo install nalcos --version 2.0.0-alpha.3 --locked
```

Requires Git 2.45+. Cargo builds also require Rust 1.88+, a C/C++ compiler, and CMake.

## Quick start

Run inside a Git checkout:

```sh
nalcos init
nalcos search "stop retrying requests after cancellation"
nalcos show COMMIT_SHA
```

`init` installs CodeRankEmbed FP16 (274 MB) for a fresh index and indexes local history. It measures compatible devices separately for indexing and queries, selects the fastest, and reports the device in use.

Existing indexes keep their selected model. To switch, run `nalcos sync --model coderank`.

To search without a model:

```sh
nalcos search "stop retrying requests after cancellation" --mode lexical --freshness wait
```

Search normally updates the index for up to two seconds and reports coverage. Use `--freshness wait` to wait for indexing of the requested scope, or `nalcos sync` to finish indexing separately.

### Narrow the search

```sh
nalcos search "requests are retried after cancellation" \
  --range v1.8.0..HEAD --path 'src/network/**' --freshness wait --limit 8
```

The default scope is the ancestry of `HEAD`. Use `--repo PATH` to search another checkout. Regression searches return candidates to investigate; ranking does not establish causality.

### Use with coding agents

Install the agent skill, which teaches coding agents to check readiness and coverage, search, and confirm evidence with Git:

```sh
npx skills add thepushkarp/nalcos
```

In Claude Code, install it as a plugin instead:

```text
/plugin marketplace add thepushkarp/nalcos
/plugin install nalcos@nalcos
```

Agents call the CLI with `--json`:

```sh
nalcos --json search "stop retrying cancelled requests" --limit 5 --max-bytes 16384
nalcos --json show --evidence EVIDENCE_ID --max-bytes 16384
```

Check coverage, warnings, and `output_truncated` before interpreting results. Progress and diagnostics go to stderr; JSON goes to stdout. The [JSON reference](skills/nalcos/references/json.md) lists fields and exit codes.

## Commands

| Command | Purpose |
| --- | --- |
| `init` | Install/select a model and index history. |
| `sync` | Update the index; `--watch` keeps it current in the foreground. |
| `search QUERY` | Find commit and patch evidence. |
| `show COMMIT` | Read a patch; `--evidence ID` expands an exact result. |
| `status` | Inspect coverage, model, and runtime without changing the index. |
| `omissions` | Inspect skipped source content and recovery guidance. |

Use `nalcos COMMAND --help` for options. The index is stored outside the working tree. NaLCoS reads local Git objects and does not fetch missing history.

## Documentation

- [Usage reference](docs/usage.md): scope, filters, freshness, evidence, and JSON output.
- [Configuration](docs/configuration.md): models, devices, storage, and API providers; [example TOML](examples/config.toml).
- [Benchmarks](docs/benchmarks.md) and [native runtime validation](docs/native-runtime.md): measurements and limitations.
- [Migration](docs/migration.md): the retired Python CLI and older Rust indexes.

## Development

```sh
cargo install --path . --locked
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
```

The normal test suite requires no model downloads. See [native runtime validation](docs/native-runtime.md) for optional model checks and the [release guide](docs/releasing.md) for publishing.

[Report bugs](https://github.com/thepushkarp/nalcos/issues/new/choose) with the command, scope, and diagnostics needed to reproduce them. Remove private repository content and credentials. [Pull requests](https://github.com/thepushkarp/nalcos/pulls) are welcome.

## License

[MIT](LICENSE).
