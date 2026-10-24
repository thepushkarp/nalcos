# Installing nalcos

Ask the user before installing anything or downloading a model.

## Check what is installed

```sh
nalcos --version
git --version
```

- `nalcos 2.x` is the supported Rust CLI. `nalcos 0.x` is the retired Python CLI, which has no
  subcommands; install the Rust CLI and check `command -v nalcos` to see which executable the
  shell resolves.
- Git must be 2.45 or later. Older Git fails with `unsupported_git`.

## Install the CLI

Apple Silicon macOS, through Homebrew (installs a prebuilt binary and Homebrew Git):

```sh
brew install thepushkarp/tap/nalcos
```

Linux, or macOS without Homebrew, through Cargo:

```sh
cargo install nalcos --version 2.0.0-alpha.3 --locked
```

Cargo builds need Rust 1.88 or later, a C/C++ compiler, and CMake, because the bundled
llama.cpp runtime is compiled from source. The first build takes several minutes. Linux GPU
builds are optional: add `--features cuda` or `--features vulkan` with the matching SDK
installed. Neither is qualified in the alpha.

## Prepare a repository

Every repository has its own index. Run commands inside the checkout or pass `--repo PATH`.

Semantic search (recommended when the user agrees to the download):

```sh
nalcos init --timeout 30m
```

`init` downloads CodeRankEmbed FP16 (about 274 MB) into the shared Hugging Face cache, measures
the available devices, and indexes and embeds the ancestry of `HEAD`. Later repositories reuse
the cached model. If `init` times out, completed batches are kept; continue with:

```sh
nalcos sync --timeout 30m
```

Lexical search needs no model and no download:

```sh
nalcos --json search "cache invalidation" --mode lexical --freshness wait --timeout 5m
```

Preview what `init` would do without writing or downloading anything:

```sh
nalcos --json init --dry-run
```

## Restricted environments

- `--offline` forbids downloads and provider calls. An already cached model still works.
- The index lives outside the working tree, in the platform data directory
  (`~/Library/Application Support/nalcos` on macOS, `$XDG_DATA_HOME/nalcos` or
  `~/.local/share/nalcos` on Linux). Set `NALCOS_DATA_DIR` to a writable directory when that
  location is read-only.
- Model files use the Hugging Face cache: `HF_HUB_CACHE`, otherwise `HF_HOME/hub`, normally
  `~/.cache/huggingface/hub`. `HF_ENDPOINT` selects a Hugging Face mirror.
- `--freshness cached` searches without writing the index at all.
