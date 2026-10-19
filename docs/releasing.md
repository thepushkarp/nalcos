# Releasing NaLCoS

NaLCoS is distributed as a Rust CLI. Homebrew serves a prebuilt Apple Silicon macOS binary; crates.io serves source packages for Cargo installation. The Python CLI is retired. Python scripts under `benchmarks/` are development tools with their own dependencies.

## First crates.io publication

An initial publication requires a maintainer API token. Trusted publishing requires the crate to exist on crates.io. Keep prerelease versions explicit in installation commands.

1. Review the source and package contents, commit the intended release changes, and run:

   ```sh
   cargo fmt --all -- --check
   cargo clippy --locked --all-targets -- -D warnings
   cargo test --locked --all-targets
   cargo package --locked --list
   cargo publish --dry-run --locked
   ```

2. Sign in to crates.io with the maintainer's GitHub account and verify its email address. Create an API token authorized to create and publish `nalcos`, then authenticate locally using `cargo login`. Do not paste the token into an issue, workflow file, or chat.
3. Publish the reviewed checkout using `cargo publish --locked`. Verify the version at <https://crates.io/crates/nalcos> and install it in a temporary location using `cargo install nalcos --version 2.0.0-alpha.2 --locked --root TEMP_DIRECTORY`. Check the installed binary's `--version` and `--help`.
4. On the crate's Settings → Trusted Publishing page, configure GitHub owner `thepushkarp`, repository `nalcos`, and workflow filename `release.yml`, with no environment. The workflow uses `rust-lang/crates-io-auth-action` to obtain a temporary publish token. Revoke the bootstrap API token after setup.

The first publication requires an API token; trusted publishing requires an existing crate. See the [crates.io trusted publishing guide](https://crates.io/docs/trusted-publishing).

## Subsequent crates.io releases

Update the version in `Cargo.toml` and `Cargo.lock`, run the checks above, and create a matching `vVERSION` tag from the reviewed commit. Publishing a GitHub release for that tag runs `.github/workflows/release.yml`, including prereleases. The workflow checks out the tag, verifies its version, tests the source, verifies the packaged build, and publishes through trusted publishing.

To retry a failed publication, manually run the workflow with the existing tag. A version already published to crates.io cannot be overwritten; inspect the registry before retrying. Do not move an existing release tag to different code.

## Homebrew releases

The [Homebrew tap](https://github.com/thepushkarp/homebrew-tap) supports NaLCoS through its `nalcos-release` repository dispatch event. This is a separate distribution channel; publishing a crate does not build the binary archive or update the formula.

Follow the [Homebrew packaging guide](../packaging/homebrew/README.md) to build and publish the Apple Silicon archive. After the asset is available, send the tap a `nalcos-release` event with `version` (without `v`) and `aarch64_sha256`. The tap validates the payload, updates the formula, and rejects stale versions or conflicting checksums. Verify installation and `brew test thepushkarp/tap/nalcos` on Apple Silicon.

## Retiring PyPI

The [PyPI project](https://pypi.org/project/nalcos/) is archived. Python releases `0.1.1` and `0.2` remain available for existing users. Archival prevents new uploads while retaining the project name and existing distributions; it can be reversed by an owner in [PyPI settings](https://pypi.org/manage/project/nalcos/settings/). See [PyPI's archival documentation](https://blog.pypi.org/posts/2025-01-30-archival/).

No workflow publishes to PyPI. The legacy Python source is available in Git history and the `v0.2` release. Python installations and external model caches are independent of this checkout.
