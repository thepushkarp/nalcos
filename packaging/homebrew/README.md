# Homebrew alpha packaging

The alpha formula targets Apple Silicon macOS and is distributed through the existing [Homebrew tap](https://github.com/thepushkarp/homebrew-tap):

```sh
brew install thepushkarp/tap/nalcos
```

The [formula](nalcos.rb) installs the prebuilt `v2.0.0-alpha.2` binary, `README.txt`, and `LICENSE`; those three files must be present at the archive's extraction root. It depends on Homebrew Git; NaLCoS requires Git 2.45 or newer. Model assets are selected and installed separately through `nalcos init`.

Release maintenance:

1. Verify that the final reviewed `nalcos-v2.0.0-alpha.2-aarch64-apple-darwin.tar.gz` archive matches the formula's pinned SHA-256. The local archive and checksum are in `dist/`; a rebuilt archive requires an updated checksum.
2. Publish that archive under the `v2.0.0-alpha.2` GitHub release at the exact URL in the formula.
3. Update `Formula/nalcos.rb` in `thepushkarp/homebrew-tap`. Its `nalcos-release` dispatch event accepts `version` (without `v`) and `aarch64_sha256`; publish the asset before dispatching the update.
4. Verify the install command above and `brew test thepushkarp/tap/nalcos` on Apple Silicon.

Fresh `nalcos init` selects MiniLM INT8 on CPU. The version smoke test requires no model download. Linux and Intel Mac packages are outside this formula's scope. See the [Homebrew tap guide](https://docs.brew.sh/How-to-Create-and-Maintain-a-Tap) for the tap layout and naming convention.
