class Nalcos < Formula
  desc "Search local Git history in natural language"
  homepage "https://github.com/thepushkarp/nalcos"
  version "2.0.0-alpha.2"
  url "https://github.com/thepushkarp/nalcos/releases/download/v#{version}/nalcos-v#{version}-aarch64-apple-darwin.tar.gz"
  sha256 "056e99da98374a0b4bd715e12fb0b1bedea7ca8c62a2d9ce669218ce1f009aa0"
  license "MIT"

  depends_on :macos
  depends_on arch: :arm64
  depends_on "git"

  def install
    bin.install "nalcos"
    doc.install "README.txt"
    prefix.install "LICENSE"
  end

  test do
    assert_match "nalcos #{version}", shell_output("#{bin}/nalcos --version")
  end
end
