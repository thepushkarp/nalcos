class Nalcos < Formula
  desc "Search local Git history in natural language"
  homepage "https://github.com/thepushkarp/nalcos"
  version "2.0.0-alpha.1"
  url "https://github.com/thepushkarp/nalcos/releases/download/v#{version}/nalcos-v#{version}-aarch64-apple-darwin.tar.gz"
  sha256 "8c941b1d273c27d23e87a2043836e2508d57f4de6e9f8951e064baeaba019aa9"
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
