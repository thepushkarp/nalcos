class Nalcos < Formula
  desc "Search local Git history in natural language"
  homepage "https://github.com/thepushkarp/nalcos"
  version "2.0.0-alpha.3"
  url "https://github.com/thepushkarp/nalcos/releases/download/v#{version}/nalcos-v#{version}-aarch64-apple-darwin.tar.gz"
  sha256 "5a6905cf069aa5f43cc9049e1718940508dc903e777596a3d8d43a3ee38adec6"
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
