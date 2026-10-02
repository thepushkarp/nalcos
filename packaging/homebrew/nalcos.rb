class Nalcos < Formula
  desc "Search local Git history in natural language"
  homepage "https://github.com/thepushkarp/nalcos"
  url "https://github.com/thepushkarp/nalcos/releases/download/v2.0.0-alpha.1/nalcos-v2.0.0-alpha.1-aarch64-apple-darwin.tar.gz"
  version "2.0.0-alpha.1"
  sha256 "ac40d507ec476aee5f91283ae228ff2452eb5d236542eca0979d70b3fbbbdcde"
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
