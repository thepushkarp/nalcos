"""Index orchestration uses local Git fixtures and a fake NaLCoS executable."""

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from index_matrix import summarize

BENCHMARKS = Path(__file__).resolve().parent


def executable(path, source):
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(source))
    path.chmod(0o755)
    return path


def local_repo(path):
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main", str(path)], check=True
    )
    script = (
        "commit refs/heads/main\ncommitter Fixture <fixture@example.invalid> 1700000000 +0000\n"
        "data 4\none\nM 100644 inline file.txt\ndata 4\none\n\n"
        "commit refs/heads/main\ncommitter Fixture <fixture@example.invalid> 1700000001 +0000\n"
        "data 4\ntwo\nM 100644 inline file.txt\ndata 4\ntwo\n\n"
    )
    subprocess.run(
        ["git", "-C", str(path), "fast-import", "--quiet"],
        input=script,
        text=True,
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-list", "main"], text=True
    ).splitlines()


def repo_manifest(path, head, base, keys=("small", "large")):
    repositories = {
        key: {
            "url": "https://example.invalid/" + key,
            "snapshot": head,
            "language": "Rust",
            "split": "development",
            "semantic_range": base + ".." + head if key == "large" else None,
        }
        for key in keys
    }
    path.write_text(json.dumps({"version": 1, "repositories": repositories}))
    return repositories


class IndexTests(unittest.TestCase):
    def test_matrix_isolates_every_run_and_separates_range_tiers(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            clones = work / "clones"
            clones.mkdir()
            head, base = local_repo(clones / "small")
            local_repo(clones / "large")
            manifest = work / "repos.json"
            repo_manifest(manifest, head, base)
            binary = executable(
                work / "nalcos",
                """
                import json, os, sys
                from pathlib import Path
                args = sys.argv[1:]
                data = Path(os.environ["NALCOS_DATA_DIR"])
                cache = Path(os.environ["NALCOS_RUNTIME_CACHE"])
                command = next(c for c in ("init", "sync", "status") if c in args)
                if command != "status":
                    assert not list(data.iterdir())
                    assert not list(cache.iterdir())
                    # Interpret only the scope and operation that determine coverage.
                    scope = args[args.index("--range") + 1] if "--range" in args else args[args.index("--ref") + 1]
                    (data / "receipt.json").write_text(json.dumps({"semantic": command == "init", "scope": scope}))
                    (cache / "probe").write_text("fresh")
                previous = json.loads((data / "receipt.json").read_text())
                semantic = previous["semantic"]
                ranged = ".." in previous["scope"]
                payload = {
                    "command": command, "commits_added": 2, "documents_embedded": 4 if semantic else 0,
                    "history_coverage": {"total": 1 if ranged else 2, "indexed": 1 if ranged else 2, "complete": True},
                    "embedding_coverage": {"total": 4, "indexed": 4 if semantic else 0, "complete": semantic},
                }
                if command == "status":
                    payload["counts"] = {"commits": 1 if ranged else 2, "documents": 4, "vectors": 6 if semantic else 0}
                print(json.dumps(payload))
            """,
            )
            output = work / "output"
            command = [
                sys.executable,
                str(BENCHMARKS / "index_matrix.py"),
                "--binary",
                str(binary),
                "--repos",
                str(manifest),
                "--clones",
                str(clones),
                "--devices",
                "cpu",
                "--repeats",
                "2",
                "--timeout",
                "2",
                "--offline",
                "--output",
                str(output),
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [
                json.loads(line)
                for line in (output / "runs.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 6)
            self.assertTrue(
                all(
                    row["peak_rss_bytes"] > 0
                    and row["docs_per_second"] > 0
                    and row["storage_bytes"] > 0
                    for row in rows
                )
            )
            for row in rows:
                self.assertEqual(
                    row["vectors"], 0 if row["tier"] == "lexical-full" else 6
                )
                self.assertEqual(
                    row["commits"], 1 if row["tier"] == "semantic-range" else 2
                )
            self.assertEqual(
                {(r["repo_key"], r["tier"]) for r in rows},
                {
                    ("small", "semantic-full"),
                    ("large", "lexical-full"),
                    ("large", "semantic-range"),
                },
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(
                all(group["successful_runs"] == 2 for group in summary["summaries"])
            )
            refused = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(refused.returncode, 2)
            self.assertIn("exists", refused.stderr)

    def test_failed_runs_do_not_improve_medians(self):
        row = {
            "repo_key": "r",
            "device": "cpu",
            "tier": "semantic-full",
            "exit_code": 0,
            "status_exit_code": 0,
            "elapsed_seconds": 10,
            "docs_per_second": 4,
            "peak_rss_bytes": 100,
            "storage_bytes": 200,
        }
        summary = summarize([row, row | {"exit_code": 1, "elapsed_seconds": 0.01}])[0]
        self.assertEqual(summary["medians"]["elapsed_seconds"], 10)
        self.assertEqual(summary["successful_runs"], 1)


if __name__ == "__main__":
    unittest.main()
