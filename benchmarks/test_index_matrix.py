"""Index orchestration uses local Git fixtures and a fake NaLCoS executable."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from harness import load_repositories, rss_bytes, run_measured
from index_matrix import phase_timings, read_payload, summarize

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
    def test_manifest_validation_and_checked_in_copy(self):
        checked_in = load_repositories(BENCHMARKS / "repos.json")
        self.assertEqual(
            set(checked_in), {"etymology", "nalcos", "toki", "pgvector", "vllm"}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repos.json"
            for field, value in (
                ("snapshot", "abc"),
                ("split", "dev"),
                ("semantic_range", "a..b"),
                ("semantic_range", "a" * 40 + "..." + "b" * 40),
            ):
                repositories = repo_manifest(path, "a" * 40, "b" * 40)
                repositories["small"][field] = value
                path.write_text(
                    json.dumps({"version": 1, "repositories": repositories})
                )
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(
                        ValueError, field if field != "split" else "split"
                    ),
                ):
                    load_repositories(path)

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
                assert data.parent == cache.parent
                assert os.environ["GIT_NO_LAZY_FETCH"] == "1"
                assert os.environ["GIT_NO_REPLACE_OBJECTS"] == "1"
                command = next(c for c in ("init", "sync", "status") if c in args)
                if command != "status":
                    assert not list(data.iterdir())
                    assert not list(cache.iterdir())
                    assert args.index("--device") < args.index(command)
                    assert "--ref" not in args[:args.index(command)]
                    assert "--range" not in args[:args.index(command)]
                    assert ("--model" in args) == (command == "init")
                    (data / "receipt.json").write_text(json.dumps(args))
                    (cache / "probe").write_text("fresh")
                    print("phase=extract duration=12ms", file=sys.stderr)
                previous = json.loads((data / "receipt.json").read_text())
                semantic = "init" in previous
                payload = {
                    "command": command, "commits_added": 2, "documents_embedded": 4 if semantic else 0,
                    "history_coverage": {"total": 2, "indexed": 2, "complete": True},
                    "embedding_coverage": {"total": 4, "indexed": 4 if semantic else 0, "complete": semantic},
                    "indexing_runtime": {"selected_device": previous[previous.index("--device") + 1]} if semantic else None,
                }
                if command == "status":
                    payload["counts"] = {"commits": 2, "documents": 4, "vectors": 6 if semantic else 0}
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
                "auto",
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
            self.assertEqual(len(rows), 12)
            self.assertEqual(
                {row["tier"] for row in rows},
                {"semantic-full", "semantic-range", "lexical-full"},
            )
            self.assertTrue(
                all(
                    row["peak_rss_bytes"] > 0
                    and row["docs_per_second"] > 0
                    and row["storage_bytes"] > 0
                    for row in rows
                )
            )
            self.assertTrue(
                all(row["phase_seconds"] == {"extract": [0.012]} for row in rows)
            )
            for row in rows:
                self.assertTrue((output / row["stderr_path"]).is_file())
                self.assertEqual(
                    row["vectors"], 0 if row["tier"] == "lexical-full" else 6
                )
                self.assertEqual(
                    "--range" in row["command"], row["tier"] == "semantic-range"
                )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["summaries"]), 6)
            self.assertTrue(
                all(group["successful_runs"] == 2 for group in summary["summaries"])
            )
            refused = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(refused.returncode, 2)
            self.assertIn("exists", refused.stderr)

    def test_wait4_handles_large_output_failure_timeout_and_rss_units(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            binary = executable(
                work / "child",
                """
                import sys, time
                print("x" * 262144, flush=True)
                print("stderr saved", file=sys.stderr, flush=True)
                if "sleep" in sys.argv:
                    time.sleep(20)
                raise SystemExit(7)
            """,
            )
            for args, timeout, expected in (([], 5, 7), (["sleep"], 0.15, 124)):
                measured = run_measured(
                    [str(binary), *args],
                    work / "stdout",
                    work / "stderr",
                    env=os.environ.copy(),
                    timeout=timeout,
                )
                self.assertEqual(measured["exit_code"], expected)
                self.assertEqual(measured["timed_out"], expected == 124)
                self.assertGreater(measured["peak_rss_bytes"], 0)
                if expected == 7:
                    self.assertGreater((work / "stdout").stat().st_size, 262144)
            with patch("harness.sys.platform", "darwin"):
                self.assertEqual(rss_bytes(1234), 1234)
            with patch("harness.sys.platform", "linux"):
                self.assertEqual(rss_bytes(1234), 1234 * 1024)

    def test_progress_is_not_a_phase_duration_and_failures_do_not_improve_medians(self):
        self.assertEqual(phase_timings("Indexing history: 32/64 commits"), {})
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

    def test_successful_unrecognized_json_fails_with_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.stdout"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "index.stdout"):
                read_payload(path, {"exit_code": 0})
            path.write_text("not JSON")
            measurement = {"exit_code": 1}
            self.assertEqual(read_payload(path, measurement), {})
            self.assertIn(str(path), measurement["json_error"])


if __name__ == "__main__":
    unittest.main()
