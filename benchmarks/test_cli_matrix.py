"""Do not turn degraded retrieval into a successful evaluation response."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cli_matrix import adapt_response, check_encoder_consistency
from evaluate import metrics
from test_evaluate import query
from test_index_matrix import BENCHMARKS, executable


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.query = {"id": "q", "snapshot": "a" * 40, "query": "Find the change"}
        self.payload = {
            "mode_used": "lexical",
            "history_coverage": {"complete": True},
            "embedding_coverage": {"complete": False},
            "scope": {"roots": [{"oid": "a" * 40}], "shallow": False},
            "output_truncated": False,
            "results": [],
        }

    def test_empty_lexical_result_is_complete_but_not_an_absence_claim(self):
        row = adapt_response(self.query, "lexical", self.payload, 0)
        self.assertTrue(row["complete"])
        self.assertFalse(row["abstained"])

    def test_regression_rows_search_their_window_and_expect_bad_as_root(self):
        regression = {
            **self.query,
            "task": "regression",
            "good": "c" * 40,
            "bad": "d" * 40,
        }
        self.assertIn(
            "snapshot_mismatch",
            adapt_response(regression, "lexical", self.payload, 0)["incomplete_reasons"],
        )
        self.payload["scope"]["roots"] = [{"oid": "d" * 40}]
        self.assertTrue(adapt_response(regression, "lexical", self.payload, 0)["complete"])

    def test_semantic_fallback_and_partial_history_are_incomplete(self):
        self.payload["history_coverage"]["complete"] = False
        row = adapt_response(self.query, "semantic", self.payload, 0)
        self.assertFalse(row["complete"])
        self.assertEqual(
            set(row["incomplete_reasons"]),
            {"mode_fallback", "history_incomplete", "embedding_incomplete"},
        )

    def test_unverified_evidence_and_wrong_snapshot_are_rejected(self):
        self.payload["results"] = [
            {"commit": {"oid": "b" * 40}, "evidence": [{"verified": False}]}
        ]
        self.payload["scope"]["roots"][0]["oid"] = "c" * 40
        row = adapt_response(self.query, "lexical", self.payload, 0)
        self.assertIn("unverified_evidence", row["incomplete_reasons"])
        self.assertIn("snapshot_mismatch", row["incomplete_reasons"])

    def test_output_clipping_keeps_verified_ranking_but_unknown_flag_is_incomplete(
        self,
    ):
        self.payload.update(
            output_truncated=True,
            results=[{"commit": {"oid": "b" * 40}, "evidence": [{"verified": True}]}],
        )
        row = adapt_response(self.query, "lexical", self.payload, 0)
        self.assertTrue(row["complete"])
        self.assertTrue(row["output_truncated"])
        self.assertEqual(row["commits"], ["b" * 40])
        self.payload["output_truncated"] = None
        row = adapt_response(self.query, "lexical", self.payload, 0)
        self.assertFalse(row["complete"])
        self.assertIn("output_truncation_unknown", row["incomplete_reasons"])

    def test_generation_swap_cannot_silently_mix_encoders(self):
        self.query["repo_key"] = "repo"
        self.payload.update(
            mode_used="semantic",
            embedding_coverage={"complete": True},
            active_generation={
                "id": 1,
                "model": "model",
                "revision": "pinned",
                "fingerprint": "first",
            },
            runtime={
                "runtime_version": "1",
                "provider": "Onnx",
                "selected_device": "cpu",
            },
        )
        seen = {}
        first = adapt_response(self.query, "semantic", self.payload, 0)
        check_encoder_consistency(self.query, first, seen)
        self.assertTrue(first["complete"])
        self.payload["active_generation"] = self.payload["active_generation"] | {
            "id": 2
        }
        changed = adapt_response(self.query, "semantic", self.payload, 0)
        check_encoder_consistency(self.query, changed, seen)
        self.assertFalse(changed["complete"])
        self.assertIn("encoder_changed_during_matrix", changed["incomplete_reasons"])

    def test_scope_selection_repeats_and_evidence_with_fake_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            questions = [
                query("snapshot", repo_key="repo"),
                query("range", repo_key="repo")
                | {"snapshot": "c" * 40, "scope_range": "d" * 40 + ".." + "c" * 40},
                query("regression", repo_key="repo")
                | {"task": "regression", "good": "d" * 40, "bad": "e" * 40},
            ]
            for q in questions:
                q.update(gold_paths=["old.rs"], supporting_hunk_contains="needle")
            manifest = work / "queries.jsonl"
            manifest.write_text("".join(json.dumps(q) + "\n" for q in questions))
            config = work / "config.toml"
            config.write_text("version = 1\n")
            data = work / "data"
            data.mkdir()
            (data / "gold").write_text("b" * 40)
            mapping = work / "repositories.json"
            mapping.write_text(
                json.dumps(
                    {
                        "repo": {
                            "path": str(work),
                            "config": str(config),
                            "data_dir": str(data),
                        }
                    }
                )
            )
            binary = executable(
                work / "nalcos",
                """
                import json, os, sys
                from pathlib import Path
                args = sys.argv[1:]
                scope = args[args.index("--range") + 1] if "--range" in args else args[args.index("--ref") + 1]
                roots = {"a" * 40: "a" * 40, "d" * 40 + ".." + "c" * 40: "c" * 40, "d" * 40 + ".." + "e" * 40: "e" * 40}
                gold = (Path(os.environ["NALCOS_DATA_DIR"]) / "gold").read_text()
                print(json.dumps({
                    "mode_used": "lexical", "history_coverage": {"complete": True},
                    "embedding_coverage": {"complete": False},
                    "scope": {"roots": [{"oid": roots[scope]}], "shallow": False},
                    "output_truncated": False,
                    "results": [{"commit": {"oid": gold}, "evidence": [
                        {"verified": True, "new_path": {"display": "new.rs"}, "old_path": {"display": "old.rs"}, "excerpt": "needle"}
                    ]}],
                }))
            """,
            )
            output = work / "out"
            command = [
                sys.executable,
                str(BENCHMARKS / "cli_matrix.py"),
                "--binary",
                str(binary),
                "--manifest",
                str(manifest),
                "--repositories",
                str(mapping),
                "--output",
                str(output),
                "--modes",
                "lexical",
                "--device",
                "metal",
                "--repeats",
                "2",
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for filename, state in (
                ("lexical.rankings.jsonl", "cold"),
                ("lexical.repeat-1.rankings.jsonl", "warm"),
            ):
                rows = [
                    json.loads(line)
                    for line in (output / filename).read_text().splitlines()
                ]
                scores = metrics(questions, rows)
                self.assertTrue(scores["coverage_complete"])
                self.assertEqual(scores["mrr"], 1)
                self.assertEqual(scores["hunk_hit"]["10"], 1)
                self.assertTrue(all(r["cache_state"] == state for r in rows))
            summary = json.loads((output / "summary.json").read_text())["modes"][
                "lexical"
            ]
            self.assertEqual(summary["by_cache_state"]["cold"]["samples"], 3)
            self.assertEqual(summary["by_cache_state"]["warm"]["samples"], 3)
            self.assertGreater(summary["by_cache_state"]["warm"]["p95_seconds"], 0)

    def test_missing_repo_key_names_query_and_mapping_input(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest, mapping = work / "queries.jsonl", work / "repositories.json"
            manifest.write_text(json.dumps(query("missing-key")) + "\n")
            mapping.write_text("{}")
            result = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARKS / "cli_matrix.py"),
                    "--binary",
                    "/unused",
                    "--manifest",
                    str(manifest),
                    "--repositories",
                    str(mapping),
                    "--output",
                    str(work / "out"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("missing-key", result.stderr)
            self.assertIn("repo_key", result.stderr)
            self.assertNotIn("KeyError", result.stderr)
            self.assertFalse((work / "out").exists())


if __name__ == "__main__":
    unittest.main()
