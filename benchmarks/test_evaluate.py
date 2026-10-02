"""Behavioral coverage for metrics and fail-closed release selection."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evaluate import cohort_digest, genuine_holdout, metrics, qualify, validate_manifest


def query(identifier="q", split="development", gold=None, **extra):
    return dict(
        id=identifier,
        repo="repo-1",
        snapshot="a" * 40,
        query="Find the change",
        split=split,
        origin="prospective",
        frozen_before_tuning=True,
        task="find",
        answerable=True,
        gold_commits=gold or ["b" * 40],
        **extra,
    )


def ranking(identifier="q", commits=None, complete=True, abstained=False):
    return {
        "query_id": identifier,
        "commits": commits or [],
        "complete": complete,
        "abstained": abstained,
        "snapshot": "a" * 40,
        "query_sha256": hashlib.sha256(b"Find the change").hexdigest(),
    }


def reviewed_query(identifier="q", split="heldout"):
    return query(identifier, split) | {
        "provenance": {
            "passed": True,
            "references": ["reviewed-question-source-and-lock.json"],
            "source_kind": "historical_user_question",
            "answer_independent": True,
            "locked_before_answer_inspection": True,
            "locked_before_ranking": True,
            "isolation_reviewed": True,
        },
        "label_review": {"passed": True, "references": ["reviewed-gold.json"]},
        "origin": "retrospective",
    }


def declaration(rows, **extra):
    return {
        "passed": True,
        "references": ["reviewed-cohort-lock.json"],
        "query_ids": [row["id"] for row in rows],
        "query_identities_sha256": cohort_digest(rows),
        **extra,
    }


class EvaluationTests(unittest.TestCase):
    def test_multi_gold_recall_differs_from_hit_rate(self):
        result = metrics(
            [query(gold=["b" * 40, "c" * 40])], [ranking(commits=["d" * 40, "b" * 40])]
        )
        self.assertEqual(result["recall"]["5"], 0.5)
        self.assertEqual(result["hit_rate"]["5"], 1)
        self.assertEqual(result["mrr"], 0.5)

    def test_incomplete_negative_is_not_correct_abstention(self):
        row = query()
        row.update(answerable=False, gold_commits=[])
        result = metrics([row], [ranking(complete=False, abstained=True)])
        self.assertEqual(result["negative_correct_abstentions"], 0)
        self.assertFalse(result["coverage_complete"])
        self.assertIsNone(result["mrr"])

    def test_unjudged_question_is_not_a_negative_or_quality_denominator(self):
        uncertain = query("uncertain")
        uncertain.update(answerable=None, gold_commits=[], label_status="unjudged")
        rows = [query(), uncertain]
        result = metrics(rows, [ranking(commits=["b" * 40]), ranking("uncertain")])
        self.assertEqual(result["mrr"], 1)
        self.assertEqual(result["negative_questions"], 0)
        self.assertEqual(result["unjudged_questions"], 1)
        self.assertFalse(result["labels_complete"])
        self.assertIn(
            "unjudged_queries_present", qualify(rows, [], Path("."))["release_blockers"]
        )

    def test_duplicate_rankings_and_missing_regression_range_rejected(self):
        with self.assertRaises(ValueError):
            metrics([query()], [ranking(), ranking()])
        row = query()
        row["task"] = "regression"
        with self.assertRaises(ValueError):
            validate_manifest([row])

    def test_ranking_cannot_be_reused_after_query_or_snapshot_changes(self):
        for field, value in (
            ("query", "Find a different change"),
            ("snapshot", "f" * 40),
        ):
            row = query()
            row[field] = value
            with self.assertRaises(ValueError):
                metrics([row], [ranking(commits=["b" * 40])])
        row = query()
        row.update(task="regression", good="c" * 40, bad="d" * 40)
        with self.assertRaises(ValueError):
            metrics([row], [ranking(commits=["b" * 40])])

    def test_retrospective_pilot_never_qualifies(self):
        rows = [query(str(i), "heldout") for i in range(30)]
        for i, row in enumerate(rows):
            row.update(repo=f"repo-{i % 3}", origin="retrospective")
        report = qualify(rows, [], Path("."))
        self.assertFalse(report["model_qualified"])
        self.assertFalse(report["release_qualified"])
        self.assertEqual(report["genuine_heldout_questions"], 0)
        self.assertIn(
            "heldout_provenance_or_labels_unverified",
            report["release_blockers"],
        )

    def test_provenance_and_negative_review_are_required_not_prospective_origin(self):
        row = reviewed_query()
        self.assertTrue(genuine_holdout(row))
        self.assertFalse(genuine_holdout(query(split="heldout")))
        for field, value in (
            ("source_kind", "answer_derived"),
            ("answer_independent", False),
            ("locked_before_answer_inspection", False),
            ("locked_before_ranking", False),
            ("isolation_reviewed", False),
        ):
            changed = row | {"provenance": row["provenance"] | {field: value}}
            self.assertFalse(genuine_holdout(changed))
        negative = row | {"answerable": False, "gold_commits": []}
        self.assertFalse(genuine_holdout(negative))
        negative["label_review"] = row["label_review"] | {"absence_reviewed": True}
        self.assertTrue(genuine_holdout(negative))

    def test_speed_tie_selects_smaller_and_failed_holdout_does_not_promote(self):
        rows = [query("dev")]
        for i in range(30):
            row = reviewed_query(f"held-{i}")
            row["repo"] = f"repo-{i % 3}"
            rows.append(row)
        rows[-1].update(answerable=False, gold_commits=[])
        rows[-1]["label_review"]["absence_reviewed"] = True
        cohort = declaration(
            rows, frozen_before_tuning=True, frozen_before_ranking=True
        )
        diagnostic = query("unrelated-diagnostic", "exploratory")
        diagnostic.update(answerable=None, gold_commits=[], label_status="unjudged")
        manifest = rows + [diagnostic]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            candidates = []
            for name, size, elapsed in [("fast", 200, 1.0), ("small", 100, 1.09)]:
                ranks = [
                    ranking(
                        q["id"],
                        (
                            ["b" * 40]
                            if name == "fast" or q["split"] == "development"
                            else ["c" * 40]
                        )
                        if q["answerable"]
                        else [],
                        abstained=not q["answerable"],
                    )
                    for q in rows
                ]
                (base / f"{name}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in ranks)
                )
                evidence = {"passed": True, "references": ["reviewed.json"]}
                candidates.append(
                    {
                        "id": name,
                        "weight_bytes": size,
                        "rankings": f"{name}.jsonl",
                        "license": evidence,
                        "correctness": evidence,
                        "cpu_portability": evidence
                        | {"platforms": ["darwin-arm64", "linux-x86_64"]},
                        "cpu_index_seconds": [elapsed] * 3,
                        "timing_cohort": "same-inputs",
                    }
                )
            ranks = [
                ranking(q["id"], q["gold_commits"], abstained=not q["answerable"])
                for q in rows
            ]
            (base / "fast.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in ranks)
            )
            # Deterministic quality-reference ties choose the lexically greatest id.
            candidates[0]["id"] = "z-fast"
            result = qualify(manifest, candidates, base, cohort=cohort)
            self.assertEqual(result["development_choice"], "small")
            self.assertEqual(result["frozen_quality_reference"], "z-fast")
            self.assertFalse(result["model_qualified"])
            self.assertFalse(result["release_qualified"])
            self.assertIn(
                "heldout_recall10_confirmation_failed", result["release_blockers"]
            )
            (base / "small.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in ranks)
            )
            model_only = qualify(manifest, candidates, base, cohort=cohort)
            self.assertTrue(model_only["model_qualified"])
            self.assertFalse(model_only["product_qualified"])
            self.assertFalse(model_only["release_qualified"])
            self.assertEqual(model_only["genuine_heldout_questions"], 30)
            self.assertEqual(model_only["genuine_heldout_positive_questions"], 29)
            self.assertEqual(model_only["genuine_heldout_negative_questions"], 1)
            self.assertEqual(model_only["evaluation_cohort"]["excluded_questions"], 1)
            self.assertEqual(model_only["candidates"][0]["heldout"]["mrr"], 1)
            workflow = {
                "passed": True,
                "references": ["reviewed-agent-results.json"],
                "same_agent": True,
                "independent_heldout": True,
                "protocol": declaration(
                    rows[1:3],
                    frozen_before_trials=True,
                    sample_rationale="Unit-test fixture; not evidence of real sample adequacy.",
                ),
                "paired_questions": 2,
                "success_rate_delta": 0,
                "median_time_reduction": 0.30,
                "dogfood_span_days": 14,
                "dogfood_active_days": 2,
            }
            confirmed = qualify(
                manifest, candidates, base, workflow=workflow, cohort=cohort
            )
            self.assertTrue(confirmed["model_qualified"])
            self.assertTrue(confirmed["product_qualified"])
            self.assertTrue(confirmed["release_qualified"])
            too_short = qualify(
                manifest,
                candidates,
                base,
                workflow=workflow | {"dogfood_span_days": 13.99},
                cohort=cohort,
            )
            self.assertIn("two_week_dogfood_unverified", too_short["release_blockers"])
            self.assertTrue(too_short["model_qualified"])
            self.assertFalse(too_short["product_qualified"])
            for invalid_cohort in (
                None,
                cohort | {"frozen_before_ranking": False},
                cohort | {"query_ids": cohort["query_ids"][:-1]},
            ):
                invalid = qualify(manifest, candidates, base, cohort=invalid_cohort)
                self.assertFalse(invalid["model_qualified"])
                self.assertIn(
                    "evaluation_cohort_lock_unverified", invalid["model_blockers"]
                )
            candidates[1]["timing_cohort"] = "different-chunking"
            incomparable = qualify(
                manifest, candidates, base, workflow=workflow, cohort=cohort
            )
            self.assertFalse(incomparable["model_qualified"])
            self.assertTrue(incomparable["product_qualified"])
            self.assertFalse(incomparable["release_qualified"])
            self.assertIn(
                "cpu_timing_cohorts_are_not_comparable",
                incomparable["release_blockers"],
            )


if __name__ == "__main__":
    unittest.main()
