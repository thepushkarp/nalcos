"""Behavioral coverage for metrics and fail-closed release selection."""

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluate import (
    cohort_digest,
    compare,
    genuine_holdout,
    metrics,
    qualify,
    validate_manifest,
)


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


def paired_ranking(identifier, rank, limit=10):
    commits = [f"{i:040x}" for i in range(1, rank)] + ["b" * 40] if rank else []
    return ranking(identifier, commits) | {"rank_limit": limit}


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
    def test_counts_and_pooled_metrics_use_query_denominators(self):
        kinds = [
            "natural",
            "paraphrase",
            "vocabulary",
            "symbol",
            "path",
            "mixed",
            "regression",
        ]
        questions = [
            query(
                kind,
                query_kind=kind,
                gold_paths=["file.rs"],
                supporting_hunk_contains="needle",
            )
            for kind in kinds
        ]
        rows = [
            paired_ranking(kind, 2 if kind == "natural" else 1) for kind in kinds[:-1]
        ]
        for row in rows:
            row["evidence"] = [
                {"commit": "b" * 40, "paths": ["file.rs"], "excerpt": "needle"}
            ]
        result = metrics(questions, rows)
        self.assertEqual(
            result["hit_rate_counts"],
            {
                "1": {"hits": 5, "n": 7},
                "5": {"hits": 6, "n": 7},
                "10": {"hits": 6, "n": 7},
            },
        )
        for group, hits, n in (
            (result, 6, 7),
            (result["pooled"]["prose"], 2, 2),
            (result["pooled"]["exact_term"], 4, 5),
        ):
            for metric in ("hit_rate", "path_hit", "hunk_hit"):
                with self.subTest(group=n, metric=metric):
                    self.assertEqual(
                        group[f"{metric}_counts"]["10"], {"hits": hits, "n": n}
                    )
                    self.assertEqual(group[metric]["10"], hits / n)
        self.assertEqual(
            result["per_query_kind"]["natural"]["hit_rate_counts"]["1"],
            {"hits": 0, "n": 1},
        )
        empty = metrics([query()], [])
        self.assertEqual(
            empty["pooled"]["prose"]["hit_rate_counts"]["10"], {"hits": 0, "n": 0}
        )
        self.assertIsNone(empty["pooled"]["prose"]["hit_rate"]["10"])

    def test_ndcg_binary_multi_gold_and_query_kinds(self):
        questions = [
            query("natural", gold=["b" * 40, "c" * 40], query_kind="natural"),
            query("plain"),
        ]
        result = metrics(
            questions,
            [
                ranking("natural", ["d" * 40, "b" * 40, "c" * 40]),
                paired_ranking("plain", 11, 20),
            ],
        )
        expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(result["ndcg@10"], expected / 2)
        self.assertEqual(result["per_query_kind"]["unspecified"]["mrr"], 1 / 11)
        self.assertEqual(result["per_query_kind"]["natural"]["recall"]["5"], 1)
        self.assertAlmostEqual(result["mrr"], (1 / 2 + 1 / 11) / 2)

    def test_evidence_needs_gold_commit_and_matching_path_and_hunk(self):
        question = query(gold_paths=["old.rs"], supporting_hunk_contains="needle")
        row = ranking(commits=["c" * 40, "b" * 40])
        row["evidence"] = [
            {"commit": "c" * 40, "paths": ["old.rs"], "excerpt": "needle"},
            {"commit": "b" * 40, "paths": ["old.rs", "new.rs"], "excerpt": "other"},
            {"commit": "b" * 40, "paths": ["wrong.rs"], "excerpt": "needle"},
        ]
        result = metrics([question], [row])
        self.assertEqual(result["path_hit"], {"1": 0, "5": 1, "10": 1})
        self.assertEqual(result["path_hit_counts"]["5"], {"hits": 1, "n": 1})
        self.assertEqual(result["hunk_hit"]["10"], 0)
        self.assertEqual(result["hunk_hit_counts"]["10"], {"hits": 0, "n": 1})
        row["evidence"][1]["excerpt"] = "prefix needle suffix"
        self.assertEqual(metrics([question], [row])["hunk_hit"]["5"], 1)
        self.assertIsNone(metrics([question], [ranking()])["path_hit"]["10"])
        missing = query(
            "missing", gold_paths=["old.rs"], supporting_hunk_contains="needle"
        )
        self.assertEqual(metrics([question, missing], [row])["hunk_hit"]["10"], 0.5)
        unlabeled = query("unlabeled")
        with_unlabeled = metrics([question, unlabeled], [row])
        self.assertEqual(with_unlabeled["hit_rate_counts"]["10"], {"hits": 1, "n": 2})
        self.assertEqual(with_unlabeled["hunk_hit_counts"]["10"], {"hits": 1, "n": 1})
        with self.assertRaisesRegex(ValueError, "evidence"):
            metrics([question], [ranking() | {"evidence": row["evidence"]}])

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


class ComparisonTests(unittest.TestCase):
    def test_paired_ranks_flips_and_pooled_groups(self):
        ranks = [
            ("gained", "natural", None, 3, "win"),
            ("lost", "paraphrase", 4, None, "loss"),
            ("absent", "natural", None, None, "tie"),
            ("same", "vocabulary", 2, 2, "tie"),
            ("improved", "symbol", 8, 2, "win"),
            ("worsened", "mixed", 2, 9, "loss"),
            ("beyond-ten", "path", 11, 12, "loss"),
        ]
        questions = [
            query(identifier, query_kind=kind) for identifier, kind, *_ in ranks
        ]
        baseline = [
            paired_ranking(identifier, a, 20) for identifier, _, a, _, _ in ranks
        ]
        candidate = [
            paired_ranking(identifier, b, 20) for identifier, _, _, b, _ in ranks
        ]
        questions[3]["gold_commits"].append("c" * 40)
        baseline[3]["commits"] = ["d" * 40, "c" * 40, "b" * 40]
        result = compare(questions, baseline, candidate, bootstrap=100)
        overall = result["overall"]
        self.assertEqual(
            (overall["n"], overall["wins"], overall["losses"], overall["ties"]),
            (7, 2, 3, 2),
        )
        self.assertEqual(overall["hit@10"], {"gained": 1, "lost": 1, "net": 0})
        self.assertEqual(overall["delta_hit@10"], 0)
        self.assertAlmostEqual(overall["delta_macro_hit@10"], -1 / 12)
        expected_mrr = (
            sum((1 / b if b else 0) - (1 / a if a else 0) for _, _, a, b, _ in ranks)
            / 7
        )
        self.assertAlmostEqual(overall["delta_mrr"], expected_mrr)
        self.assertEqual(result["pooled"]["prose"]["n"], 3)
        self.assertEqual(result["pooled"]["exact_term"]["n"], 4)
        self.assertEqual(result["per_query_kind"]["symbol"]["wins"], 1)
        details = {row["query_id"]: row for row in result["details"]}
        for identifier, _, a, b, outcome in ranks:
            with self.subTest(query_id=identifier):
                self.assertEqual(details[identifier]["baseline_rank"], a)
                self.assertEqual(details[identifier]["candidate_rank"], b)
                self.assertEqual(details[identifier]["outcome"], outcome)

    def test_seeded_paired_bootstrap_and_stratified_macro(self):
        questions = [
            query(f"q{i}", query_kind="natural" if i < 3 else "paraphrase")
            for i in range(4)
        ]
        baseline = [paired_ranking(f"q{i}", None if i < 3 else 1) for i in range(4)]
        candidate = [paired_ranking(f"q{i}", 1 if i < 3 else None) for i in range(4)]
        result = compare(questions, baseline, candidate, bootstrap=100, seed=19)
        repeated = compare(
            questions[::-1], baseline[::-1], candidate[::-1], bootstrap=100, seed=19
        )
        self.assertEqual(result, repeated)
        for group in (result["overall"], result["pooled"]["prose"]):
            self.assertEqual(group["delta_hit@10"], 0.5)
            self.assertEqual(group["delta_macro_hit@10"], 0)
            self.assertEqual(group["ci95"]["delta_macro_hit@10"], [0, 0])
            self.assertEqual(group["ci95"]["delta_mrr"], group["ci95"]["delta_hit@10"])
        # With one question per kind, resampling within each kind preserves both.
        tiny = compare(questions[2:], baseline[2:], candidate[2:], bootstrap=4, seed=5)
        # The four paired means are -1, -1, -1, 1; interpolate at positions .075/2.925.
        self.assertEqual(tiny["overall"]["ci95"]["delta_hit@10"][0], -1)
        self.assertAlmostEqual(tiny["overall"]["ci95"]["delta_hit@10"][1], 0.85)

    def test_negatives_unjudged_and_split_are_separate_from_rank_metrics(self):
        questions = [
            query("positive", query_kind="natural"),
            query("heldout", "heldout"),
        ]
        questions += [
            query(identifier, query_kind="negative")
            | {"answerable": False, "gold_commits": []}
            for identifier in ("gained", "lost", "partial")
        ]
        questions.append(
            query("unjudged")
            | {"answerable": None, "gold_commits": [], "label_status": "unjudged"}
        )
        baseline = [paired_ranking(q["id"], None) for q in questions]
        candidate = [dict(row) for row in baseline]
        baseline[3]["abstained"] = True
        candidate[2]["abstained"] = True
        candidate[4].update(abstained=True, complete=False)
        result = compare(
            questions, baseline, candidate, split="development", bootstrap=20
        )
        self.assertEqual(result["questions"], 5)
        self.assertEqual(result["overall"]["n"], 1)
        self.assertEqual(result["unjudged_questions"], 1)
        self.assertNotIn("negative", result["per_query_kind"])
        negatives = result["negative_abstentions"]
        self.assertEqual(
            {key: value for key, value in negatives.items() if key != "details"},
            {
                "n": 3,
                "baseline_correct": 1,
                "candidate_correct": 1,
                "gained": 1,
                "lost": 1,
                "net": 0,
                "ties": 1,
            },
        )
        empty = compare(
            questions, baseline, candidate, split="exploratory", bootstrap=20
        )
        self.assertIsNone(empty["overall"]["delta_mrr"])
        self.assertIsNone(empty["overall"]["ci95"]["delta_mrr"])
        self.assertFalse(empty["acceptance"]["accepted"])

    def test_acceptance_requires_each_clause(self):
        prose = [("natural", 2, 1), ("paraphrase", 2, 1)]
        exact = [("vocabulary", 1, 1)]
        cases = [
            (None, prose + exact),
            (
                "prose_wins_gt_losses",
                [("natural", 10, 1)] * 10 + [("natural", 9, 10)] * 10 + exact,
            ),
            (
                "prose_mrr_ci_lower_gt_zero",
                [("natural", 10, 9)] * 2 + [("natural", 1, None)] + exact,
            ),
            (
                "exact_term_hit10_ci_lower_ge_minus_0.05",
                prose + [("vocabulary", 1, None)],
            ),
            # Dilute two losses in one kind across a stable exact-term pool so
            # only the per-kind guard, not the pooled interval, rejects them.
            (
                "no_query_kind_hit10_net_loss_gt_one",
                prose + [("vocabulary", 1, None)] * 2 + [("symbol", 1, 1)] * 200,
            ),
        ]
        for failed_clause, ranks in cases:
            questions = [
                query(str(i), query_kind=kind) for i, (kind, _, _) in enumerate(ranks)
            ]
            baseline = [paired_ranking(str(i), a) for i, (_, a, _) in enumerate(ranks)]
            candidate = [paired_ranking(str(i), b) for i, (_, _, b) in enumerate(ranks)]
            with self.subTest(failed_clause=failed_clause):
                acceptance = compare(questions, baseline, candidate, bootstrap=100)[
                    "acceptance"
                ]
                self.assertEqual(acceptance["accepted"], failed_clause is None)
                for name, passed in acceptance["clauses"].items():
                    self.assertEqual(passed, name != failed_clause, name)


if __name__ == "__main__":
    unittest.main()
