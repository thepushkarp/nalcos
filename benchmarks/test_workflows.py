"""Paired workflow evaluation must not reward failures or mismatched conditions."""

import unittest

from test_evaluate import declaration, query, ranking, reviewed_query
from workflows import evaluate_trials


def trial(condition, seconds=20, **extra):
    return (
        ranking()
        | {
            "condition": condition,
            "agent_fingerprint": "same-model-and-prompt",
            "isolation_id": condition,
            "budget_seconds": 60,
            "budget_tokens": 1000,
            "elapsed_seconds": seconds,
            "tokens_used": 500,
            "outcome": "completed",
            "order": 0 if condition == "git_rg" else 1,
            "started_at": "2026-01-01T10:00:00+00:00",
            "evidence_review": {
                "passed": True,
                "blind_to_condition": True,
                "reviewer_id": "independent-judge",
                "references": ["review.json"],
                "verified_commits": ["b" * 40],
            },
        }
        | extra
    )


class WorkflowTests(unittest.TestCase):
    def test_unverified_rapid_answer_is_failure_and_costs_budget(self):
        result = evaluate_trials(
            [query(split="heldout")],
            [trial("git_rg", 40), trial("nalcos_git_rg", 1, evidence_review={})],
        )
        self.assertEqual(result["conditions"]["nalcos_git_rg"]["successes"], 0)
        self.assertEqual(
            result["conditions"]["nalcos_git_rg"]["median_budget_charged_seconds"], 60
        )
        self.assertLess(result["median_time_reduction"], 0)
        self.assertFalse(result["passed"])

    def test_mismatched_agent_or_budget_cannot_form_a_comparison(self):
        for changed in (
            {"agent_fingerprint": "different-agent"},
            {"budget_tokens": 2000},
            {"isolation_id": "git_rg"},
        ):
            result = evaluate_trials(
                [query(split="heldout")],
                [trial("git_rg"), trial("nalcos_git_rg", **changed)],
            )
            self.assertFalse(result["available"])
            self.assertEqual(result["incomparable_pairs"], 1)

    def test_absent_comparator_is_explicitly_unavailable(self):
        result = evaluate_trials(
            [query(split="heldout")], [], candidate="commitmux_git_rg"
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["missing_pairs"], 1)
        self.assertIsNone(result["median_time_reduction"])

    def test_reviewed_sample_uses_two_week_span_not_daily_use_or_retrieval_quota(self):
        questions, trials = [], []
        for index in range(4):
            question = reviewed_query(f"q-{index}")
            question["repo"] = f"repo-{index % 3}"
            if index == 3:
                question.update(answerable=False, gold_commits=[])
                question["label_review"]["absence_reviewed"] = True
            questions.append(question)
            for condition, seconds in (("git_rg", 40), ("nalcos_git_rg", 20)):
                row = trial(condition, seconds)
                row.update(
                    query_id=question["id"],
                    isolation_id=f"{index}-{condition}",
                    order=(index + (condition == "nalcos_git_rg")) % 2,
                    started_at=f"2026-01-{1 if index % 2 == 0 else 15:02}T10:00:00+00:00",
                )
                if not question["answerable"]:
                    row["evidence_review"].update(
                        verified_commits=[], absence_reviewed=True
                    )
                trials.append(row)
        protocol = declaration(
            questions,
            frozen_before_trials=True,
            sample_rationale="Unit-test fixture; real sample adequacy needs external review.",
        )
        result = evaluate_trials(questions, trials, protocol=protocol)
        self.assertTrue(result["passed"])
        self.assertEqual(result["dogfood_span_days"], 14)
        self.assertEqual(result["dogfood_active_days"], 2)
        self.assertEqual(result["genuine_heldout_questions"], 4)
        self.assertEqual(result["median_time_reduction"], 0.5)
        for invalid_protocol in (
            None,
            protocol | {"frozen_before_trials": False},
            protocol | {"sample_rationale": ""},
            protocol | {"query_ids": protocol["query_ids"][:-1]},
        ):
            result = evaluate_trials(questions, trials, protocol=invalid_protocol)
            self.assertFalse(result["passed"])
            self.assertIn("workflow_protocol_unverified", result["product_blockers"])
        incomplete = evaluate_trials(questions, trials[:-1], protocol=protocol)
        self.assertIn("missing_or_incomparable_pairs", incomplete["product_blockers"])
        shortened = [
            row | {"started_at": row["started_at"].replace("15T10:00", "15T09:59")}
            for row in trials
        ]
        result = evaluate_trials(questions, shortened, protocol=protocol)
        self.assertIn("two_week_dogfood_unverified", result["product_blockers"])
        trials[2]["isolation_id"] = trials[0]["isolation_id"]
        with self.assertRaisesRegex(ValueError, "reused across questions"):
            evaluate_trials(questions, trials, protocol=protocol)


if __name__ == "__main__":
    unittest.main()
