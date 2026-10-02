"""Do not turn degraded retrieval into a successful evaluation response."""

import unittest

from cli_matrix import adapt_response, check_encoder_consistency


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


if __name__ == "__main__":
    unittest.main()
