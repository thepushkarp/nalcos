#!/usr/bin/env python3
"""Evaluate paired, independently reviewed local agent trials.

This adapter runs no agents or tools. Trial collection must enforce the recorded
budgets and isolate conditions. Review attestations require independent audit.
"""

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from evaluate import (
    POLICY,
    cohort_rows,
    genuine_holdout,
    read_jsonl,
    validate_manifest,
    workflow_protocol_passed,
)


def evaluate_trials(
    manifest, trials, candidate="nalcos_git_rg", split="heldout", protocol=None
):
    if candidate == "git_rg":
        raise ValueError("candidate must differ from git_rg")
    validate_manifest(manifest)
    planned, protocol_reviewed = cohort_rows(manifest, protocol)
    if protocol and any(q["split"] != split for q in planned):
        raise ValueError(
            "workflow protocol query_ids must belong to the selected split"
        )
    selected = [q for q in planned if q["split"] == split]
    queries = {q["id"]: q for q in manifest}
    indexed = {}
    isolation_queries = {}
    for row in trials:
        identifier = row.get("query_id")
        if identifier not in queries:
            raise ValueError(f"unknown trial query_id: {identifier!r}")
        query = queries[identifier]
        if (
            row.get("snapshot") != query["snapshot"]
            or row.get("query_sha256")
            != hashlib.sha256(query["query"].encode()).hexdigest()
            or (
                query["task"] == "regression"
                and any(row.get(k) != query[k] for k in ("good", "bad"))
            )
        ):
            raise ValueError(
                f"{identifier}: trial query/snapshot/range identity differs"
            )
        condition = row.get("condition")
        if not isinstance(condition, str) or not condition:
            raise ValueError(f"{identifier}: condition must be a nonempty string")
        key = (identifier, condition)
        if key in indexed:
            raise ValueError(f"duplicate trial: {identifier}/{condition}")
        for field in ("agent_fingerprint", "isolation_id"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{identifier}: {field} is missing")
        if isolation_queries.get(row["isolation_id"], identifier) != identifier:
            raise ValueError(f"{identifier}: session was reused across questions")
        isolation_queries[row["isolation_id"]] = identifier
        for field in ("budget_seconds", "budget_tokens", "elapsed_seconds"):
            value = row.get(field)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{identifier}: {field} must be finite and positive")
        tokens = row.get("tokens_used")
        if type(tokens) is not int or tokens < 0:
            raise ValueError(f"{identifier}: tokens_used must be a nonnegative integer")
        if row.get("outcome") not in {"completed", "timeout", "failed"}:
            raise ValueError(f"{identifier}: invalid outcome")
        if type(row.get("order")) is not int or row["order"] not in (0, 1):
            raise ValueError(f"{identifier}: order must be 0 or 1")
        started = datetime.fromisoformat(row.get("started_at", ""))
        if started.tzinfo is None:
            raise ValueError(f"{identifier}: started_at must include a timezone")
        row = dict(
            row,
            observed_at=started.astimezone(timezone.utc),
            observed_date=started.astimezone(timezone.utc).date().isoformat(),
        )
        indexed[key] = row

    missing = []
    incompatible = []
    pairs = []
    for query in selected:
        baseline = indexed.get((query["id"], "git_rg"))
        treatment = indexed.get((query["id"], candidate))
        if baseline is None or treatment is None:
            missing.append(query["id"])
            continue
        comparable = (
            all(
                baseline[k] == treatment[k]
                for k in ("agent_fingerprint", "budget_seconds", "budget_tokens")
            )
            and baseline["isolation_id"] != treatment["isolation_id"]
            and baseline["order"] != treatment["order"]
        )
        if not comparable:
            incompatible.append(query["id"])
            continue
        if query["answerable"] is not None:
            pairs.append((query, baseline, treatment))

    def success(query, trial):
        review = trial.get("evidence_review", {})
        if not (
            trial["outcome"] == "completed"
            and trial["elapsed_seconds"] <= trial["budget_seconds"]
            and trial["tokens_used"] <= trial["budget_tokens"]
            and review.get("passed") is True
            and review.get("blind_to_condition") is True
            and review.get("reviewer_id")
            and review.get("references")
        ):
            return False
        verified = review.get("verified_commits", [])
        if not isinstance(verified, list):
            return False
        if query["answerable"]:
            return bool(set(verified).intersection(query["gold_commits"]))
        return not verified and review.get("absence_reviewed") is True

    conditions = {}
    for index, condition in ((1, "git_rg"), (2, candidate)):
        outcomes = [success(pair[0], pair[index]) for pair in pairs]
        # A quick failure must not appear faster than a successful investigation.
        # Unsuccessful trials consume their allotted budget in the comparison.
        charged = []
        for pair, succeeded in zip(pairs, outcomes, strict=True):
            trial = pair[index]
            charged.append(
                trial["elapsed_seconds"]
                if succeeded
                else max(trial["budget_seconds"], trial["elapsed_seconds"])
            )
        conditions[condition] = {
            "successes": sum(outcomes),
            "failures": len(outcomes) - sum(outcomes),
            "success_rate": sum(outcomes) / len(outcomes) if outcomes else None,
            "median_budget_charged_seconds": statistics.median(charged)
            if charged
            else None,
            "median_tokens_used": statistics.median(
                pair[index]["tokens_used"] for pair in pairs
            )
            if pairs
            else None,
        }
    baseline = conditions["git_rg"]
    treatment = conditions[candidate]
    delta = treatment["success_rate"] - baseline["success_rate"] if pairs else None
    reduction = (
        1
        - treatment["median_budget_charged_seconds"]
        / baseline["median_budget_charged_seconds"]
        if pairs
        else None
    )
    days = {trial["observed_date"] for _, *pair in pairs for trial in pair}
    observed = [trial["observed_at"] for _, *pair in pairs for trial in pair]
    first = min(observed) if observed else None
    last = max(observed) if observed else None
    span = (last - first).total_seconds() / 86400 if observed else 0
    independent = bool(pairs) and all(genuine_holdout(q) for q, _, _ in pairs)
    order_counterbalanced = {treatment["order"] for _, _, treatment in pairs} == {0, 1}
    blockers = []
    if not protocol_reviewed or not workflow_protocol_passed(protocol):
        blockers.append("workflow_protocol_unverified")
    if missing or incompatible:
        blockers.append("missing_or_incomparable_pairs")
    if any(q["answerable"] is None for q in selected):
        blockers.append("unjudged_queries_present")
    if not independent:
        blockers.append("independent_heldout_unverified")
    if span < POLICY["minimum_dogfood_span_days"]:
        blockers.append("two_week_dogfood_unverified")
    if not order_counterbalanced:
        blockers.append("condition_order_not_counterbalanced")
    if delta is None or delta < 0:
        blockers.append("task_success_not_confirmed")
    if reduction is None or reduction < POLICY["minimum_agent_time_reduction"]:
        blockers.append("agent_time_reduction_not_confirmed")
    return {
        "schema_version": 2,
        "available": bool(pairs),
        "passed": not blockers,
        "baseline": "git_rg",
        "candidate": candidate,
        "split": split,
        "protocol": protocol,
        "planned_questions": len(selected),
        "paired_questions": len(pairs),
        "genuine_heldout_questions": sum(genuine_holdout(q) for q, _, _ in pairs),
        "repositories": len({q["repo"] for q, _, _ in pairs}),
        "unjudged_questions": sum(q["answerable"] is None for q in selected),
        "missing_pairs": len(missing),
        "incomparable_pairs": len(incompatible),
        "same_agent": bool(pairs) and not incompatible,
        "independent_heldout": independent,
        "dogfood_span_days": span,
        "dogfood_active_days": len(days),
        "dogfood_started_at": first.isoformat() if first else None,
        "dogfood_ended_at": last.isoformat() if last else None,
        "condition_order_counterbalanced": order_counterbalanced,
        "success_rate_delta": delta,
        "median_time_reduction": reduction,
        "conditions": conditions,
        "product_blockers": blockers,
        "attestation_notice": "Budgets, isolation, timestamps and blind evidence review require external audit. No agent trials are run by this adapter. Failure time is charged at least the full task budget.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--candidate", default="nalcos_git_rg")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument(
        "--split", choices=["exploratory", "development", "heldout"], default="heldout"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.candidate == "git_rg":
            raise ValueError("candidate must differ from git_rg")
        result = evaluate_trials(
            read_jsonl(args.manifest),
            read_jsonl(args.trials),
            args.candidate,
            args.split,
            json.loads(args.protocol.read_text()) if args.protocol else None,
        )
        result["references"] = [
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.manifest, args.trials, args.protocol)
            if path is not None
        ]
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output)
        else:
            print(output, end="")
        return 0 if result["passed"] else 3
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"workflow evaluation failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
