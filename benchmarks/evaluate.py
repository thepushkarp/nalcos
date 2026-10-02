#!/usr/bin/env python3
"""Score frozen query manifests and apply the documented default-selection policy.

This tool evaluates evidence; it cannot certify that a human collected a genuine
holdout. Provenance and license attestations still require review.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

OID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
POLICY = {
    "max_weight_bytes": 500 * 1024 * 1024,
    "recall10_tolerance": 0.05,
    "mrr_tolerance": 0.03,
    "timing_tie_fraction": 0.10,
    "minimum_cpu_timing_runs": 3,
    "minimum_heldout_questions": 30,
    "minimum_heldout_repositories": 3,
    "minimum_agent_time_reduction": 0.30,
    "minimum_dogfood_span_days": 14,
}


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    return rows


def validate_manifest(rows):
    seen = set()
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"missing or duplicate query id: {identifier!r}")
        seen.add(identifier)
        for field in ("repo", "query"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{identifier}: {field} must be a nonempty string")
        if not OID.fullmatch(row.get("snapshot", "")):
            raise ValueError(f"{identifier}: snapshot must be a full Git object id")
        if row.get("split") not in {"exploratory", "development", "heldout"}:
            raise ValueError(f"{identifier}: invalid split")
        if row.get("origin") not in {"prospective", "retrospective"}:
            raise ValueError(f"{identifier}: invalid origin")
        if type(row.get("frozen_before_tuning")) is not bool:
            raise ValueError(f"{identifier}: frozen_before_tuning must be boolean")
        unjudged = row.get("label_status") == "unjudged"
        if unjudged:
            if row.get("answerable") is not None:
                raise ValueError(f"{identifier}: unjudged answerable must be null")
        elif type(row.get("answerable")) is not bool:
            raise ValueError(f"{identifier}: answerable must be boolean")
        gold = row.get("gold_commits")
        if not isinstance(gold, list) or any(
            not isinstance(oid, str) or not OID.fullmatch(oid) for oid in gold
        ):
            raise ValueError(f"{identifier}: gold_commits must contain full Git ids")
        if len(set(gold)) != len(gold) or bool(gold) != (row["answerable"] is True):
            raise ValueError(f"{identifier}: gold commits disagree with answerable")
        if row.get("task") not in {"find", "regression"}:
            raise ValueError(f"{identifier}: task must be find or regression")
        if row["task"] == "regression" and any(
            not OID.fullmatch(row.get(field, "")) for field in ("good", "bad")
        ):
            raise ValueError(f"{identifier}: regression requires frozen good/bad ids")
    return rows


def metrics(manifest, rankings, split=None):
    """Macro recall counts all labeled relevant commits; hit rate counts any gold.

    Negatives have no reciprocal rank. Missing/partial responses are reported
    separately and cannot be treated as successful abstentions.
    """
    validate_manifest(manifest)
    queries = [q for q in manifest if split is None or q["split"] == split]
    by_id = {}
    known = {q["id"]: q for q in manifest}
    for row in rankings:
        identifier = row.get("query_id")
        if identifier not in known or identifier in by_id:
            raise ValueError(f"unknown or duplicate ranking query_id: {identifier!r}")
        expected = known[identifier]
        if (
            row.get("snapshot") != expected["snapshot"]
            or row.get("query_sha256")
            != hashlib.sha256(expected["query"].encode()).hexdigest()
        ):
            raise ValueError(
                f"{identifier}: ranking snapshot/query digest differs from manifest"
            )
        if expected["task"] == "regression" and any(
            row.get(field) != expected[field] for field in ("good", "bad")
        ):
            raise ValueError(
                f"{identifier}: ranking regression range differs from manifest"
            )
        commits = row.get("commits")
        if not isinstance(commits, list) or any(
            not isinstance(oid, str) or not OID.fullmatch(oid) for oid in commits
        ):
            raise ValueError(f"{identifier}: commits must contain full Git ids")
        if len(set(commits)) != len(commits):
            raise ValueError(f"{identifier}: duplicate ranked commits")
        if type(row.get("complete")) is not bool:
            raise ValueError(f"{identifier}: complete must be boolean")
        if type(row.get("abstained")) is not bool:
            raise ValueError(f"{identifier}: abstained must be boolean")
        if row["abstained"] and commits:
            raise ValueError(f"{identifier}: abstained response contains candidates")
        by_id[identifier] = row
    positives = [q for q in queries if q["answerable"] is True]
    negatives = [q for q in queries if q["answerable"] is False]
    unjudged = [q for q in queries if q["answerable"] is None]
    recall = {k: 0.0 for k in (1, 5, 10)}
    hit = {k: 0 for k in recall}
    rr = 0.0
    details = []
    for query in positives:
        ranked = by_id.get(query["id"], {}).get("commits", [])
        gold = set(query["gold_commits"])
        rank = next((i for i, oid in enumerate(ranked, 1) if oid in gold), None)
        rr += 1 / rank if rank else 0
        for k in recall:
            found = len(gold.intersection(ranked[:k]))
            recall[k] += found / len(gold)
            hit[k] += int(found > 0)
        details.append({"query_id": query["id"], "first_relevant_rank": rank})
    count = len(positives)
    complete = sum(by_id.get(q["id"], {}).get("complete") is True for q in queries)
    return {
        "split": split,
        "questions": len(queries),
        "answerable_questions": count,
        "negative_questions": len(negatives),
        "unjudged_questions": len(unjudged),
        "labels_complete": not unjudged,
        "repositories": len({q["repo"] for q in queries}),
        "responses": sum(q["id"] in by_id for q in queries),
        "complete_responses": complete,
        "coverage_complete": bool(queries) and complete == len(queries),
        "recall": {str(k): recall[k] / count if count else None for k in recall},
        "hit_rate": {str(k): hit[k] / count if count else None for k in hit},
        "mrr": rr / count if count else None,
        "negative_correct_abstentions": sum(
            by_id.get(q["id"], {}).get("complete") is True
            and by_id[q["id"]]["abstained"]
            for q in negatives
        ),
        "details": details,
    }


def evidence_passed(evidence):
    return (
        isinstance(evidence, dict)
        and evidence.get("passed") is True
        and isinstance(evidence.get("references"), list)
        and bool(evidence["references"])
        and all(isinstance(ref, str) and ref.strip() for ref in evidence["references"])
    )


def genuine_holdout(row):
    """Historical questions can be independent; answer-derived prompts cannot."""
    provenance = row.get("provenance", {})
    label = row.get("label_review", {})
    return (
        row["split"] == "heldout"
        and row["frozen_before_tuning"] is True
        and row["answerable"] is not None
        and evidence_passed(provenance)
        and provenance.get("source_kind")
        in {"prospective_user_question", "historical_user_question"}
        and all(
            provenance.get(field) is True
            for field in (
                "answer_independent",
                "locked_before_answer_inspection",
                "locked_before_ranking",
                "isolation_reviewed",
            )
        )
        and evidence_passed(label)
        and (row["answerable"] or label.get("absence_reviewed") is True)
    )


def cohort_digest(rows):
    """Bind membership and query identities without freezing later gold labels."""
    identities = [
        {
            key: row[key]
            for key in (
                "id",
                "repo",
                "snapshot",
                "query",
                "split",
                "task",
                "good",
                "bad",
            )
            if key in row
        }
        for row in sorted(rows, key=lambda row: row["id"])
    ]
    return hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cohort_rows(manifest, declaration):
    """A reviewed lock, not a filtered score report, defines qualification scope."""
    if not declaration:
        return list(manifest), False
    if not isinstance(declaration, dict):
        raise TypeError("cohort declaration must be an object")
    identifiers = declaration.get("query_ids")
    if (
        not isinstance(identifiers, list)
        or not identifiers
        or any(not isinstance(identifier, str) for identifier in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError("cohort query_ids must be a nonempty list of unique ids")
    known = {row["id"]: row for row in manifest}
    if any(identifier not in known for identifier in identifiers):
        raise ValueError("cohort contains a query id absent from the manifest")
    rows = [known[identifier] for identifier in identifiers]
    return rows, (
        evidence_passed(declaration)
        and declaration.get("query_identities_sha256") == cohort_digest(rows)
    )


def workflow_protocol_passed(protocol):
    return (
        evidence_passed(protocol)
        and protocol.get("frozen_before_trials") is True
        and isinstance(protocol.get("sample_rationale"), str)
        and bool(protocol["sample_rationale"].strip())
    )


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def qualify(manifest, candidates, base, policy=None, workflow=None, cohort=None):
    """Choose on development evidence, then confirm that fixed choice on holdout.

    A failed holdout never promotes a runner-up; doing so would tune on holdout.
    Timing cohorts must describe identical corpus/runtime/backend/threads/chunks.
    Attestations are explicit inputs, not facts inferred from a model's name.
    """
    validate_manifest(manifest)
    scoped, cohort_reviewed = cohort_rows(manifest, cohort)
    scoped_ids = {q["id"] for q in scoped}
    cohort_locked = (
        cohort_reviewed
        and cohort.get("frozen_before_tuning") is True
        and cohort.get("frozen_before_ranking") is True
    )
    policy = POLICY | (policy or {})
    for name, value in policy.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid policy value: {name}")
    reports = []
    identifiers = set()
    for candidate in candidates:
        identifier = candidate["id"]
        if identifier in identifiers:
            raise ValueError(f"duplicate candidate id: {identifier}")
        identifiers.add(identifier)
        reasons = []
        weight = candidate.get("weight_bytes")
        if not isinstance(weight, int) or not 0 < weight <= policy["max_weight_bytes"]:
            reasons.append("weight_limit_unverified_or_exceeded")
        for gate in ("license", "correctness", "cpu_portability"):
            if not evidence_passed(candidate.get(gate)):
                reasons.append(f"{gate}_unverified")
        platforms = candidate.get("cpu_portability", {}).get("platforms", [])
        if not any(p.startswith("linux-") for p in platforms):
            reasons.append("linux_cpu_unverified")
        if not any(p.startswith("darwin-") for p in platforms):
            reasons.append("macos_cpu_unverified")
        path = candidate.get("rankings")
        rankings = read_jsonl(base / path) if path else []
        # Validate every supplied identity before selecting the predeclared cohort.
        metrics(manifest, rankings)
        rankings = [row for row in rankings if row["query_id"] in scoped_ids]
        development = metrics(scoped, rankings, "development")
        if (
            not development["coverage_complete"]
            or not development["answerable_questions"]
            or not development["labels_complete"]
        ):
            reasons.append("development_coverage_missing")
        samples = candidate.get("cpu_index_seconds", [])
        valid_samples = (
            isinstance(samples, list)
            and len(samples) >= policy["minimum_cpu_timing_runs"]
            and all(
                type(n) in (int, float) and math.isfinite(n) and n > 0 for n in samples
            )
        )
        if not valid_samples or not candidate.get("timing_cohort"):
            reasons.append("matched_cpu_timings_missing")
        reports.append(
            {
                "id": identifier,
                "weight_bytes": weight,
                "blockers": reasons,
                "development": development,
                "heldout": metrics(scoped, rankings, "heldout"),
                "cpu_median_seconds": statistics.median(samples)
                if valid_samples
                else None,
                "timing_cohort": candidate.get("timing_cohort"),
            }
        )
    # Quality frontiers use all complete development measurements, including
    # larger/slower candidates. Filtering by speed or license first biases quality.
    measured = [
        r
        for r in reports
        if r["development"]["coverage_complete"]
        and r["development"]["answerable_questions"]
        and r["development"]["labels_complete"]
    ]
    best_recall = max(
        (r["development"]["recall"]["10"] for r in measured), default=None
    )
    best_mrr = max((r["development"]["mrr"] for r in measured), default=None)
    for report in reports:
        dev = report["development"]
        if dev["answerable_questions"] and best_recall is not None:
            if dev["recall"]["10"] + policy["recall10_tolerance"] + 1e-12 < best_recall:
                report["blockers"].append("development_recall10_below_tolerance")
            if dev["mrr"] + policy["mrr_tolerance"] + 1e-12 < best_mrr:
                report["blockers"].append("development_mrr_below_tolerance")
    eligible = [r for r in reports if not r["blockers"]]
    cohorts = {r["timing_cohort"] for r in eligible}
    blockers = []
    if not cohort_locked:
        blockers.append("evaluation_cohort_lock_unverified")
    if any(q["split"] == "exploratory" for q in scoped):
        blockers.append("evaluation_cohort_contains_exploratory_questions")
    if any(q["answerable"] is None for q in scoped):
        blockers.append("unjudged_queries_present")
    selected = None
    reference = None
    if len(cohorts) > 1:
        blockers.append("cpu_timing_cohorts_are_not_comparable")
    elif eligible:
        fastest = min(r["cpu_median_seconds"] for r in eligible)
        tied = [
            r
            for r in eligible
            if r["cpu_median_seconds"] <= fastest * (1 + policy["timing_tie_fraction"])
        ]
        selected = min(
            tied, key=lambda r: (r["weight_bytes"], r["cpu_median_seconds"], r["id"])
        )
        reference = max(
            measured,
            key=lambda r: (
                r["development"]["recall"]["10"],
                r["development"]["mrr"],
                r["id"],
            ),
        )
    else:
        blockers.append("no_development_candidate_passes_prerequisites")
    heldout = [q for q in scoped if genuine_holdout(q)]
    if len(heldout) < policy["minimum_heldout_questions"]:
        blockers.append("insufficient_genuine_heldout_questions")
    if len({q["repo"] for q in heldout}) < policy["minimum_heldout_repositories"]:
        blockers.append("insufficient_genuine_heldout_repositories")
    if any(q["split"] == "heldout" and not genuine_holdout(q) for q in scoped):
        blockers.append("heldout_provenance_or_labels_unverified")
    if selected:
        for row in (selected, reference):
            if (
                not row["heldout"]["coverage_complete"]
                or not row["heldout"]["answerable_questions"]
            ):
                blockers.append("heldout_confirmation_missing")
        if "heldout_confirmation_missing" not in blockers:
            if (
                selected["heldout"]["recall"]["10"]
                + policy["recall10_tolerance"]
                + 1e-12
                < reference["heldout"]["recall"]["10"]
            ):
                blockers.append("heldout_recall10_confirmation_failed")
            if (
                selected["heldout"]["mrr"] + policy["mrr_tolerance"] + 1e-12
                < reference["heldout"]["mrr"]
            ):
                blockers.append("heldout_mrr_confirmation_failed")
    model_qualified = selected is not None and not blockers
    product_blockers = []
    workflow = workflow or {}
    protocol = workflow.get("protocol", {})
    planned, protocol_reviewed = cohort_rows(manifest, protocol)
    if not (
        protocol_reviewed
        and workflow_protocol_passed(protocol)
        and all(genuine_holdout(q) for q in planned)
        and workflow.get("paired_questions") == len(planned)
    ):
        product_blockers.append("workflow_protocol_or_sample_unverified")
    span = workflow.get("dogfood_span_days")
    if not finite_number(span) or span < policy["minimum_dogfood_span_days"]:
        product_blockers.append("two_week_dogfood_unverified")
    delta = workflow.get("success_rate_delta")
    reduction = workflow.get("median_time_reduction")
    if not (
        evidence_passed(workflow)
        and workflow.get("same_agent") is True
        and workflow.get("independent_heldout") is True
        and finite_number(delta)
        and delta >= 0
        and finite_number(reduction)
        and reduction >= policy["minimum_agent_time_reduction"]
    ):
        product_blockers.append("agent_git_rg_task_success_gate_unverified")
    product_qualified = not product_blockers
    return {
        "schema_version": 2,
        "model_qualified": model_qualified,
        "product_qualified": product_qualified,
        "release_qualified": model_qualified and product_qualified,
        "development_choice": selected["id"] if selected else None,
        "frozen_quality_reference": reference["id"] if reference else None,
        "model_blockers": sorted(set(blockers)),
        "product_blockers": sorted(set(product_blockers)),
        "release_blockers": sorted(set(blockers + product_blockers)),
        "policy": policy,
        "evaluation_cohort": {
            "lock_reviewed": cohort_locked,
            "questions": len(scoped),
            "excluded_questions": len(manifest) - len(scoped),
            "query_identities_sha256": cohort_digest(scoped),
        },
        "genuine_heldout_questions": len(heldout),
        "genuine_heldout_positive_questions": sum(q["answerable"] for q in heldout),
        "genuine_heldout_negative_questions": sum(not q["answerable"] for q in heldout),
        "genuine_heldout_repositories": len({q["repo"] for q in heldout}),
        "candidates": reports,
        "attestation_notice": "License, provenance, correctness and workflow references require independent review; this tool checks supplied evidence structure and metrics.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("metrics")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--rankings", type=Path, required=True)
    score.add_argument("--split", choices=["exploratory", "development", "heldout"])
    select = sub.add_parser("qualify")
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--candidates", type=Path, required=True)
    select.add_argument("--policy", type=Path)
    for item in (score, select):
        item.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        manifest = read_jsonl(args.manifest)
        if args.command == "metrics":
            result = metrics(manifest, read_jsonl(args.rankings), args.split)
        else:
            data = json.loads(args.candidates.read_text())
            policy = json.loads(args.policy.read_text()) if args.policy else None
            result = qualify(
                manifest,
                data["candidates"],
                args.candidates.parent,
                policy,
                data.get("workflow_evaluation"),
                data.get("evaluation_cohort"),
            )
        result["manifest_sha256"] = hashlib.sha256(
            args.manifest.read_bytes()
        ).hexdigest()
        output = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output)
        else:
            print(output, end="")
        return 3 if args.command == "qualify" and not result["release_qualified"] else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"evaluation failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
