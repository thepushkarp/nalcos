#!/usr/bin/env python3
"""Run frozen queries against existing local indexes without indexing or networking.

Raw queries, Git evidence and private paths belong in a local output directory,
not a published report. Aggregate reports contain hashes and counts only.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evaluate import read_jsonl


def latency_summary(rows):
    samples = sorted(row["elapsed_seconds"] for row in rows)
    position = (len(samples) - 1) * 0.95
    low, high = math.floor(position), math.ceil(position)
    p95 = samples[low] + (samples[high] - samples[low]) * (position - low)
    return {
        "samples": len(samples),
        "p50_seconds": statistics.median(samples),
        "p95_seconds": p95,
        "max_seconds": max(samples),
    }


def adapt_response(query, requested_mode, payload, returncode):
    reasons = []
    if returncode != 0:
        reasons.append("command_failed")
    if payload.get("mode_used") != requested_mode:
        reasons.append("mode_fallback")
    history = payload.get("history_coverage", {})
    embedding = payload.get("embedding_coverage", {})
    if history.get("complete") is not True:
        reasons.append("history_incomplete")
    if requested_mode != "lexical" and embedding.get("complete") is not True:
        reasons.append("embedding_incomplete")
    scope = payload.get("scope", {})
    if query["snapshot"] not in {r.get("oid") for r in scope.get("roots", [])}:
        reasons.append("snapshot_mismatch")
    if scope.get("shallow") is not False:
        reasons.append("shallow_or_unknown_history")
    if type(payload.get("output_truncated")) is not bool:
        reasons.append("output_truncation_unknown")
    results = payload.get("results", [])
    commits = []
    for result in results:
        oid = result.get("commit", {}).get("oid")
        if oid is None or oid in commits:
            reasons.append("missing_or_duplicate_commit")
        elif oid:
            commits.append(oid)
        evidence = result.get("evidence", [])
        if not evidence or any(e.get("verified") is not True for e in evidence):
            reasons.append("unverified_evidence")
    return {
        "query_id": query["id"],
        "snapshot": query["snapshot"],
        "query_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
        "commits": commits,
        "complete": not reasons,
        # Retrieval candidates do not constitute an evidence-backed absence claim.
        "abstained": False,
        "incomplete_reasons": sorted(set(reasons)),
        # Presentation budgets can clip messages/excerpts without losing ranked
        # commit identities. Each returned result still needs verified evidence.
        "output_truncated": payload.get("output_truncated"),
        "mode_requested": requested_mode,
        "mode_used": payload.get("mode_used"),
        "history_coverage": history,
        "embedding_coverage": embedding,
        "scope_signature": scope.get("signature"),
        "runtime": payload.get("runtime"),
        "active_generation": payload.get("active_generation"),
        "warnings": payload.get("warnings", []),
        **(
            {"good": query["good"], "bad": query["bad"]}
            if query.get("task") == "regression"
            else {}
        ),
    }


def check_encoder_consistency(query, row, seen):
    if row["mode_used"] not in {"semantic", "hybrid"}:
        return
    generation = row.get("active_generation")
    runtime = row.get("runtime")
    if not (
        isinstance(generation, dict)
        and all(
            generation.get(k) is not None
            for k in ("id", "model", "revision", "fingerprint")
        )
        and isinstance(runtime, dict)
        and all(
            runtime.get(k) is not None
            for k in ("runtime_version", "provider", "selected_device")
        )
    ):
        row["incomplete_reasons"].append("encoder_identity_missing")
        row["complete"] = False
        return
    identity = {
        "generation": generation,
        "runtime": {
            k: runtime[k] for k in ("runtime_version", "provider", "selected_device")
        },
    }
    previous = seen.setdefault(query["repo_key"], identity)
    if previous != identity:
        row["incomplete_reasons"].append("encoder_changed_during_matrix")
        row["complete"] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--repositories",
        type=Path,
        required=True,
        help="Local JSON mapping repo_key to path, config, data_dir",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["lexical", "semantic", "hybrid"],
        default=["lexical", "semantic", "hybrid"],
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--max-bytes", type=int, default=1048576)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "output already exists; choose a new directory to preserve evidence"
        )
    if not 1 <= args.limit <= 100 or args.timeout_seconds <= 0:
        parser.error("limit must be 1..100 and timeout must be positive")
    queries = read_jsonl(args.manifest)
    if not queries or len(set(args.modes)) != len(args.modes):
        parser.error("manifest must be nonempty and modes must be unique")
    repositories = json.loads(args.repositories.read_text())
    ids = [q["id"] for q in queries]
    if len(ids) != len(set(ids)):
        parser.error("duplicate query ids")
    for query in queries:
        if query.get("task", "find") != "find":
            parser.error(
                "this adapter currently supports find tasks only; regression range indexing must be explicit"
            )
        if query["repo_key"] not in repositories:
            parser.error("missing local repository mapping for " + query["repo_key"])
    args.output.mkdir(parents=True)
    raw = args.output / "raw"
    raw.mkdir()
    records = {mode: [] for mode in args.modes}
    encoder_identities = {}
    # Modes rotate across queries to reduce systematic warm-cache ordering bias.
    for index, query in enumerate(queries):
        repo = repositories[query["repo_key"]]
        modes = (
            args.modes[index % len(args.modes) :]
            + args.modes[: index % len(args.modes)]
        )
        for mode in modes:
            command = [
                str(args.binary.resolve()),
                "--repo",
                repo["path"],
                "--config",
                repo["config"],
                "--json",
                "--offline",
                "--timeout",
                f"{args.timeout_seconds}s",
                "search",
                query["query"],
                "--mode",
                mode,
                "--freshness",
                "cached",
                "--ref",
                query["snapshot"],
                "--limit",
                str(args.limit),
                "--max-bytes",
                str(args.max_bytes),
            ]
            env = {
                **os.environ,
                "NALCOS_DATA_DIR": repo["data_dir"],
                "GIT_NO_LAZY_FETCH": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
            }
            start = time.perf_counter()
            try:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_seconds + 5,
                    env=env,
                    check=False,
                )
                stdout, stderr, returncode = (
                    process.stdout,
                    process.stderr,
                    process.returncode,
                )
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout or b""
                stderr = error.stderr or b""
                stdout = (
                    stdout.decode(errors="replace")
                    if isinstance(stdout, bytes)
                    else stdout
                )
                stderr = (
                    stderr.decode(errors="replace")
                    if isinstance(stderr, bytes)
                    else stderr
                )
                returncode = 124
            elapsed = time.perf_counter() - start
            key = f"{index:03}-{mode}"
            (raw / f"{key}.stdout").write_text(stdout)
            (raw / f"{key}.stderr").write_text(stderr)
            try:
                payload = json.loads(stdout)
                if not isinstance(payload, dict):
                    payload = {}
            except json.JSONDecodeError:
                payload = {}
            row = adapt_response(query, mode, payload, returncode)
            check_encoder_consistency(query, row, encoder_identities)
            row.update(
                elapsed_seconds=elapsed,
                exit_code=returncode,
                rank_limit=args.limit,
                response_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
            )
            records[mode].append(row)
            with (args.output / f"{mode}.rankings.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {
                        "event": "query_finished",
                        "index": index + 1,
                        "mode": mode,
                        "complete": row["complete"],
                        "elapsed_seconds": elapsed,
                    }
                ),
                flush=True,
                file=sys.stderr,
            )
    summary = {
        "schema_version": 2,
        "kind": "cli_search_matrix",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": {
            key: hashlib.sha256(
                Path(repositories[key]["config"]).read_bytes()
            ).hexdigest()
            for key in sorted({q["repo_key"] for q in queries})
        },
        "questions": len(queries),
        "repositories": len({q["repo_key"] for q in queries}),
        "rank_limit": args.limit,
        "indexing_performed": False,
        "network_allowed": False,
        "modes": {
            mode: {
                "responses": len(rows),
                "complete_responses": sum(r["complete"] for r in rows),
                "truncated_output_responses": sum(
                    r["output_truncated"] is True for r in rows
                ),
                "median_wall_seconds": statistics.median(
                    r["elapsed_seconds"] for r in rows
                ),
                "max_wall_seconds": max(r["elapsed_seconds"] for r in rows),
                "incomplete_reasons": sorted(
                    {reason for r in rows for reason in r["incomplete_reasons"]}
                ),
                "latency": latency_summary(rows),
                "per_repository": {
                    repo_key: latency_summary(
                        [
                            row
                            for row in rows
                            if row["query_id"]
                            in {q["id"] for q in queries if q["repo_key"] == repo_key}
                        ]
                    )
                    for repo_key in sorted({q["repo_key"] for q in queries})
                },
            }
            for mode, rows in records.items()
        },
        "caveats": [
            "Full CLI wall time includes process/model startup and exact Git evidence checks.",
            "No-match retrieval is not automatically a correct negative answer.",
            "Output clipping is reported separately from retrieval completeness; every returned commit still requires verified evidence.",
            "MRR computed from these outputs is truncated at the recorded rank limit.",
            "Latency percentiles use linear interpolation over one observation per query, not repeated-run tail estimates.",
            "Model generation, runtime, scope and coverage are retained in local rankings; raw content stays in the chosen output directory.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 3 if any(not r["complete"] for rows in records.values() for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
