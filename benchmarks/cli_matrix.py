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
import sys
from pathlib import Path

from evaluate import read_jsonl
from harness import GIT_ENV, run_measured


def latency_summary(rows):
    if not rows:
        return {
            "samples": 0,
            "p50_seconds": None,
            "p95_seconds": None,
            "max_seconds": None,
        }
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
    evidence_rows = []
    for result in results:
        oid = result.get("commit", {}).get("oid")
        if oid is None or oid in commits:
            reasons.append("missing_or_duplicate_commit")
        elif oid:
            commits.append(oid)
        evidence = result.get("evidence", [])
        if not evidence or any(e.get("verified") is not True for e in evidence):
            reasons.append("unverified_evidence")
        for item in evidence:
            paths = []
            for field in ("new_path", "old_path"):
                path = item.get(field)
                if isinstance(path, dict):
                    path = path.get("display")
                if isinstance(path, str) and path not in paths:
                    paths.append(path)
            evidence_rows.append(
                {"commit": oid, "paths": paths, "excerpt": item.get("excerpt", "")}
            )
    return {
        "query_id": query["id"],
        "snapshot": query["snapshot"],
        "query_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
        "commits": commits,
        "evidence": evidence_rows,
        "repo_key": query.get("repo_key"),
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
    parser.add_argument("--device", help="Global inference-device override")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "output already exists; choose a new directory to preserve evidence"
        )
    if not 1 <= args.limit <= 100 or args.timeout_seconds <= 0 or args.repeats < 1:
        parser.error("limit must be 1..100; timeout and repeats must be positive")
    queries = read_jsonl(args.manifest)
    if not queries or len(set(args.modes)) != len(args.modes):
        parser.error("manifest must be nonempty and modes must be unique")
    repositories = json.loads(args.repositories.read_text())
    if not isinstance(repositories, dict):
        parser.error(f"{args.repositories}: expected a repo_key mapping")
    ids = [q["id"] for q in queries]
    if len(ids) != len(set(ids)):
        parser.error("duplicate query ids")
    for query in queries:
        if query.get("task", "find") != "find":
            parser.error(
                "this adapter currently supports find tasks only; regression range indexing must be explicit"
            )
        key = query.get("repo_key")
        if not isinstance(key, str) or not key:
            parser.error(
                f"{args.manifest}: query {query.get('id')!r} requires a nonempty repo_key matching --repositories"
            )
        if key not in repositories:
            parser.error(
                f"{args.repositories}: missing local repository mapping for {key}"
            )
        if not isinstance(repositories[key], dict):
            parser.error(
                f"{args.repositories}: {key}: expected path, config, data_dir object"
            )
        for field in ("path", "config", "data_dir"):
            if (
                not isinstance(repositories[key].get(field), str)
                or not repositories[key][field]
            ):
                parser.error(f"{args.repositories}: {key}: missing local {field}")
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
        for repeat in range(args.repeats):
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
                ]
                if args.device:
                    command += ["--device", args.device]
                command += [
                    "search",
                    query["query"],
                    "--mode",
                    mode,
                    "--freshness",
                    "cached",
                    # Range rows were labeled within an index built for that range only;
                    # searching full ancestry would score commits the index never embedded.
                    *(
                        ["--range", query["scope_range"]]
                        if query.get("scope_range")
                        else ["--ref", query["snapshot"]]
                    ),
                    "--limit",
                    str(args.limit),
                    "--max-bytes",
                    str(args.max_bytes),
                ]
                env = {**os.environ, **GIT_ENV, "NALCOS_DATA_DIR": repo["data_dir"]}
                key = f"{index:03}-{mode}-{repeat:02}"
                measurement = run_measured(
                    command,
                    raw / f"{key}.stdout",
                    raw / f"{key}.stderr",
                    env=env,
                    timeout=args.timeout_seconds + 5,
                )
                stdout = (raw / f"{key}.stdout").read_text(errors="replace")
                try:
                    payload = json.loads(stdout)
                    if not isinstance(payload, dict):
                        raise TypeError("expected a JSON object")
                except (ValueError, TypeError) as error:
                    payload = {}
                    measurement["json_error"] = f"{raw / (key + '.stdout')}: {error}"
                row = adapt_response(query, mode, payload, measurement["exit_code"])
                check_encoder_consistency(query, row, encoder_identities)
                row.update(
                    **measurement,
                    repeat=repeat,
                    cache_state="cold" if repeat == 0 else "warm",
                    rank_limit=args.limit,
                    response_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
                )
                records[mode].append(row)
                # Separate repeat files keep every ranking set consumable by metrics,
                # which deliberately rejects duplicate query identities.
                filename = (
                    f"{mode}.rankings.jsonl"
                    if repeat == 0
                    else f"{mode}.repeat-{repeat}.rankings.jsonl"
                )
                with (args.output / filename).open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(
                    json.dumps(
                        {
                            "event": "query_finished",
                            "index": index + 1,
                            "mode": mode,
                            "repeat": repeat,
                            "cache_state": row["cache_state"],
                            "complete": row["complete"],
                            "elapsed_seconds": row["elapsed_seconds"],
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
        "repeats": args.repeats,
        "device_requested": args.device,
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
                "by_cache_state": {
                    state: latency_summary(
                        [r for r in rows if r["cache_state"] == state]
                    )
                    for state in ("cold", "warm")
                },
                "median_peak_rss_bytes": statistics.median(
                    r["peak_rss_bytes"] for r in rows
                ),
                "per_repository": {
                    repo_key: latency_summary(
                        [row for row in rows if row["repo_key"] == repo_key]
                    )
                    | {
                        "by_cache_state": {
                            state: latency_summary(
                                [
                                    row
                                    for row in rows
                                    if row["repo_key"] == repo_key
                                    and row["cache_state"] == state
                                ]
                            )
                            for state in ("cold", "warm")
                        }
                    }
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
            "Latency percentiles use linear interpolation. Cold labels the first repeat of each query/mode; warm labels subsequent CLI processes. Filesystem and model caches are not flushed.",
            "RSS is the direct CLI child peak from wait4, not aggregate descendant memory.",
            "Model generation, runtime, scope and coverage are retained in local rankings; raw content stays in the chosen output directory.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 3 if any(not r["complete"] for rows in records.values() for r in rows) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"CLI matrix failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
