#!/usr/bin/env python3
"""Measure isolated local indexing across pinned repositories and devices."""

import argparse
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from harness import (
    GIT_ENV,
    directory_size,
    load_repositories,
    run_measured,
    validate_clone,
)
from scale import digest, emit, host
from vector_scale import runtime_conditions


def read_payload(path, measurement):
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError("expected a JSON object")
        if measurement["exit_code"] == 0:
            fields = {
                "history_coverage": ("total", "indexed"),
                "embedding_coverage": ("total", "indexed"),
            }
            if payload.get("command") == "status":
                fields["counts"] = ("commits", "documents", "vectors")
            elif payload.get("command") in {"init", "sync"}:
                fields[""] = ("commits_added", "documents_embedded")
            else:
                raise ValueError("expected init, sync, or status command output")
            for parent, keys in fields.items():
                values = payload.get(parent) if parent else payload
                for key in keys:
                    if (
                        not isinstance(values, dict)
                        or type(values.get(key)) is not int
                        or values[key] < 0
                    ):
                        raise ValueError(
                            f"missing nonnegative integer {parent + '.' if parent else ''}{key}"
                        )
        return payload
    except (ValueError, TypeError) as error:
        if measurement["exit_code"] == 0:
            raise ValueError(f"parse indexing output {path}: {error}") from error
        measurement["json_error"] = f"{path}: {error}"
        return {}


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["repo_key"], row["device"], row["tier"]), []).append(row)
    summaries = []
    for (key, device, tier), samples in groups.items():
        successful = [
            r for r in samples if r["exit_code"] == 0 and r["status_exit_code"] == 0
        ]
        medians = {}
        for field in (
            "elapsed_seconds",
            "docs_per_second",
            "peak_rss_bytes",
            "storage_bytes",
        ):
            values = [r[field] for r in successful if r[field] is not None]
            medians[field] = statistics.median(values) if values else None
        summaries.append(
            {
                "repo_key": key,
                "device": device,
                "tier": tier,
                "runs": len(samples),
                "successful_runs": len(successful),
                "medians": medians,
            }
        )
    return summaries


def run(args):
    if args.output.exists():
        raise ValueError(
            f"index matrix output {args.output} exists; choose a new directory"
        )
    repositories = load_repositories(args.repos)
    selected = args.repos_select or list(repositories)
    if len(set(selected)) != len(selected) or any(
        key not in repositories for key in selected
    ):
        raise ValueError(
            f"{args.repos}: repos-select must name unique known repository keys: {selected}"
        )
    for key in selected:
        validate_clone(args.clones / key, repositories[key])
    binary = args.binary.resolve()
    binary_hash = digest(binary)
    args.output.mkdir(parents=True)
    config = args.output.resolve() / "config.toml"
    # An ambient default_model would turn a fresh lexical sync into embedding work.
    config.write_text("version = 1\n")
    rows = []
    for key in selected:
        repo = repositories[key]
        tiers = (
            ["lexical-full", "semantic-range"]
            if repo.get("semantic_range")
            else ["semantic-full"]
        )
        for device in args.devices:
            for tier in tiers:
                for repeat in range(args.repeats):
                    work = args.output.resolve() / key / device / tier / str(repeat)
                    work.mkdir(parents=True)
                    data, runtime = work / "data", work / "runtime"
                    data.mkdir()
                    runtime.mkdir()
                    env = {
                        **os.environ,
                        **GIT_ENV,
                        "NALCOS_DATA_DIR": str(data),
                        "NALCOS_RUNTIME_CACHE": str(runtime),
                    }
                    clone = (args.clones / key).resolve()
                    scope = (
                        ["--range", repo["semantic_range"]]
                        if tier == "semantic-range"
                        else ["--ref", repo["snapshot"]]
                    )
                    command = [
                        str(binary),
                        "--repo",
                        str(clone),
                        "--config",
                        str(config),
                        "--json",
                        "-vv",
                        "--timeout",
                        f"{args.timeout:g}s",
                        "--device",
                        device,
                    ]
                    if args.offline:
                        command.append("--offline")
                    command += (
                        ["sync", *scope]
                        if tier == "lexical-full"
                        else ["init", *scope, "--model", args.model]
                    )
                    conditions = runtime_conditions()
                    measurement = run_measured(
                        command,
                        work / "index.stdout",
                        work / "index.stderr",
                        env=env,
                        timeout=args.timeout + 5,
                    )
                    payload = read_payload(work / "index.stdout", measurement)
                    storage = directory_size(data)
                    status_command = [
                        str(binary),
                        "--repo",
                        str(clone),
                        "--config",
                        str(config),
                        "--json",
                        "--offline",
                        "status",
                        *scope,
                    ]
                    status_run = run_measured(
                        status_command,
                        work / "status.stdout",
                        work / "status.stderr",
                        env=env,
                        timeout=30,
                    )
                    status = read_payload(work / "status.stdout", status_run)
                    counts = status.get("counts", {})
                    documents = counts.get(
                        "documents", payload.get("embedding_coverage", {}).get("total")
                    )
                    embedded = payload.get("documents_embedded")
                    processed = documents if tier == "lexical-full" else embedded
                    row = {
                        "repo_key": key,
                        "snapshot": repo["snapshot"],
                        "semantic_range": repo.get("semantic_range"),
                        "device": device,
                        "tier": tier,
                        "repeat": repeat,
                        "command": command,
                        **measurement,
                        "status_exit_code": status_run["exit_code"],
                        "commits": counts.get(
                            "commits",
                            payload.get("history_coverage", {}).get("indexed"),
                        ),
                        "documents": documents,
                        "vectors": counts.get("vectors"),
                        "embedded_documents": status.get("embedding_coverage", {}).get(
                            "indexed"
                        ),
                        "documents_embedded": embedded,
                        "counts": counts,
                        "indexing_runtime": payload.get("indexing_runtime"),
                        "runtime": payload.get("runtime"),
                        "selected_device": (payload.get("indexing_runtime") or {}).get(
                            "selected_device"
                        ),
                        "docs_per_second": processed / measurement["elapsed_seconds"]
                        if processed is not None
                        else None,
                        "storage_bytes": storage,
                        "stderr_path": str(
                            (work / "index.stderr").relative_to(args.output.resolve())
                        ),
                        "runtime_conditions": conditions,
                        "index_response": payload,
                        "status_response": status,
                    }
                    rows.append(row)
                    with (args.output / "runs.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    emit(
                        "index_finished",
                        repo_key=key,
                        device=device,
                        tier=tier,
                        repeat=repeat,
                        **measurement,
                    )
    if digest(binary) != binary_hash:
        raise ValueError(f"index matrix binary {binary} changed during measurement")
    summary = {
        "schema_version": 1,
        "kind": "index_matrix",
        "host": host(),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "binary_sha256": binary_hash,
        "repos_sha256": digest(args.repos),
        "model": args.model,
        "offline": args.offline,
        "summaries": summarize(rows),
        "measurement": "Index command only; status collection is untimed. RSS is the direct child's wait4 peak. Runtime caches are fresh per run; HF and filesystem caches are shared.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 3 if any(r["exit_code"] or r["status_exit_code"] for r in rows) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--repos", type=Path, default=Path("benchmarks/repos.json"))
    parser.add_argument("--clones", type=Path, required=True)
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=["cpu", "metal", "auto"],
        default=["cpu", "metal", "auto"],
    )
    parser.add_argument("--repos-select", nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--timeout", type=float, default=3600, help="Seconds per indexing run"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="coderank")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if (
        args.repeats < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or len(set(args.devices)) != len(args.devices)
    ):
        parser.error("repeats and timeout must be positive; devices must be unique")
    try:
        return run(args)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"index matrix failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
