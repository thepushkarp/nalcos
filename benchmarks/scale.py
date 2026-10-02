#!/usr/bin/env python3
"""Measure lexical CLI latency on a deterministic synthetic Git history.

This measures scale and source verification, not real-world retrieval quality.
The fixture and raw responses stay in a newly created local work directory.
No model, remote provider, or existing repository is used.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DOMAINS = [
    "cache",
    "network",
    "parser",
    "scheduler",
    "storage",
    "auth",
    "queue",
    "transport",
]
CHANGES = [
    "retry",
    "timeout",
    "validation",
    "cleanup",
    "invalidation",
    "fallback",
    "dispatch",
    "allocation",
]


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), file=sys.stderr, flush=True)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def prepare(work, commits):
    work.mkdir(parents=True, exist_ok=False)
    repo = work / "repository"
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main", str(repo)], check=True
    )
    started = time.perf_counter()
    with (work / "fast-import.stderr").open("wb") as errors:
        process = subprocess.Popen(
            ["git", "-C", str(repo), "fast-import", "--quiet"],
            stdin=subprocess.PIPE,
            stderr=errors,
        )
        try:
            for number in range(1, commits + 1):
                domain = DOMAINS[number % len(DOMAINS)]
                change = CHANGES[(number // len(DOMAINS)) % len(CHANGES)]
                component = number % 128
                message = f"fix({domain}): adjust {change} policy for component{component:03d} event{number:06d}\n"
                content = (
                    f"// {domain} {change} policy\n"
                    f"pub const REVISION: u64 = {number};\n"
                    f"pub const RETRY_LIMIT: u32 = {1 + number % 9};\n"
                    f"pub const TIMEOUT_MS: u32 = {50 + number % 950};\n"
                    f"// historical marker event{number:06d}\n"
                ).encode()
                header = (
                    "commit refs/heads/main\n"
                    f"committer Synthetic Benchmark <benchmark@example.invalid> {1700000000 + number} +0000\n"
                    f"data {len(message.encode())}\n{message}"
                    f"M 100644 inline src/{domain}/component{component:03d}.rs\n"
                    f"data {len(content)}\n"
                ).encode()
                process.stdin.write(header + content + b"\n")
                if number % 10000 == 0:
                    emit("fixture_progress", commits=number)
            process.stdin.write(b"done\n")
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("git fast-import failed; inspect fast-import.stderr")
        except BaseException:
            process.terminate()
            process.wait()
            raise
    observed = int(git(repo, "rev-list", "--count", "HEAD"))
    if observed != commits:
        raise RuntimeError(f"expected {commits} commits, found {observed}")
    manifest = {
        "version": 1,
        "commits": commits,
        "head": git(repo, "rev-parse", "HEAD"),
        "files_at_head": min(commits, 128),
        "shape": "linear ancestry; one small UTF-8 Rust file changed per commit; 128 rotating paths; no merges, binaries, renames, or large patches",
        "creation_seconds": time.perf_counter() - started,
    }
    (work / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (work / "config.toml").write_text("version = 1\n")
    emit("fixture_ready", **manifest)
    return manifest


def host():
    result = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    if platform.system() == "Darwin":
        for field, name in [
            ("cpu", "machdep.cpu.brand_string"),
            ("memory_bytes", "hw.memsize"),
        ]:
            probe = subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True, check=False
            )
            if probe.returncode == 0:
                value = probe.stdout.strip()
                result[field] = int(value) if field == "memory_bytes" else value
            else:
                result[field] = None
                result[field + "_probe_error"] = probe.stderr.strip()
    return result


def run(args, manifest):
    work = args.work.resolve()
    repo = work / "repository"
    binary = args.binary.resolve()
    if (
        int(git(repo, "rev-list", "--count", "HEAD")) != manifest["commits"]
        or git(repo, "rev-parse", "HEAD") != manifest["head"]
    ):
        raise RuntimeError("fixture changed since preparation")
    if (work / "data").exists() and not args.reuse_index:
        raise RuntimeError(
            "index already exists; use a fresh work directory for an initial-index measurement"
        )
    env = {
        **os.environ,
        "NALCOS_DATA_DIR": str(work / "data"),
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    base = [
        str(binary),
        "--repo",
        str(repo),
        "--config",
        str(work / "config.toml"),
        "--json",
        "--offline",
    ]
    binary_hash = digest(binary)

    def invoke(name, command, timeout):
        start = time.perf_counter()
        with (work / f"{name}.stderr").open("wb") as errors:
            result = subprocess.run(
                base + command,
                capture_output=False,
                stdout=subprocess.PIPE,
                stderr=errors,
                env=env,
                timeout=timeout,
                check=False,
            )
        elapsed = time.perf_counter() - start
        (work / f"{name}.json").write_bytes(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"{name} exited {result.returncode}; inspect local JSON and stderr"
            )
        payload = json.loads(result.stdout)
        if (
            payload["history_coverage"]["complete"] is not True
            or payload["history_coverage"]["indexed"] != manifest["commits"]
        ):
            raise RuntimeError(f"{name} did not cover the complete fixture")
        return elapsed, payload

    if args.reuse_index:
        checkpoint = json.loads((work / "index-measurement.json").read_text())
        index_seconds = checkpoint["seconds"]
    else:
        emit("index_started", commits=manifest["commits"])
        index_seconds, indexed = invoke(
            "index",
            ["--timeout", f"{args.index_timeout}s", "sync"],
            args.index_timeout + 5,
        )
        checkpoint = {
            "binary_sha256": binary_hash,
            "seconds": index_seconds,
            "concurrency": args.concurrency_note,
        }
        (work / "index-measurement.json").write_text(
            json.dumps(checkpoint, indent=2) + "\n"
        )
        emit(
            "index_complete",
            seconds=index_seconds,
            coverage=indexed["history_coverage"],
        )
    if args.index_only:
        return
    records = []
    for number in range(args.queries):
        # Alternate common terms and one historical marker to expose candidate and scope costs.
        query = (
            f"{DOMAINS[(number // 2) % len(DOMAINS)]} {CHANGES[(number // 2) % len(CHANGES)]}"
            if number % 2 == 0
            else f"event{1 + (number * 7919) % manifest['commits']:06d}"
        )
        elapsed, response = invoke(
            f"search-{number:02d}",
            [
                "--timeout",
                "60s",
                "search",
                query,
                "--mode",
                "lexical",
                "--freshness",
                "cached",
                "--limit",
                "10",
            ],
            65,
        )
        evidence = [
            item for result in response["results"] for item in result["evidence"]
        ]
        if not evidence or not all(item["verified"] for item in evidence):
            raise RuntimeError("search did not return source-verified evidence")
        records.append(
            {
                "query": query,
                "seconds": elapsed,
                "results": len(response["results"]),
                "evidence_items": len(evidence),
                "output_truncated": response["output_truncated"],
            }
        )
        emit("query_complete", number=number + 1, seconds=elapsed)
    status_seconds, status = invoke("status", ["status"], 65)
    if digest(binary) != binary_hash:
        raise RuntimeError("binary changed during measurement; results are invalid")
    timings = sorted(row["seconds"] for row in records)
    result = {
        "schema_version": 1,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "synthetic lexical scale only; not retrieval quality or semantic/hybrid qualification",
        "host": host(),
        "git_version": subprocess.check_output(["git", "--version"], text=True).strip(),
        "binary": {
            "sha256": binary_hash,
            "version": subprocess.check_output(
                [str(binary), "--version"], text=True
            ).strip(),
            "build": "cargo build --release --locked",
            "index_build_sha256": checkpoint["binary_sha256"],
            "query_build_sha256": binary_hash,
            "same_build_for_index_and_queries": checkpoint["binary_sha256"]
            == binary_hash,
        },
        "corpus": manifest,
        "measurement": {
            "process": "new CLI process per search; includes startup, live Git scope resolution, SQLite retrieval, source verification, and JSON output",
            "cache": "Filesystem/page caches were not flushed; the first search is included and later searches may benefit from warm caches; no omitted warm-up queries",
            "device": "CPU, lexical only",
            "freshness": "cached",
            "limit": 10,
            "evidence_budget_bytes": 16384,
            "concurrency": args.concurrency_note,
            "percentile_method": "nearest-rank ceil(0.95*n)",
        },
        "index_seconds": index_seconds,
        "index_concurrency": checkpoint["concurrency"],
        "history_coverage": status["history_coverage"],
        "counts": status["counts"],
        "index_storage_bytes": status["storage_bytes"],
        "application_data_bytes": sum(
            path.stat().st_size for path in (work / "data").rglob("*") if path.is_file()
        ),
        "status_seconds": status_seconds,
        "search": {
            "n": len(timings),
            "p50_seconds": statistics.median(timings),
            "p95_seconds": timings[math.ceil(0.95 * len(timings)) - 1],
            "min_seconds": timings[0],
            "max_seconds": timings[-1],
            "samples": records,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    emit(
        "benchmark_complete",
        output=str(args.output),
        p95_seconds=result["search"]["p95_seconds"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work",
        type=Path,
        required=True,
        help="New local fixture/output directory, preferably under /private/tmp",
    )
    parser.add_argument("--binary", type=Path, default=Path("target/release/nalcos"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/output/lexical-100k.json"),
    )
    parser.add_argument("--commits", type=int, default=100000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--index-timeout", type=int, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument(
        "--index-only",
        action="store_true",
        help="Pause after initial ingestion, preserving its timing",
    )
    phase.add_argument(
        "--reuse-index",
        action="store_true",
        help="Measure searches after a completed --index-only run; index and query binary hashes are recorded separately",
    )
    parser.add_argument(
        "--reuse-fixture",
        action="store_true",
        help="Use a prepared, unchanged fixture that has never been indexed",
    )
    parser.add_argument(
        "--concurrency-note",
        default="No deliberate competing benchmark; ordinary host activity was not controlled",
    )
    args = parser.parse_args()
    if args.commits < 1 or args.queries < 2 or args.index_timeout < 1:
        parser.error(
            "commits and index timeout must be positive; at least two queries are required"
        )
    if args.output.exists() and not args.prepare_only:
        parser.error("output already exists; choose a new result file")
    manifest = (
        json.loads((args.work / "fixture.json").read_text())
        if args.reuse_fixture or args.reuse_index
        else prepare(args.work, args.commits)
    )
    if not args.prepare_only:
        run(args, manifest)


if __name__ == "__main__":
    main()
