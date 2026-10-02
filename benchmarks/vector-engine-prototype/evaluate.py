"""Compare mmap ANN commit rankings with an exact local F32-vector reference."""

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("prefix")
p.add_argument("--queries", type=pathlib.Path, required=True)
p.add_argument("--expansions", default="128,512,2048,8192,32768")
a = p.parse_args()
here = pathlib.Path(__file__).parent
prefix = pathlib.Path(a.prefix)
binary = here / "target/release/nalcos-usearch-prototype"
query = a.queries
out = prefix.parent / (prefix.name + "-results")
out.mkdir(exist_ok=True)


def call(mode, ef, scope):
    start = time.perf_counter()
    run = subprocess.run(
        [str(binary), mode, str(prefix), str(query), str(ef), str(scope)],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    data = json.loads(run.stdout)
    data["process_wall_seconds"] = time.perf_counter() - start
    (out / f"{mode}-{scope}-{ef}.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def percentile(xs, p):
    return sorted(xs)[min(len(xs) - 1, math.ceil(len(xs) * p) - 1)]


report = {
    "purpose": "ANN commit-ranking recall against exact; engine-only timings exclude query encoding/Git/CLI work. Synthetic corpus is not retrieval-quality evidence. OS caches are not flushed.",
    "engine_version": "2.26.2",
    "query_sha256": hashlib.sha256(query.read_bytes()).hexdigest(),
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "index_sha256": hashlib.sha256(
        pathlib.Path(str(prefix) + ".usearch").read_bytes()
    ).hexdigest(),
    "runs": [],
}
for scope in [1, 10, 100]:
    exact = call("exact", 128, scope)
    for ef in map(int, a.expansions.split(",")):
        ann = call("query", ef, scope)
        recalls = {str(k): [] for k in [10, 100]}
        for expected, actual in zip(exact["results"], ann["results"], strict=True):
            for k in [10, 100]:
                gold = set(expected["commits"][:k])
                got = set(actual["commits"][:k])
                recalls[str(k)].append(len(gold & got) / len(gold))
        times = [r["seconds"] for r in ann["results"]]
        row = {
            "scope_divisor": scope,
            "eligible_commits": ann["eligible_commits"],
            "expansion": ef,
            "recall_mean": {k: statistics.mean(v) for k, v in recalls.items()},
            "recall_min": {k: min(v) for k, v in recalls.items()},
            "query_median_seconds": statistics.median(times),
            "query_p95_seconds": percentile(times, 0.95),
            "startup_seconds": ann["startup_seconds"],
            "process_wall_seconds": ann["process_wall_seconds"],
            "exact_query_median_seconds": statistics.median(
                r["seconds"] for r in exact["results"]
            ),
        }
        report["runs"].append(row)
        print(json.dumps(row), flush=True)
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if all(v >= 0.99 for v in row["recall_mean"].values()):
            break
