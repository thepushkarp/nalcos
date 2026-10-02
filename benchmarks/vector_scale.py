#!/usr/bin/env python3
"""Measure exact-vector CLI scale using explicitly synthetic document vectors.

An existing scale.py fixture is copied into a separate disposable index. Dense,
deterministic 384-dimensional vectors are installed there in resumable batches.
They do not represent document meaning. A cached native MiniLM model encodes the
queries, so latency includes real model startup, exact retrieval, live Git scope,
source verification, and JSON output. No retrieval-quality metric is valid here.
The canonical fixture and model-reference index are opened read-only.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import struct
import subprocess
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from scale import CHANGES, DOMAINS, digest, emit, git, host

DIMENSIONS = 384
ALGORITHM = "shake256-document-id-rademacher-384-v1"
MODEL_ID = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
PURPOSE = "SYNTHETIC VECTORS: exact-vector scale only; no retrieval-quality evidence"
SCALE = 1 / math.sqrt(DIMENSIONS)
PACKER = struct.Struct("<" + "f" * DIMENSIONS)


def read_only(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def synthetic_vector(document_id):
    signs = hashlib.shake_256(ALGORITHM.encode() + b"\0" + document_id.encode()).digest(
        DIMENSIONS
    )
    return PACKER.pack(*(SCALE if byte & 1 else -SCALE for byte in signs))


def runtime_conditions():
    """Keep power/thermal observations without battery or machine identifiers."""
    context = {"observed_at_utc": datetime.now(timezone.utc).isoformat()}
    pmset = Path("/usr/bin/pmset")
    if not pmset.is_file():
        return {**context, "available": False}
    battery = subprocess.run(
        [str(pmset), "-g", "batt"], capture_output=True, text=True, check=False
    )
    source = re.search(r"Now drawing from '([^']+)'", battery.stdout)
    charge = re.search(r"(\d+)%;\s*([^;]+);", battery.stdout)
    thermal = subprocess.run(
        [str(pmset), "-g", "therm"], capture_output=True, text=True, check=False
    )
    summary = [
        line.strip().removeprefix("Note: ")
        for line in (thermal.stdout + "\n" + thermal.stderr).splitlines()
    ]
    return {
        **context,
        "available": True,
        "power": {
            "command_exit_code": battery.returncode,
            "source": source.group(1) if source else None,
            "charge_percent": int(charge.group(1)) if charge else None,
            "charge_state": charge.group(2) if charge else None,
        },
        "thermal": {
            "command_exit_code": thermal.returncode,
            "reported_summary": [
                line
                for line in summary
                if line.startswith(
                    ("No thermal", "No performance", "No CPU", "CPU_", "GPU_")
                )
            ],
        },
        "limitation": "Current observation only; no inference about earlier runs or future thermal state.",
    }


def fixture_state(fixture):
    manifest = json.loads((fixture / "fixture.json").read_text())
    repo = fixture / "repository"
    if (
        git(repo, "rev-parse", "HEAD") != manifest["head"]
        or int(git(repo, "rev-list", "--count", "HEAD")) != manifest["commits"]
    ):
        raise RuntimeError("synthetic fixture does not match its creation manifest")
    indexes = list((fixture / "data" / "repos").glob("*/index.sqlite3"))
    if len(indexes) != 1:
        raise RuntimeError("fixture must contain exactly one canonical index")
    source = indexes[0]
    with closing(read_only(source)) as conn:
        if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise RuntimeError("vector scale harness requires index schema 1")
        commits = conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
        documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        omissions = conn.execute("SELECT COUNT(*) FROM omissions").fetchone()[0]
        generations = conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
    if commits != manifest["commits"] or omissions or documents < commits:
        raise RuntimeError("finish complete canonical indexing without omissions first")
    if generations:
        raise RuntimeError("source must be the lexical-only disposable scale fixture")
    return source, manifest, documents


def model_state(path):
    with closing(read_only(path)) as conn:
        row = conn.execute(
            "SELECT fingerprint,model,dimensions FROM generations WHERE state='active'"
        ).fetchone()
    if row is None:
        raise RuntimeError("model-reference index has no active generation")
    fingerprint, encoded, dimensions = row
    model = json.loads(encoded)
    profile = model["resolved"]["profile"]
    if (
        profile["id"] != MODEL_ID
        or profile["backend"] != "onnx"
        or dimensions != DIMENSIONS
        or profile["dimensions"] != DIMENSIONS
        or profile.get("endpoint") is not None
        or profile.get("api_key_env") is not None
    ):
        raise RuntimeError("reference must be a cached native 384-D MiniLM generation")
    for file in model["resolved"]["files"].values():
        if not Path(file["path"]).is_file():
            raise RuntimeError("reference model artifact is not locally available")
    descriptor_hash = hashlib.sha256(encoded.encode()).hexdigest()
    model["benchmark_vectors"] = {
        "synthetic": True,
        "algorithm": ALGORITHM,
        "purpose": PURPOSE,
    }
    return fingerprint, model, descriptor_hash


def prepare(args):
    fixture = args.fixture.resolve()
    work = args.work.resolve()
    source, manifest, documents = fixture_state(fixture)
    reference_fingerprint, model, descriptor_hash = model_state(args.model_index)
    if (
        work.is_relative_to(fixture)
        or source.is_relative_to(work)
        or args.model_index.resolve().is_relative_to(work)
        or work.is_relative_to(args.model_index.resolve().parent)
    ):
        raise RuntimeError("work must be separate from canonical and model index data")
    identity = {
        "schema_version": 1,
        "purpose": PURPOSE,
        "algorithm": ALGORITHM,
        "dimensions": DIMENSIONS,
        "fixture_head": manifest["head"],
        "commits": manifest["commits"],
        "documents": documents,
        "repo_identity": source.parent.name,
        "reference_fingerprint": reference_fingerprint,
        "reference_descriptor_sha256": descriptor_hash,
        "query_encoder": {
            key: model["resolved"]["profile"][key]
            for key in [
                "id",
                "backend",
                "artifact",
                "revision",
                "dimensions",
                "pooling",
                "max_tokens",
                "query_prefix",
            ]
        },
    }
    identity["query_encoder"]["artifact_sha256"] = model["resolved"]["files"][
        model["resolved"]["profile"]["artifact"]
    ]["sha256"]
    identity["query_encoder"]["native"] = True
    receipt_path = work / "preparation.json"
    if work.exists():
        if not args.resume or not receipt_path.is_file():
            raise RuntimeError(
                "work exists; use --resume only for this harness's own directory"
            )
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity:
            raise RuntimeError(
                "resume inputs differ from the existing preparation receipt"
            )
    else:
        work.mkdir(parents=True)
        receipt = {"identity": identity, "phase": "copying"}
        write_json(receipt_path, receipt)
        (work / "WARNING.txt").write_text(
            PURPOSE + "\nDo not use this index for real history-search decisions.\n"
        )
    destination = work / "data" / "repos" / source.parent.name / "index.sqlite3"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if receipt["phase"] == "copying":
        started = time.perf_counter()
        with (
            closing(read_only(source)) as original,
            closing(sqlite3.connect(destination)) as copied,
        ):
            original.backup(copied, pages=1024)
        receipt.update(phase="populating", backup_seconds=time.perf_counter() - started)
        write_json(receipt_path, receipt)
    fingerprint = (
        "synthetic-scale:"
        + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    )
    started = time.perf_counter()
    conn = sqlite3.connect(destination)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        generation = conn.execute(
            "SELECT id,fingerprint,state FROM generations"
        ).fetchall()
        if not generation:
            with conn:
                cursor = conn.execute(
                    "INSERT INTO generations(fingerprint,model,dimensions,watermark,state,created_at,allow_prior_reuse) "
                    "VALUES(?,?,?,(SELECT MAX(rowid) FROM documents),'staging',?,0)",
                    (fingerprint, json.dumps(model), DIMENSIONS, int(time.time())),
                )
                generation_id = cursor.lastrowid
                conn.execute(
                    "INSERT INTO metadata(key,value) VALUES('synthetic_vector_benchmark',?)",
                    (json.dumps(identity),),
                )
        elif len(generation) == 1 and generation[0][1] == fingerprint:
            generation_id = generation[0][0]
        else:
            raise RuntimeError("disposable database contains an unexpected generation")
        completed = conn.execute(
            "SELECT COUNT(*) FROM embedded_documents WHERE generation_id=?",
            (generation_id,),
        ).fetchone()[0]
        resumed_from = completed
        emit("synthetic_vectors_started", completed=completed, documents=documents)
        last_progress = time.monotonic()
        position = 0
        while True:
            pending = conn.execute(
                "SELECT d.rowid,d.id,length(CAST(d.text AS BLOB)) FROM documents d "
                "WHERE d.rowid>? AND NOT EXISTS(SELECT 1 FROM embedded_documents e "
                "WHERE e.generation_id=? AND e.document_id=d.id) ORDER BY d.rowid LIMIT 1000",
                (position, generation_id),
            ).fetchall()
            if not pending:
                break
            vectors = [
                (
                    generation_id,
                    doc_id,
                    0,
                    0,
                    length,
                    DIMENSIONS,
                    synthetic_vector(doc_id),
                )
                for _, doc_id, length in pending
                if length
            ]
            with conn:
                conn.executemany(
                    "INSERT INTO embedded_documents(generation_id,document_id) VALUES(?,?)",
                    [(generation_id, doc_id) for _, doc_id, _ in pending],
                )
                conn.executemany(
                    "INSERT INTO embeddings(generation_id,document_id,chunk_index,byte_start,byte_end,dimensions,vector) "
                    "VALUES(?,?,?,?,?,?,?)",
                    vectors,
                )
            position = pending[-1][0]
            completed += len(pending)
            if time.monotonic() - last_progress >= 2:
                emit(
                    "synthetic_vectors_progress",
                    completed=completed,
                    documents=documents,
                )
                last_progress = time.monotonic()
        nonempty = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE length(CAST(text AS BLOB))>0"
        ).fetchone()[0]
        vectors, invalid = conn.execute(
            "SELECT COUNT(*),COALESCE(SUM(dimensions<>? OR length(vector)<>?),0) "
            "FROM embeddings WHERE generation_id=?",
            (DIMENSIONS, PACKER.size, generation_id),
        ).fetchone()
        if completed != documents or vectors != nonempty or invalid:
            raise RuntimeError("synthetic vector coverage or shape is incomplete")
        sample = conn.execute(
            "SELECT vector FROM embeddings WHERE generation_id=? LIMIT 1",
            (generation_id,),
        ).fetchone()
        norm_error = 0.0
        if sample:
            norm_error = abs(sum(v * v for v in PACKER.unpack(sample[0])) - 1)
            if norm_error > 1e-5:
                raise RuntimeError(
                    "synthetic vectors are not normalized within tolerance"
                )
        with conn:
            conn.execute(
                "UPDATE generations SET state='active' WHERE id=?", (generation_id,)
            )
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        receipt.update(
            phase="ready",
            generation_id=generation_id,
            generation_fingerprint=fingerprint,
            completed_documents=completed,
            vectors=vectors,
            vector_bytes=vectors * PACKER.size,
            normalized_vector_squared_norm_error=norm_error,
            resumed_from_documents=resumed_from,
            latest_population_seconds=time.perf_counter() - started,
            fixture=manifest,
        )
        write_json(receipt_path, receipt)
    finally:
        conn.close()
    (work / "config.toml").write_text(
        'version = 1\n[runtime]\nquery_device = "cpu"\nindex_device = "cpu"\nthreads = 4\n'
    )
    emit("synthetic_vectors_ready", documents=documents, vectors=receipt["vectors"])
    return receipt


def measure(args, receipt):
    if receipt["phase"] != "ready":
        raise RuntimeError("complete synthetic vector preparation before measuring")
    work, fixture, binary = (
        args.work.resolve(),
        args.fixture.resolve(),
        args.binary.resolve(),
    )
    raw = work / (args.output.stem + "-queries")
    if args.output.exists() or raw.exists():
        raise RuntimeError(
            "measurement output exists; preserve it and choose a new output filename"
        )
    if not args.ort_library.is_file():
        raise RuntimeError("the explicit cached ONNX Runtime library is missing")
    fixture_state(fixture)
    binary_hash = digest(binary)
    conditions = runtime_conditions()
    env = {
        **os.environ,
        "NALCOS_DATA_DIR": str(work / "data"),
        "NALCOS_ORT_LIBRARY": str(args.ort_library.resolve()),
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    raw.mkdir()
    records = []
    for number in range(args.queries):
        query = (
            f"{DOMAINS[(number // 2) % len(DOMAINS)]} {CHANGES[(number // 2) % len(CHANGES)]}"
            if number % 2 == 0
            else f"event{1 + (number * 7919) % receipt['identity']['commits']:06d}"
        )
        command = [
            str(binary),
            "--repo",
            str(fixture / "repository"),
            "--config",
            str(work / "config.toml"),
            "--json",
            "--offline",
            "--timeout",
            "60s",
            "search",
            query,
            "--mode",
            "semantic",
            "--freshness",
            args.freshness,
            "--ref",
            receipt["identity"]["fixture_head"],
            "--limit",
            "10",
            "--max-bytes",
            "16384",
        ]
        started = time.perf_counter()
        process = subprocess.run(
            command, env=env, capture_output=True, timeout=65, check=False
        )
        elapsed = time.perf_counter() - started
        (raw / f"{number:02d}.json").write_bytes(process.stdout)
        (raw / f"{number:02d}.stderr").write_bytes(process.stderr)
        if process.returncode:
            raise RuntimeError(
                f"semantic query {number} failed; inspect local raw output"
            )
        payload = json.loads(process.stdout)
        evidence = [
            item for result in payload["results"] for item in result["evidence"]
        ]
        runtime = payload.get("runtime", {})
        if (
            payload.get("mode_used") != "semantic"
            or payload["history_coverage"]["complete"] is not True
            or payload["history_coverage"]["indexed"] != receipt["identity"]["commits"]
            or payload["embedding_coverage"]["complete"] is not True
            or payload["embedding_coverage"]["indexed"]
            != receipt["identity"]["documents"]
            or runtime.get("selected_device") != "cpu"
            or "ONNX Runtime 1.23.2" not in runtime.get("runtime_version", "")
            or not evidence
            or not all(item["verified"] for item in evidence)
        ):
            raise RuntimeError(
                "semantic measurement has incomplete coverage, wrong runtime, or unverified evidence"
            )
        records.append(
            {
                "query": query,
                "seconds": elapsed,
                "results": len(payload["results"]),
                "verified_evidence": len(evidence),
                "output_truncated": payload["output_truncated"],
                "runtime": runtime,
            }
        )
        emit("semantic_scale_query_complete", number=number + 1, seconds=elapsed)
    if digest(binary) != binary_hash:
        raise RuntimeError("binary changed during timing; discard this measurement")
    values = sorted(row["seconds"] for row in records)
    report = {
        "schema_version": 1,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": PURPOSE,
        "synthetic_vectors": True,
        "retrieval_quality_evaluated": False,
        "host": host(),
        "runtime_conditions_at_start": conditions,
        "binary_sha256": binary_hash,
        "runtime_library_sha256": digest(args.ort_library),
        "corpus": receipt["fixture"],
        "vectors": {
            "algorithm": ALGORITHM,
            "dimensions": DIMENSIONS,
            "count": receipt["vectors"],
            "bytes": receipt["vector_bytes"],
            "chunks_per_nonempty_document": 1,
            "derived_from_source_text": False,
            "normalized_vector_squared_norm_error": receipt[
                "normalized_vector_squared_norm_error"
            ],
        },
        "query_encoder": receipt["identity"]["query_encoder"],
        "measurement": {
            "process": "new CLI process per query; includes native encoder startup, exact vector scan, Git scope, evidence verification, and JSON output",
            "cache": "filesystem/page caches not flushed; first query retained; vector preparation precedes measurements",
            "freshness": args.freshness,
            "limit": 10,
            "evidence_budget_bytes": 16384,
            "threads": 4,
            "concurrency": args.concurrency_note,
            "percentile_method": "nearest-rank ceil(0.95*n)",
        },
        "search": {
            "n": len(values),
            "p50_seconds": statistics.median(values),
            "p95_seconds": values[math.ceil(0.95 * len(values)) - 1],
            "min_seconds": values[0],
            "max_seconds": values[-1],
            "samples": records,
        },
        "caveats": [
            "Vectors are synthetic random signs, not model embeddings of Git documents; rankings have no semantic meaning.",
            "Only query encoding uses the native model. This does not measure real corpus embedding throughput or retrieval quality.",
            "One vector per source document underestimates corpora whose tokenizer requires multiple chunks per document.",
            "The fixture has simple linear history and small textual patches; it does not represent all 100k-commit repositories.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    emit("semantic_scale_complete", p95_seconds=report["search"]["p95_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        required=True,
        help="Completed scale.py fixture directory",
    )
    parser.add_argument(
        "--work",
        type=Path,
        required=True,
        help="Separate new disposable output directory",
    )
    parser.add_argument(
        "--model-index",
        type=Path,
        required=True,
        help="Read-only reference with active native MiniLM",
    )
    parser.add_argument(
        "--ort-library",
        type=Path,
        required=True,
        help="Cached ONNX Runtime 1.23.2 library",
    )
    parser.add_argument("--binary", type=Path, default=Path("target/release/nalcos"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--freshness", choices=["cached", "auto"], default="cached")
    parser.add_argument("--resume", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--prepare-only", action="store_true")
    phase.add_argument("--measure-only", action="store_true")
    parser.add_argument(
        "--concurrency-note",
        default="No deliberate competing benchmark; ordinary host activity was not controlled",
    )
    args = parser.parse_args()
    if args.queries < 2:
        parser.error("at least two queries are required")
    if args.measure_only:
        receipt = json.loads((args.work / "preparation.json").read_text())
    else:
        receipt = prepare(args)
    if not args.prepare_only:
        measure(args, receipt)


if __name__ == "__main__":
    main()
