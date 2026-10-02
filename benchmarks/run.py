#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "onnxruntime==1.30.0", "tokenizers==0.23.2"]
# ///
"""Matched ONNX/GGUF embedding checks and local timing; never downloads models.

Use a pinned Hugging Face snapshot as --snapshot. GGUF checks require exact
native tokenizer agreement before comparing vectors from identical token ids.
The reference ONNX graph must belong to that same pinned snapshot.
"""

import argparse
import contextlib
import hashlib
import json
import platform
import re
import resource
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

STOP = {
    "when",
    "did",
    "does",
    "the",
    "which",
    "where",
    "what",
    "find",
    "change",
    "changed",
    "that",
    "a",
    "an",
    "to",
    "from",
    "for",
    "of",
    "and",
    "in",
    "with",
    "was",
    "were",
    "start",
    "started",
    "instead",
    "so",
    "could",
    "its",
    "it",
    "this",
    "then",
    "into",
    "before",
    "after",
    "using",
    "use",
    "used",
    "had",
    "been",
    "while",
    "on",
    "only",
    "more",
    "again",
    "below",
}
PROBES = [
    "",
    " ",
    "Find where retries were removed after authentication failures.",
    "- retry(request)\n+ return cached_response\n",
    "def hello_world(x: str):\n    return x.lower()",
    "naïve café 東京 🙂",
    "user_id != None && userId === 4",
    "<s> [CLS] [SEP] </s>",
    "padding and batch invariance",
    "return 0;\n" * 180,
]


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), file=sys.stderr, flush=True)


def rss_mib(who):
    value = resource.getrusage(who).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("embedding output is not a finite matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("embedding output contains a zero vector")
    return vectors / norms


class OnnxEncoder:
    def __init__(self, model, pooling, threads):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.pooling = pooling
        self.runtime = "onnxruntime " + ort.__version__

    def encode(self, encodings):
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            ),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        hidden = self.session.run(
            None, {k: v for k, v in feed.items() if k in self.inputs}
        )[0]
        if hidden.ndim == 2:
            vectors = hidden
        elif self.pooling == "cls":
            vectors = hidden[:, 0, :]
        else:
            mask = feed["attention_mask"][..., None]
            vectors = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        return normalize(vectors)


class LlamaEncoder:
    def __init__(self, base):
        self.base = base

    def post(self, path, body):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"{path}: {error.code}: {error.read().decode()[:2000]}"
            ) from error

    def tokenize(self, text):
        return self.post("/tokenize", {"content": text, "add_special": True})["tokens"]

    def encode(self, encodings):
        tokens = [
            [token for token, mask in zip(e.ids, e.attention_mask) if mask]
            for e in encodings
        ]
        rows = self.post(
            "/v1/embeddings",
            {"input": tokens, "model": "benchmark", "encoding_format": "float"},
        )["data"]
        rows.sort(key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(len(tokens))):
            raise ValueError("server returned missing or duplicate embedding indices")
        return normalize([row["embedding"] for row in rows])


@contextlib.contextmanager
def llama_server(args, logpath):
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    command = [
        args.llama_server,
        "--model",
        str(args.gguf),
        "--embedding",
        "--pooling",
        args.pooling,
        "--embd-normalize",
        "2",
        "--threads",
        str(args.threads),
        "--threads-batch",
        str(args.threads),
        "--parallel",
        str(args.batch_size),
        "--ctx-size",
        str(args.max_tokens * args.batch_size),
        "--batch-size",
        str(args.max_tokens * args.batch_size),
        "--ubatch-size",
        str(args.max_tokens * args.batch_size),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--offline",
        "--no-webui",
        "--no-cache-prompt",
        "--no-warmup",
    ]
    if args.backend == "cpu":
        command += ["--device", "none", "--gpu-layers", "0", "--no-op-offload"]
    else:
        command += ["--gpu-layers", "all"]
    version = subprocess.check_output(
        [args.llama_server, "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()
    base = f"http://127.0.0.1:{port}"
    with logpath.open("w") as log:
        start = time.perf_counter()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while time.perf_counter() - start < 60:
                if child.poll() is not None:
                    raise RuntimeError(
                        "llama-server exited: " + logpath.read_text()[-3000:]
                    )
                try:
                    with urllib.request.urlopen(
                        base + "/health", timeout=0.5
                    ) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("llama-server did not become healthy in 60 seconds")
            encoder = LlamaEncoder(base)
            encoder.runtime = version
            # Paths are deliberately omitted from portable metadata.
            portable_command = [
                "llama-server"
                if i == 0
                else ("${MODEL}" if value == str(args.gguf) else value)
                for i, value in enumerate(command)
            ]
            yield encoder, time.perf_counter() - start, portable_command
        finally:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def compare(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(
            f"vector shape mismatch: {reference.shape} != {candidate.shape}"
        )
    cosines = np.sum(normalize(reference) * normalize(candidate), axis=1)
    return {
        "cosine_min": float(np.min(cosines)),
        "cosine_mean": float(np.mean(cosines)),
        "max_absolute_error": float(np.max(np.abs(reference - candidate))),
    }


def parity(args, encoder, tokenizer, texts):
    # Check untruncated native tokenization, including code punctuation, special
    # tokens and Unicode. Explicit token ids then isolate graph/pooling differences.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    samples = PROBES + texts[:16]
    mismatches = []
    for index, text in enumerate(samples):
        expected = tokenizer.encode(text).ids
        actual = encoder.tokenize(text)
        if expected != actual:
            first = next(
                (
                    i
                    for i, pair in enumerate(zip(expected, actual))
                    if pair[0] != pair[1]
                ),
                min(len(expected), len(actual)),
            )
            mismatches.append(
                {
                    "sample": index,
                    "first_difference": first,
                    "reference_tokens": len(expected),
                    "candidate_tokens": len(actual),
                    "reference_slice": expected[max(0, first - 2) : first + 3],
                    "candidate_slice": actual[max(0, first - 2) : first + 3],
                }
            )
    configure_tokenizer(tokenizer, args)
    reference = OnnxEncoder(args.snapshot / args.onnx_file, args.pooling, args.threads)
    ref_vectors, vectors = [], []
    for start in range(0, len(samples), args.batch_size):
        encodings = tokenizer.encode_batch(samples[start : start + args.batch_size])
        ref_vectors.append(reference.encode(encodings))
        vectors.append(encoder.encode(encodings))
    ref_vectors, vectors = np.concatenate(ref_vectors), np.concatenate(vectors)
    comparison = compare(ref_vectors, vectors)
    # A singleton versus padded batch detects pooling over padding and batch bugs.
    singleton = encoder.encode(tokenizer.encode_batch([samples[2]]))
    reference_singleton = reference.encode(tokenizer.encode_batch([samples[2]]))
    invariant = compare(singleton, vectors[2:3])
    reference_invariant = compare(reference_singleton, ref_vectors[2:3])
    passed = (
        not mismatches
        and comparison["cosine_min"] >= args.min_cosine
        and comparison["max_absolute_error"] <= args.max_abs
        and invariant["cosine_min"] >= args.min_cosine
        and reference_invariant["cosine_min"] >= args.min_cosine
    )
    return {
        "passed": passed,
        "tokenizer_exact": not mismatches,
        "tokenizer_mismatches": mismatches,
        "sample_count": len(samples),
        "vector_comparison": comparison,
        "batch_invariance": invariant,
        "reference_batch_invariance": reference_invariant,
        "thresholds": {
            "minimum_cosine": args.min_cosine,
            "maximum_absolute_error": args.max_abs,
        },
        "reference_runtime": reference.runtime,
        "note": "Finite unit vectors and identical explicit token ids, masked mean/CLS pooling, L2 normalization. Native raw-text tokenization must also match; a vector pass does not override tokenizer failure.",
    }


def configure_tokenizer(tokenizer, args):
    config = json.loads((args.snapshot / "tokenizer_config.json").read_text())
    pad = config.get("pad_token", "[PAD]")
    if isinstance(pad, dict):
        pad = pad["content"]
    pad_id = tokenizer.token_to_id(pad)
    if pad_id is None:
        raise ValueError(f"tokenizer has no configured padding token: {pad}")
    tokenizer.enable_truncation(max_length=args.max_tokens)
    tokenizer.enable_padding(pad_id=pad_id, pad_token=pad)


def rankings(corpus, documents, queries):
    docs = corpus["documents"]

    def commits(indices):
        return list(dict.fromkeys(docs[int(index)]["sha"] for index in indices))

    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE VIRTUAL TABLE docs USING fts5(text)")
        db.executemany(
            "INSERT INTO docs(rowid,text) VALUES (?,?)",
            [(i + 1, doc["text"]) for i, doc in enumerate(docs)],
        )
        output = {
            mode: []
            for mode in (
                "semantic_messages",
                "semantic_message_diff",
                "lexical",
                "hybrid",
            )
        }
        message_ids = np.array(
            [i for i, doc in enumerate(docs) if doc["kind"] == "message"], dtype=int
        )
        for i, query in enumerate(corpus["queries"]):
            scores = documents @ queries[i]
            semantic = commits(np.argsort(-scores, kind="stable"))
            messages = commits(
                message_ids[np.argsort(-scores[message_ids], kind="stable")]
            )
            terms = sorted(
                set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", query["query"].lower()))
                - STOP
            )
            expression = " OR ".join('"' + term + '"' for term in terms)
            rows = (
                db.execute(
                    "SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs),rowid",
                    (expression,),
                ).fetchall()
                if expression
                else []
            )
            lexical = commits([row[0] - 1 for row in rows])
            fused = {}
            for order in (semantic[:50], lexical[:50]):
                for rank, oid in enumerate(order, 1):
                    fused[oid] = fused.get(oid, 0) + 1 / (60 + rank)
            hybrid = sorted(fused, key=lambda oid: (-fused[oid], oid))
            for mode, order in zip(output, (messages, semantic, lexical, hybrid)):
                output[mode].append(
                    {
                        "query_id": query.get("id", f"nalcos-pilot-{i + 1:02}"),
                        "snapshot": corpus["head"],
                        "query_sha256": hashlib.sha256(
                            query["query"].encode()
                        ).hexdigest(),
                        "commits": order,
                        "complete": True,
                        "abstained": False,
                    }
                )
        return output


def benchmark(args, encoder, tokenizer, corpus):
    text = [doc["text"] for doc in corpus["documents"]]
    query_text = [args.query_prefix + query["query"] for query in corpus["queries"]]
    if not text or not query_text:
        raise ValueError("benchmark requires documents and queries")
    tokenizer.no_truncation()
    tokenizer.no_padding()
    counts = [len(encoded.ids) for encoded in tokenizer.encode_batch(text)]
    configure_tokenizer(tokenizer, args)

    def encode(values):
        return encoder.encode(tokenizer.encode_batch(values))

    start = time.perf_counter()
    encode(query_text[:1])
    first = time.perf_counter() - start
    for _ in range(3):
        encode(query_text[:1])
    latency = []
    for i in range(args.query_samples):
        start = time.perf_counter()
        encode([query_text[i % len(query_text)]])
        latency.append(time.perf_counter() - start)
    qv = np.concatenate(
        [
            encode(query_text[i : i + args.batch_size])
            for i in range(0, len(query_text), args.batch_size)
        ]
    )
    encode(text[: args.batch_size])
    durations = []
    dv = None
    for repeat in range(args.repeats):
        vectors = []
        start = time.perf_counter()
        for i in range(0, len(text), args.batch_size):
            vectors.append(encode(text[i : i + args.batch_size]))
        durations.append(time.perf_counter() - start)
        if repeat == 0:
            dv = np.concatenate(vectors)
        emit(
            "index_pass", repeat=repeat + 1, documents=len(text), seconds=durations[-1]
        )
    np.savez(args.output / f"{args.label}.npz", documents=dv, queries=qv)
    for mode, rows in rankings(corpus, dv, qv).items():
        (args.output / f"{args.label}.rankings.{mode}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    return {
        "dimensions": dv.shape[1],
        "documents": len(text),
        "truncated_documents": sum(n > args.max_tokens for n in counts),
        "tokens_mean": float(np.mean(counts)),
        "tokens_max": max(counts),
        "first_query_ms": first * 1000,
        "warm_query_p50_ms": float(np.percentile(latency, 50)) * 1000,
        "warm_query_p95_ms": float(np.percentile(latency, 95)) * 1000,
        "query_latency_ms": [elapsed * 1000 for elapsed in latency],
        "query_warmups": 3,
        "corpus_seconds": durations,
        "docs_per_second": [len(text) / elapsed for elapsed in durations],
        "unit_norm_max_error": float(np.max(np.abs(np.linalg.norm(dv, axis=1) - 1))),
        "query_timing_includes": "tokenization, embedding, normalization; GGUF includes loopback HTTP; excludes search/Git/CLI startup",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "benchmark"], required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--onnx-file", default="onnx/model.onnx")
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--backend", choices=["cpu", "metal"], default="cpu")
    parser.add_argument("--llama-server", default="llama-server")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-samples", type=int, default=36)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--max-abs", type=float, default=0.002)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Fixed corpus JSON; the archived pilot includes benchmarks/pilot/2026-10-02/corpus.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if (
        min(
            args.max_tokens,
            args.threads,
            args.batch_size,
            args.query_samples,
            args.repeats,
        )
        <= 0
    ):
        parser.error(
            "token, thread, batch, sample and repetition counts must be positive"
        )
    if args.mode == "check" and not args.gguf:
        parser.error("check requires --gguf and the matching reference ONNX snapshot")
    if not args.gguf and args.backend != "cpu":
        parser.error("ONNX runner supports CPU only")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.label):
        parser.error(
            "label must contain only letters, numbers, dots, hyphens and underscores"
        )
    if (
        not re.fullmatch(r"[0-9a-f]{40}", args.revision)
        or args.snapshot.name != args.revision
    ):
        parser.error(
            "snapshot basename must match the full pinned Hugging Face revision"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if (
        any(args.output.glob(args.label + ".*"))
        or (args.output / f"{args.label}-server.log").exists()
    ):
        parser.error(
            "output label already exists; use a fresh label to preserve evidence"
        )
    artifact = args.gguf or args.snapshot / args.onnx_file
    artifact_hash = digest(artifact)
    if args.mode == "benchmark" and args.gguf:
        if not args.parity_report:
            parser.error(
                "GGUF timing requires a passing --parity-report from this artifact/backend"
            )
        report = json.loads(args.parity_report.read_text())
        if (
            not report.get("parity", {}).get("passed")
            or report.get("artifact_sha256") != artifact_hash
            or report.get("backend") != args.backend
            or report.get("revision") != args.revision
            or report.get("max_tokens") != args.max_tokens
            or report.get("pooling") != args.pooling
            or report.get("model_id") != args.model_id
            or report.get("tokenizer_sha256")
            != digest(args.snapshot / "tokenizer.json")
            or report.get("reference_artifact_sha256")
            != digest(args.snapshot / args.onnx_file)
        ):
            parser.error(
                "parity report does not pass for this artifact/backend/revision/token limit/pooling"
            )
    corpus = json.loads(args.corpus.read_text())
    tokenizer = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
    configure_tokenizer(tokenizer, args)
    result = {
        "schema_version": 1,
        "label": args.label,
        "mode": args.mode,
        "model_id": args.model_id,
        "revision": args.revision,
        "artifact_sha256": artifact_hash,
        "artifact_bytes": artifact.stat().st_size,
        "tokenizer_sha256": digest(args.snapshot / "tokenizer.json"),
        "corpus_sha256": digest(args.corpus),
        "backend": args.backend,
        "pooling": args.pooling,
        "normalization": "L2",
        "max_tokens": args.max_tokens,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "query_prefix": args.query_prefix,
        "platform": platform.system().lower() + "-" + platform.machine(),
        "python": platform.python_version(),
        "harness_sha256": digest(Path(__file__)),
    }

    def run(encoder, load_seconds):
        if (
            args.mode == "benchmark"
            and args.gguf
            and report.get("runtime") != encoder.runtime
        ):
            raise ValueError(
                "runtime differs from parity report; recheck correctness before timing"
            )
        result.update(runtime=encoder.runtime, load_seconds=load_seconds)
        if args.mode == "check":
            result["reference_artifact_sha256"] = digest(args.snapshot / args.onnx_file)
            result["parity"] = parity(
                args, encoder, tokenizer, [d["text"] for d in corpus["documents"]]
            )
        else:
            result.update(benchmark(args, encoder, tokenizer, corpus))

    if args.gguf:
        logpath = args.output / f"{args.label}-server.log"
        with llama_server(args, logpath) as (encoder, elapsed, command):
            result["startup_command"] = command
            run(encoder, elapsed)
        result["server_log_sha256"] = digest(logpath)
        result["server_peak_rss_mib"] = rss_mib(resource.RUSAGE_CHILDREN)
    else:
        start = time.perf_counter()
        encoder = OnnxEncoder(artifact, args.pooling, args.threads)
        run(encoder, time.perf_counter() - start)
    result["driver_peak_rss_mib"] = rss_mib(resource.RUSAGE_SELF)
    output = args.output / f"{args.label}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    return 3 if args.mode == "check" and not result["parity"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
