"""Synthetic vector profiles read generation snapshots without model execution."""

import argparse
import hashlib
import json
import math
import sqlite3
import struct
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from test_index_matrix import executable, local_repo
from vector_scale import (
    MODEL_ID,
    PROFILES,
    measure,
    model_state,
    prepare,
    synthetic_vector,
)


class VectorScaleTests(unittest.TestCase):
    def test_default_vector_bytes_are_backward_compatible_and_768_is_normalized(self):
        algorithm = b"shake256-document-id-rademacher-384-v1"
        signs = hashlib.shake_256(algorithm + b"\0doc").digest(384)
        expected = struct.pack(
            "<384f",
            *(1 / math.sqrt(384) if b & 1 else -1 / math.sqrt(384) for b in signs),
        )
        self.assertEqual(synthetic_vector("doc"), expected)
        values = struct.unpack("<768f", synthetic_vector("doc", 768))
        self.assertAlmostEqual(sum(v * v for v in values), 1, places=5)

    def test_coderank_generation_preparation_and_measurement_need_no_ort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.mkdir()
            head, _ = local_repo(fixture / "repository")
            manifest = {"head": head, "commits": 2}
            (fixture / "fixture.json").write_text(json.dumps(manifest))
            source = fixture / "data/repos/fixture-id/index.sqlite3"
            source.parent.mkdir(parents=True)
            with closing(sqlite3.connect(source)) as conn, conn:
                conn.executescript("""
                    PRAGMA user_version=1;
                    CREATE TABLE commits(oid TEXT);
                    CREATE TABLE documents(id TEXT PRIMARY KEY, text TEXT);
                    CREATE TABLE omissions(record TEXT);
                    CREATE TABLE generations(id INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT, model TEXT, dimensions INTEGER, watermark INTEGER, state TEXT, created_at INTEGER, allow_prior_reuse INTEGER);
                    CREATE TABLE embedded_documents(generation_id INTEGER, document_id TEXT, PRIMARY KEY(generation_id,document_id));
                    CREATE TABLE embeddings(generation_id INTEGER, document_id TEXT, chunk_index INTEGER, byte_start INTEGER, byte_end INTEGER, dimensions INTEGER, vector BLOB);
                    CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
                    INSERT INTO commits VALUES('one'),('two');
                    INSERT INTO documents VALUES('one','source text'),('two','other source');
                """)
            model_index = root / "reference/index.sqlite3"
            model_index.parent.mkdir()
            artifact = root / "fake.gguf"
            artifact.write_bytes(b"fixture artifact; never loaded")
            model = {
                "resolved": {
                    "profile": {
                        "id": PROFILES["coderank"]["model_id"],
                        "backend": "gguf",
                        "dimensions": 768,
                        "artifact": "CodeRankEmbed-f16.gguf",
                        "revision": "a" * 40,
                        "pooling": "cls",
                        "max_tokens": 512,
                        "query_prefix": "",
                        "document_prefix": "",
                    },
                    "files": {
                        "CodeRankEmbed-f16.gguf": {
                            "path": str(artifact),
                            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        }
                    },
                },
                "chunk_overlap_tokens": 32,
                "provider_probe": None,
                "runtime": {"provider": "Gguf", "selected_device": "cpu"},
            }
            with closing(sqlite3.connect(model_index)) as conn, conn:
                conn.execute(
                    "CREATE TABLE generations(fingerprint TEXT, model TEXT, dimensions INTEGER, state TEXT)"
                )
                conn.execute(
                    "INSERT INTO generations VALUES(?,?,768,'active')",
                    ("reference", json.dumps(model)),
                )
            reference_bytes = model_index.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "384-D"):
                model_state(model_index)
            binary = executable(
                root / "fake-nalcos",
                """
                import json, sys
                assert "--offline" in sys.argv
                print(json.dumps({
                    "mode_used": "semantic", "history_coverage": {"complete": True, "indexed": 2},
                    "embedding_coverage": {"complete": True, "indexed": 2},
                    "runtime": {"provider": "Gguf", "selected_device": "cpu", "runtime_version": "llama.cpp fixture"},
                    "results": [{"evidence": [{"verified": True}]}], "output_truncated": False,
                }))
            """,
            )
            args = argparse.Namespace(
                fixture=fixture,
                work=root / "work",
                model_index=model_index,
                resume=False,
                dimensions=768,
                model_id=PROFILES["coderank"]["model_id"],
                backend="gguf",
                profile="coderank",
                device="cpu",
                ort_library=None,
                output=root / "measurement.json",
                binary=binary,
                queries=2,
                freshness="cached",
                concurrency_note="fixture",
            )
            receipt = prepare(args)
            self.assertEqual(receipt["vector_bytes"], 2 * 768 * 4)
            self.assertEqual(receipt["identity"]["query_encoder"]["backend"], "gguf")
            with closing(
                sqlite3.connect(args.work / "data/repos/fixture-id/index.sqlite3")
            ) as conn:
                dimensions, length = conn.execute(
                    "SELECT dimensions,length(vector) FROM embeddings LIMIT 1"
                ).fetchone()
                self.assertEqual((dimensions, length), (768, 3072))
                saved = json.loads(
                    conn.execute(
                        "SELECT model FROM generations WHERE state='active'"
                    ).fetchone()[0]
                )
                self.assertEqual(saved["runtime"], model["runtime"])
            measure(args, receipt)
            report = json.loads(args.output.read_text())
            self.assertEqual(report["vectors"]["dimensions"], 768)
            self.assertIsNone(report["runtime_library_sha256"])
            self.assertEqual(report["search"]["n"], 2)
            self.assertEqual(model_index.read_bytes(), reference_bytes)
            args.resume = True
            self.assertEqual(prepare(args)["vectors"], 2)
            args.dimensions = 384
            args.model_id = MODEL_ID
            with self.assertRaisesRegex(RuntimeError, "reference"):
                prepare(args)


if __name__ == "__main__":
    unittest.main()
