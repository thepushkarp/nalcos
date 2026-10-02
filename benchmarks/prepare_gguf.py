#!/usr/bin/env python3
"""Convert an existing pinned HF snapshot with a reviewed llama.cpp checkout.

No models, code or dependencies are downloaded by this command. Run it in the
converter's pinned Python environment. The first artifact is always FP32;
quantization is a separate experiment after tokenizer/vector parity passes.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot.name != args.revision or len(args.revision) != 40:
        parser.error("snapshot basename must equal the full pinned HF revision")
    if args.output.exists():
        parser.error("output exists; choose a new output to preserve its identity")
    converter = args.llama_source / "convert_hf_to_gguf.py"
    if not converter.is_file():
        parser.error("llama-source must contain convert_hf_to_gguf.py")
    weights = sorted(args.snapshot.glob("*.safetensors"))
    if not weights:
        parser.error("snapshot must contain original safetensors weights")
    source_files = [
        converter,
        *sorted((args.llama_source / "conversion").rglob("*.py")),
        *sorted((args.llama_source / "gguf-py").rglob("*.py")),
    ]
    files = {
        str(path.relative_to(args.llama_source)): sha256(path) for path in source_files
    }
    metadata = {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "llama_cpp_revision": args.llama_revision,
        "precision": "F32",
        "source_weights": {path.name: sha256(path) for path in weights},
        "tokenizer_sha256": sha256(args.snapshot / "tokenizer.json"),
        "pooling_config": json.loads(
            (args.snapshot / "1_Pooling/config.json").read_text()
        ),
        "converter_source_tree_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "converter_sha256": files["convert_hf_to_gguf.py"],
        "conversion_command": [
            "python",
            "${LLAMA_SOURCE}/convert_hf_to_gguf.py",
            "${HF_SNAPSHOT}",
            "--outfile",
            "${OUTPUT}",
            "--outtype",
            "f32",
        ],
        "qualification": "Unverified until tokenizer, pooling and vector parity pass; conversion success alone is insufficient.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(converter),
        str(args.snapshot),
        "--outfile",
        str(args.output),
        "--outtype",
        "f32",
    ]
    subprocess.run(
        command,
        check=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "4",
        },
    )
    metadata.update(
        artifact_sha256=sha256(args.output), artifact_bytes=args.output.stat().st_size
    )
    args.output.with_suffix(".conversion.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
