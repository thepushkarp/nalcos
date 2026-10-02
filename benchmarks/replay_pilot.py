#!/usr/bin/env python3
"""Export the archived pilot's original Python sources as runnable local scripts.

These scripts intentionally retain the exploratory method, including its
documented token-count bug. Prefer run.py for new matched comparisons.
"""

import argparse
import json
import shlex
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--llama-server", default="llama-server")
    args = parser.parse_args()
    archive = Path(__file__).parent / "pilot/2026-10-02"
    harness = json.loads((archive / "harness.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in harness.items():
        if not isinstance(source, str):
            continue
        source = source.replace(
            "Path('${NALCOS_BENCH_OUTPUT}')", f"Path({str(args.output.resolve())!r})"
        )
        source = source.replace(
            "Path('${HOME}/projects/nalcos')", f"Path({str(args.repo.resolve())!r})"
        )
        source = source.replace("'llama-server'", repr(args.llama_server))
        (args.output / f"{name}.py").write_text(source)
    (args.output / "corpus.json").write_bytes((archive / "corpus.json").read_bytes())
    (args.output / "onnx_configs.json").write_text(
        json.dumps(harness["onnx_configs"], indent=2) + "\n"
    )
    print(f"Exported scripts and the fixed pilot corpus to {args.output}")
    print(
        "Example ONNX configuration:",
        shlex.quote(json.dumps(harness["onnx_configs"][0])),
    )


if __name__ == "__main__":
    main()
