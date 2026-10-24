"""Shared local corpus validation and direct-child resource measurement."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from evaluate import OID

GIT_ENV = {"GIT_NO_LAZY_FETCH": "1", "GIT_NO_REPLACE_OBJECTS": "1"}


def load_repositories(path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"{path}: expected repository manifest version 1")
    repositories = payload.get("repositories")
    if not isinstance(repositories, dict) or not repositories:
        raise ValueError(f"{path}: repositories must be a nonempty mapping")
    for key, repo in repositories.items():
        if not key or Path(key).name != key or key in {".", ".."}:
            raise ValueError(f"{path}: invalid repository key {key!r}")
        if not isinstance(repo, dict):
            raise TypeError(f"{path}: {key}: repository must be an object")
        for field in ("url", "language"):
            if not isinstance(repo.get(field), str) or not repo[field].strip():
                raise ValueError(f"{path}: {key}: {field} must be a nonempty string")
        snapshot = repo.get("snapshot")
        if not isinstance(snapshot, str) or not OID.fullmatch(snapshot):
            raise ValueError(f"{path}: {key}: snapshot must be a full Git object id")
        if repo.get("split") not in {"exploratory", "development", "heldout"}:
            raise ValueError(f"{path}: {key}: invalid split")
        scope = repo.get("semantic_range")
        if scope is not None and (
            not isinstance(scope, str)
            or len(scope.split("..")) != 2
            or not all(OID.fullmatch(oid) for oid in scope.split(".."))
        ):
            raise ValueError(f"{path}: {key}: semantic_range must be full OIDs A..B")
    return repositories


def validate_clone(clone, repo):
    for oid in [repo["snapshot"], *(repo.get("semantic_range") or "").split("..")]:
        if not oid:
            continue
        process = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "--verify", oid + "^{commit}"],
            capture_output=True,
            text=True,
            env={**os.environ, **GIT_ENV},
            check=False,
        )
        if process.returncode or process.stdout.strip() != oid:
            raise ValueError(
                f"verify clone {clone}: commit {oid} is unavailable: {process.stderr.strip()}"
            )


def rss_bytes(value):
    return int(value) if sys.platform == "darwin" else int(value * 1024)


def directory_size(directory):
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def run_measured(command, stdout, stderr, *, env, timeout, cwd=None):
    """Use files so verbose children cannot deadlock while wait4 owns reaping.

    Popen.wait/communicate would reap the child and lose its per-process rusage.
    A separate process group lets a deadline also stop outstanding tool children.
    RSS describes the direct child, not the sum of concurrent descendants.
    """
    started = time.monotonic()
    with Path(stdout).open("wb") as output, Path(stderr).open("wb") as errors:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=errors,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        timed_out = False
        try:
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    break
                if time.monotonic() - started >= timeout:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    _, status, usage = os.wait4(process.pid, 0)
                    break
                time.sleep(0.01)
            process.returncode = os.waitstatus_to_exitcode(status)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            _, status, _ = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            raise
    return {
        "elapsed_seconds": time.monotonic() - started,
        "exit_code": 124 if timed_out else process.returncode,
        "timed_out": timed_out,
        "peak_rss_bytes": rss_bytes(usage.ru_maxrss),
    }
