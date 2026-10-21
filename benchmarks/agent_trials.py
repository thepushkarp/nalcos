#!/usr/bin/env python3
"""Plan or collect isolated history-search trials and export blind reviews.

Collection requires --run. Each agent has its own workflows.py trial file because
that evaluator pairs by query/condition, not by agent. Token budgets are requested
in the prompt and checked against reported usage; neither CLI exposes a hard
total-token cap. Wall deadlines terminate the trial's process group.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evaluate import OID, read_jsonl, validate_manifest
from harness import GIT_ENV, load_repositories, run_measured, validate_clone
from scale import digest
from workflows import evaluate_trials

CONDITIONS = ["git_rg", "nalcos_git_rg"]
AGENTS = ["claude", "codex"]
PROTOCOL = "Use only local Git history, rg, and file-reading tools available in this repository. Do not use the network, other repositories, or delegate to other agents. Do not modify repository files. Cite full commit IDs, paths, and supporting hunks; for absence, describe the history and searches covered."


def task_manifest(tasks, repositories):
    rows = []
    for task in tasks:
        identifier, key = task.get("id"), task.get("repo_key")
        if not isinstance(key, str) or key not in repositories:
            raise ValueError(f"task {identifier!r}: unknown repo_key {key!r}")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise ValueError(f"task {identifier!r}: prompt must be a nonempty string")
        answer_type = task.get("answer_type")
        if answer_type not in {"find", "absence", "explain"}:
            raise ValueError(
                f"task {identifier!r}: answer_type must be find, absence, or explain"
            )
        repo = repositories[key]
        rows.append(
            {
                "id": identifier,
                "repo_key": key,
                "repo": repo["url"],
                "snapshot": repo["snapshot"],
                "query": task["prompt"],
                "split": repo["split"],
                "origin": task.get("origin", "retrospective"),
                "frozen_before_tuning": task.get("frozen_before_tuning", False),
                "task": "find",
                "answer_type": answer_type,
                "answerable": answer_type != "absence",
                "gold_commits": task.get("gold_commits"),
            }
        )
    if not rows:
        raise ValueError("tasks manifest must be nonempty")
    return validate_manifest(rows)


def trial_prompt(query, budget_seconds, budget_tokens):
    return f"{PROTOCOL}\nBudget: {budget_seconds:g} seconds and {budget_tokens} total input/output tokens, including cached input. Stop within both budgets.\n\n{query['query']}"


def agent_command(agent, executable, prompt, condition, max_turns=20, model=None):
    if agent == "claude":
        allowed = ["Read", "Grep", "Glob", "Bash(git *)", "Bash(rg *)"]
        if condition == "nalcos_git_rg":
            allowed += ["Bash(nalcos *)", "Skill(nalcos)"]
        command = [
            str(executable),
            "-p",
            prompt,
            "--output-format",
            "json",
            "--allowed-tools",
            ",".join(allowed),
            "--tools",
            "Read,Grep,Glob,Bash,Skill",
            "--max-turns",
            str(max_turns),
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            "{}",
        ]
    else:
        command = [
            str(executable),
            "exec",
            "--json",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--ignore-user-config",
            "-c",
            'approval_policy="never"',
            prompt,
        ]
    if model:
        command += ["--model", model]
    return command


def planned_trials(manifest, args):
    planned = []
    for index, query in enumerate(manifest):
        reverse = args.order == "ba" or (
            args.order == "counterbalanced" and index % 2 == 1
        )
        order = list(reversed(CONDITIONS)) if reverse else CONDITIONS
        for agent in args.agents:
            for position, condition in enumerate(order):
                if condition not in args.conditions:
                    continue
                directory = (
                    args.output.resolve() / agent / f"trial-{index:03}-{position}"
                )
                command = agent_command(
                    agent,
                    getattr(args, agent),
                    trial_prompt(query, args.budget_seconds, args.budget_tokens),
                    condition,
                    args.max_turns,
                    getattr(args, agent + "_model"),
                )
                planned.append(
                    {
                        "query": query,
                        "agent": agent,
                        "condition": condition,
                        "order": position,
                        "directory": str(directory),
                        "command": command,
                    }
                )
    return planned


def parse_agent_output(agent, text):
    if agent == "claude":
        payload = json.loads(text)
        usage = payload.get("usage")
        answer = payload.get("result", "")
        fields = (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
        usages = [usage]
        failed = payload.get("is_error") is True
    else:
        events = [json.loads(line) for line in text.splitlines() if line.strip()]
        usages = [e.get("usage") for e in events if e.get("type") == "turn.completed"]
        answers = [
            e["item"].get("text", "")
            for e in events
            if e.get("type") == "item.completed"
            and e.get("item", {}).get("type") == "agent_message"
        ]
        answer = answers[-1] if answers else ""
        # Codex input_tokens already includes cached_input_tokens.
        fields = ("input_tokens", "output_tokens")
        failed = any(e.get("type") in {"error", "turn.failed"} for e in events)
    if not usages or any(
        not isinstance(u, dict)
        or any(
            type(u.get(k)) is not int or u[k] < 0
            for k in ("input_tokens", "output_tokens")
        )
        for u in usages
    ):
        raise ValueError(f"{agent} transcript has no valid input/output token usage")
    for usage in usages:
        if any(type(usage.get(k, 0)) is not int or usage.get(k, 0) < 0 for k in fields):
            raise ValueError(f"{agent} transcript has invalid token usage")
    if not isinstance(answer, str):
        raise TypeError(f"{agent} transcript final answer must be a string")
    return {
        "tokens_used": sum(u.get(k, 0) for u in usages for k in fields),
        "usage": usages,
        "final_answer": answer,
        "agent_failed": failed,
    }


def scratch_clone(source, destination, snapshot):
    env = {
        **os.environ,
        **GIT_ENV,
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    commands = [
        [
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            "--no-checkout",
            "--",
            str(source),
            str(destination),
        ],
        ["git", "-C", str(destination), "checkout", "--quiet", "--detach", snapshot],
        ["git", "-C", str(destination), "remote", "remove", "origin"],
    ]
    for command in commands:
        process = subprocess.run(
            command, env=env, capture_output=True, text=True, check=False
        )
        if process.returncode:
            raise ValueError(
                f"prepare scratch clone {source} at {snapshot}: {command}: {process.stderr.strip()}"
            )


def collect_trial(plan, args, manifest):
    query, agent, condition = plan["query"], plan["agent"], plan["condition"]
    directory = Path(plan["directory"])
    directory.mkdir(parents=True)
    repository = directory / "repository"
    scratch_clone(
        (args.clones / query["repo_key"]).resolve(), repository, query["snapshot"]
    )
    for root in (".claude", ".agents"):
        destination = repository / root / "skills" / "nalcos"
        if (
            not destination.resolve().is_relative_to(repository.resolve())
            or destination.exists()
        ):
            raise ValueError(
                f"prepare trial skill {destination}: expected an absent path inside scratch clone"
            )
        if condition == "nalcos_git_rg":
            shutil.copytree(args.skill, destination)
    tools = directory / "bin"
    tools.mkdir()
    if condition == "nalcos_git_rg":
        (tools / "nalcos").symlink_to(args.binary.resolve())
    else:
        # A preinstalled CLI on the host PATH must not give the control arm NaLCoS.
        (tools / "nalcos").write_text(
            "#!/bin/sh\necho 'nalcos is unavailable in this trial' >&2\nexit 127\n"
        )
        (tools / "nalcos").chmod(0o755)
    env = {
        **os.environ,
        **GIT_ENV,
        "PATH": str(tools) + os.pathsep + os.environ.get("PATH", ""),
        "NALCOS_DATA_DIR": str(directory / "data"),
        "NALCOS_RUNTIME_CACHE": str(directory / "runtime"),
    }
    config = directory / "nalcos.toml"
    config.write_text("version = 1\n")
    env["NALCOS_CONFIG"] = str(config)
    executable = shutil.which(plan["command"][0])
    if executable is None:
        raise ValueError(
            f"run {agent} trial {query['id']}: executable {plan['command'][0]!r} not found"
        )
    command = [str(Path(executable).resolve()), *plan["command"][1:]]
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "agent": agent,
                "binary_sha256": digest(Path(executable)),
                "model": getattr(args, agent + "_model"),
                "protocol": PROTOCOL,
                "max_turns": args.max_turns,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    started = datetime.now(timezone.utc).isoformat()
    transcript, errors = directory / "transcript.jsonl", directory / "stderr.txt"
    measured = run_measured(
        command,
        transcript,
        errors,
        env=env,
        timeout=args.budget_seconds,
        cwd=repository,
    )
    parsed = {"tokens_used": 0, "usage": [], "final_answer": "", "agent_failed": True}
    parse_error = None
    try:
        parsed = parse_agent_output(agent, transcript.read_text())
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        parse_error = f"parse {agent} transcript {transcript}: {error}"
    outcome = "timeout" if measured["timed_out"] else "completed"
    if outcome != "timeout" and (
        measured["exit_code"]
        or parsed["agent_failed"]
        or not parsed["final_answer"]
        or parsed["tokens_used"] > args.budget_tokens
    ):
        outcome = "failed"
    row = {
        "query_id": query["id"],
        "snapshot": query["snapshot"],
        "query_sha256": hashlib.sha256(query["query"].encode()).hexdigest(),
        "condition": condition,
        "agent": agent,
        "agent_fingerprint": fingerprint,
        "trial_id": uuid.uuid4().hex,
        "isolation_id": uuid.uuid4().hex,
        "budget_seconds": args.budget_seconds,
        "budget_tokens": args.budget_tokens,
        **measured,
        **parsed,
        "usage_complete": parse_error is None,
        "parse_error": parse_error,
        "outcome": outcome,
        "review_status": "pending_review",
        "evidence_review": {},
        "order": plan["order"],
        "started_at": started,
        "transcript_path": str(transcript.relative_to(args.output.resolve())),
        "stderr_path": str(errors.relative_to(args.output.resolve())),
        "command": command,
        "token_budget_enforcement": "prompt limit and posthoc usage check; no CLI hard total-token cap",
    }
    evaluate_trials(manifest, [row])
    return row


def blind_answers(manifest, trials):
    queries = {q["id"]: q for q in manifest}
    answers = []
    for row in trials:
        query = queries[row["query_id"]]
        answers.append(
            {
                "review_id": row["trial_id"],
                "query_id": query["id"],
                "repo": query["repo"],
                "snapshot": query["snapshot"],
                "prompt": query["query"],
                "answer_type": query["answer_type"],
                "gold_commits": query["gold_commits"],
                "answer": row["final_answer"],
            }
        )
    return sorted(answers, key=lambda row: row["review_id"])


def import_reviews(trials, reviews):
    indexed = {row["trial_id"]: row for row in trials}
    if len(indexed) != len(trials):
        raise ValueError("review import: duplicate trial_id in trials")
    by_id = {}
    for review in reviews:
        identifier = review.get("review_id")
        if identifier not in indexed or identifier in by_id:
            raise ValueError(f"review {identifier!r}: unknown or duplicate review_id")
        by_id[identifier] = review
        for field in ("passed", "blind_to_condition"):
            if type(review.get(field)) is not bool:
                raise ValueError(f"review {identifier}: {field} must be boolean")
        if (
            not isinstance(review.get("reviewer_id"), str)
            or not review["reviewer_id"].strip()
        ):
            raise ValueError(f"review {identifier}: reviewer_id is required")
        references, commits = review.get("references"), review.get("verified_commits")
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(ref, str) or not ref.strip() for ref in references)
        ):
            raise ValueError(
                f"review {identifier}: references must be nonempty strings"
            )
        if not isinstance(commits, list) or any(
            not isinstance(oid, str) or not OID.fullmatch(oid) for oid in commits
        ):
            raise ValueError(
                f"review {identifier}: verified_commits must be full Git IDs"
            )
        if (
            "absence_reviewed" in review
            and type(review["absence_reviewed"]) is not bool
        ):
            raise ValueError(f"review {identifier}: absence_reviewed must be boolean")
    merged = []
    for row in trials:
        review = by_id.get(row["trial_id"])
        merged.append(
            row
            | {
                "review_status": "reviewed",
                "evidence_review": {
                    k: v for k, v in review.items() if k != "review_id"
                },
            }
            if review
            else dict(row)
        )
    return merged


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def run(args):
    if args.output.exists():
        raise ValueError(f"trial output {args.output} exists; choose a new directory")
    if args.blind_export:
        manifest = read_jsonl(args.blind_export / "manifest.jsonl")
        trials = [
            row
            for path in sorted(args.blind_export.glob("*/trials.jsonl"))
            for row in read_jsonl(path)
        ]
        if not trials:
            raise ValueError(f"blind export {args.blind_export}: no agent trials found")
        answers = blind_answers(manifest, trials)
        args.output.mkdir(parents=True)
        write_jsonl(args.output / "answers.jsonl", answers)
        return 0
    if args.import_reviews:
        if not args.trials:
            raise ValueError("--import-reviews requires --trials FILE or RUN_DIR")
        paths = (
            sorted(args.trials.glob("*/trials.jsonl"))
            if args.trials.is_dir()
            else [args.trials]
        )
        if not paths:
            raise ValueError(f"review import {args.trials}: no trial files found")
        trials = [row for path in paths for row in read_jsonl(path)]
        rows = import_reviews(trials, read_jsonl(args.import_reviews))
        args.output.mkdir(parents=True)
        if args.trials.is_dir():
            manifest = read_jsonl(args.trials / "manifest.jsonl")
            write_jsonl(args.output / "manifest.jsonl", manifest)
            for path in paths:
                agent_rows = [row for row in rows if row["agent"] == path.parent.name]
                evaluate_trials(manifest, agent_rows)
                target = args.output / path.relative_to(args.trials)
                target.parent.mkdir(parents=True)
                write_jsonl(target, agent_rows)
        else:
            write_jsonl(args.output / "trials.jsonl", rows)
        return 0
    if args.tasks is None or args.clones is None:
        raise ValueError("trial planning requires --tasks and --clones")
    if "nalcos_git_rg" in args.conditions and (
        args.skill is None
        or not (args.skill / "SKILL.md").is_file()
        or args.binary is None
        or not args.binary.is_file()
    ):
        raise ValueError(
            "nalcos_git_rg requires --skill DIR containing SKILL.md and --binary FILE"
        )
    repositories = load_repositories(args.repos)
    manifest = task_manifest(read_jsonl(args.tasks), repositories)
    for key in {q["repo_key"] for q in manifest}:
        validate_clone(args.clones / key, repositories[key])
    planned = planned_trials(manifest, args)
    if not args.run:
        for plan in planned:
            print(json.dumps(plan))
        return 0
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "manifest.jsonl", manifest)
    write_jsonl(args.output / "plan.jsonl", planned)
    failed = False
    for plan in planned:
        row = collect_trial(plan, args, manifest)
        with (args.output / plan["agent"] / "trials.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    "event": "trial_finished",
                    "query_id": row["query_id"],
                    "agent": row["agent"],
                    "condition": row["condition"],
                    "outcome": row["outcome"],
                    "review_status": row["review_status"],
                }
            ),
            flush=True,
        )
        failed |= row["outcome"] != "completed"
    return 3 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        type=Path,
        help="JSONL: id, repo_key, prompt, gold_commits, answer_type",
    )
    parser.add_argument("--repos", type=Path, default=Path("benchmarks/repos.json"))
    parser.add_argument("--clones", type=Path)
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS
    )
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=AGENTS)
    parser.add_argument("--budget-seconds", type=float, default=300)
    parser.add_argument("--budget-tokens", type=int, default=20000)
    parser.add_argument(
        "--order", choices=["counterbalanced", "ab", "ba"], default="counterbalanced"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skill", type=Path)
    parser.add_argument(
        "--binary", type=Path, help="NaLCoS binary for the treatment PATH"
    )
    parser.add_argument("--claude", default="claude")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--claude-model")
    parser.add_argument("--codex-model")
    parser.add_argument("--max-turns", type=int, default=20)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument(
        "--dry-run", action="store_true", help="Print plans; also the default"
    )
    mode.add_argument("--blind-export", type=Path, metavar="RUN_DIR")
    mode.add_argument("--import-reviews", type=Path, metavar="REVIEWS_JSONL")
    parser.add_argument(
        "--trials",
        type=Path,
        help="Agent trials.jsonl or collection RUN_DIR for review import",
    )
    args = parser.parse_args()
    if (
        not math.isfinite(args.budget_seconds)
        or args.budget_seconds <= 0
        or args.budget_tokens < 1
        or args.max_turns < 1
    ):
        parser.error("time/token budgets and max-turns must be positive")
    if len(set(args.agents)) != len(args.agents) or len(set(args.conditions)) != len(
        args.conditions
    ):
        parser.error("agents and conditions must be unique")
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"agent trials failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
