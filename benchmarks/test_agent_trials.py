"""Agent collection and blind review with fake executables; no model calls."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_trials import (
    agent_command,
    blind_answers,
    checked_command,
    collect_trial,
    copy_directory,
    import_reviews,
    parse_agent_output,
    task_manifest,
    trial_environment,
)
from evaluate import read_jsonl
from test_index_matrix import BENCHMARKS, executable, local_repo, repo_manifest
from workflows import evaluate_trials


class AgentTrialTests(unittest.TestCase):
    def test_commands_use_real_noninteractive_flags_and_condition_tools(self):
        for condition in ("git_rg", "nalcos_git_rg"):
            command = agent_command(
                "claude", "/fake/claude", "prompt", condition, max_turns=7
            )
            self.assertEqual(command[1:5], ["-p", "prompt", "--output-format", "json"])
            self.assertEqual(command[command.index("--max-turns") + 1], "7")
            allowed = command[command.index("--allowed-tools") + 1]
            self.assertNotIn("Bash(git *)", allowed)
            for name in (
                "log",
                "show",
                "diff",
                "grep",
                "blame",
                "rev-list",
                "rev-parse",
                "merge-base",
                "cat-file",
                "ls-files",
                "ls-tree",
                "describe",
                "status",
            ):
                self.assertIn(f"Bash(git {name} *)", allowed)
            self.assertIn("Bash(git status)", allowed)
            self.assertEqual(command[command.index("--setting-sources") + 1], "project")
            self.assertEqual(
                command[command.index("--disallowed-tools") + 1],
                "Write,Edit,NotebookEdit,WebFetch,WebSearch",
            )
            self.assertIn("--verbose", command)
            self.assertEqual("Bash(nalcos *)" in allowed, condition == "nalcos_git_rg")
        command = agent_command(
            "codex", "/fake/codex", "prompt", "git_rg", model="pinned-model"
        )
        self.assertEqual(command[1:3], ["exec", "--json"])
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn('approval_policy="never"', command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(command[-2:], ["--model", "pinned-model"])

    def test_environment_keeps_only_matching_agent_credentials(self):
        planted = {
            "GITHUB_TOKEN": "secret",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "GH_TOKEN": "secret",
            "SSH_AUTH_SOCK": "/host/socket",
            "ANTHROPIC_API_KEY": "claude-only",
            "OPENAI_API_KEY": "codex-only",
            "CODEX_HOME": "/codex/home",
            "NALCOS_DATA_DIR": "/host/index",
            "HF_TOKEN": "secret",
            "HOME": "/home",
            "PATH": "/bin",
        }
        with patch.dict(os.environ, planted, clear=True):
            for agent, permitted in (
                ("claude", {"ANTHROPIC_API_KEY"}),
                ("codex", {"OPENAI_API_KEY", "CODEX_HOME"}),
            ):
                env = trial_environment(agent)
                self.assertEqual(env["HOME"], "/home")
                self.assertEqual(env["GIT_CONFIG_VALUE_0"], os.devnull)
                self.assertEqual(
                    set(env).intersection(planted), permitted | {"HOME", "PATH"}
                )

    def test_worktree_cleanup_runs_after_setup_and_agent_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "small"
            head, _ = local_repo(source)
            original_config = (source / ".git/config").read_bytes()
            for stage in ("add", "agent"):
                trial = work / stage
                plan = {
                    "directory": str(trial),
                    "agent": "claude",
                    "condition": "git_rg",
                    "query": {"repo_key": "small", "snapshot": head},
                }

                def run_command(command, operation, stage=stage):
                    checked_command(command, operation)
                    if stage == "add" and "add" in command:
                        raise OSError("fake preparation failure after registration")

                def fail_agent(plan, args, directory, repository):
                    for name in ("data", "runtime"):
                        (directory / name).mkdir()
                        (directory / name / "partial").write_text("trial state")
                    raise OSError("fake agent launch failure")

                with (
                    patch(
                        "agent_trials.checked_command", side_effect=run_command
                    ) as commands,
                    patch(
                        "agent_trials.collect_worktree_trial",
                        side_effect=fail_agent,
                    ),
                    self.assertRaisesRegex(OSError, "fake"),
                ):
                    collect_trial(plan, SimpleNamespace(clones=work), [])
                calls = [call.args[0] for call in commands.call_args_list]
                self.assertEqual(
                    [call[4] for call in calls], ["add", "remove", "prune"]
                )
                self.assertIn("--detach", calls[0])
                self.assertIn("--force", calls[1])
                self.assertFalse((trial / "repo").exists())
                self.assertFalse((trial / "data").exists())
                self.assertFalse((trial / "runtime").exists())
                self.assertFalse((source / ".git/worktrees").exists())
            self.assertEqual((source / ".git/config").read_bytes(), original_config)

    def test_cleanup_failure_is_reported_in_row_and_stderr_and_still_prunes(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            head, _ = local_repo(work / "small")
            trial = work / "trial"
            plan = {
                "directory": str(trial),
                "query": {"repo_key": "small", "snapshot": head},
            }

            def run_command(command, operation):
                checked_command(command, operation)
                if "remove" in command:
                    raise ValueError("fake worktree removal diagnostic")

            errors = io.StringIO()
            with (
                patch(
                    "agent_trials.checked_command", side_effect=run_command
                ) as commands,
                patch(
                    "agent_trials.collect_worktree_trial",
                    return_value={"outcome": "completed"},
                ),
                patch("agent_trials.evaluate_trials"),
                patch("agent_trials.sys.stderr", errors),
            ):
                row = collect_trial(plan, SimpleNamespace(clones=work), [])
            self.assertEqual(commands.call_args.args[0][-1], "prune")
            self.assertEqual(row["outcome"], "failed")
            self.assertEqual(
                row["cleanup_errors"], ["fake worktree removal diagnostic"]
            )
            self.assertIn(
                "trial cleanup failed: fake worktree removal diagnostic",
                errors.getvalue(),
            )

    def test_directory_copy_uses_platform_copy_on_write_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "index"
            source.mkdir()
            for platform, flag in (("darwin", "-cR"), ("linux", "-R")):
                with (
                    patch("agent_trials.sys.platform", platform),
                    patch("agent_trials.checked_command") as command,
                ):
                    copy_directory(source, work / "data")
                self.assertEqual(
                    command.call_args.args[0],
                    ["cp", flag, str(source.resolve()), str(work / "data")],
                )
            with (
                patch(
                    "agent_trials.checked_command", side_effect=ValueError("cp failed")
                ),
                self.assertRaisesRegex(ValueError, "cp failed"),
            ):
                copy_directory(source, work / "data")

    def test_usage_counts_claude_cache_tokens_and_codex_cache_once(self):
        parsed = parse_agent_output(
            "claude",
            json.dumps(
                {
                    "result": "answer",
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 3,
                        "cache_read_input_tokens": 5,
                        "cache_creation_input_tokens": 7,
                    },
                }
            ),
        )
        self.assertEqual(parsed["tokens_used"], 17)
        self.assertEqual(parsed["skill_used"], "unknown")
        events = [
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "answer"},
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 5,
                    "output_tokens": 3,
                },
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 20, "output_tokens": 4},
            },
        ]
        self.assertEqual(
            parse_agent_output("codex", "\n".join(json.dumps(e) for e in events))[
                "tokens_used"
            ],
            37,
        )
        for agent, text in (("claude", "{}"), ("codex", "{}"), ("codex", "not json")):
            with self.subTest(agent=agent), self.assertRaises(ValueError):
                parse_agent_output(agent, text)

    def test_skill_use_requires_transcript_evidence(self):
        result = {
            "type": "result",
            "result": "nalcos skill was used",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        invocation = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Skill", "input": {"skill": "nalcos"}}
                ]
            },
        }
        self.assertTrue(
            parse_agent_output("claude", json.dumps([invocation, result]))["skill_used"]
        )
        self.assertIs(
            parse_agent_output("claude", json.dumps([result]))["skill_used"], False
        )
        self.assertIs(
            parse_agent_output("claude", json.dumps(result | {"skills_used": []}))[
                "skill_used"
            ],
            False,
        )
        completed = {
            "type": "turn.completed",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        read = {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "cat .agents/skills/nalcos/SKILL.md",
                "exit_code": 0,
            },
        }
        for code, expected in ((0, True), (1, False)):
            read["item"]["exit_code"] = code
            parsed = parse_agent_output(
                "codex", "\n".join(map(json.dumps, [read, completed]))
            )
            self.assertIs(parsed["skill_used"], expected)

    def test_fake_collection_schema_counterbalancing_blind_export_and_review_import(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            clones = work / "clones"
            clones.mkdir()
            head, base = local_repo(clones / "small")
            repos = work / "repos.json"
            repo_manifest(repos, head, base, keys=("small",))
            tasks = work / "tasks.jsonl"
            tasks.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": f"q-{index}",
                            "repo_key": "small",
                            "prompt": "Find the behavior change",
                            "gold_commits": [head],
                            "answer_type": "find",
                        }
                    )
                    + "\n"
                    for index in range(2)
                )
            )
            skill = work / "skill"
            skill.mkdir()
            (skill / "SKILL.md").write_text(
                "---\nname: nalcos\ndescription: Search history\n---\nUse nalcos.\n"
            )
            binary = executable(work / "nalcos", 'print("fake nalcos")')
            index = work / "prebuilt-data"
            index.mkdir()
            (index / "seed").write_text("original index")
            (index / "common").write_text(str((clones / "small/.git").resolve()))
            indexes = work / "indexes.json"
            indexes.write_text(json.dumps({"small": "prebuilt-data"}))
            runtime = work / "prebuilt-runtime"
            runtime.mkdir()
            (runtime / "device").write_text("calibrated")
            original_config = (clones / "small/.git/config").read_bytes()
            agent_source = """
                import json, os, subprocess, sys
                from pathlib import Path
                args = sys.argv[1:]
                treatment = (Path.cwd() / ".agents/skills/nalcos/SKILL.md").is_file()
                assert (Path.cwd() / ".claude/skills/nalcos/SKILL.md").is_file() == treatment
                probe = subprocess.run(["nalcos", "--version"], capture_output=True)
                assert probe.returncode == (0 if treatment else 127)
                assert os.environ["GIT_NO_LAZY_FETCH"] == "1"
                assert (Path.cwd() / ".git").is_file()
                for key in ("GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "GH_TOKEN", "SSH_AUTH_SOCK", "HF_TOKEN"):
                    assert key not in os.environ, key
                assert os.environ["HF_HOME"] == "/shared/hf"
                assert os.environ["HF_HUB_CACHE"] == "/shared/hf/hub"
                assert subprocess.check_output(["git", "config", "core.hooksPath"], text=True).strip() == os.devnull
                for key, value in {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "PAGER": "cat"}.items():
                    assert os.environ[key] == value
                data, runtime = Path(os.environ["NALCOS_DATA_DIR"]), Path(os.environ["NALCOS_RUNTIME_CACHE"])
                assert data == Path.cwd().parent / "data"
                assert runtime == Path.cwd().parent / "runtime"
                if treatment:
                    assert (data / "seed").read_text() == "original index"
                    common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()
                    assert str(Path(common).resolve()) == (data / "common").read_text()
                    (data / "seed").write_text("trial writes")
                    assert (runtime / "device").read_text() == "calibrated"
                    (runtime / "device").write_text("trial calibration")
                else:
                    assert not list(data.iterdir()) and not list(runtime.iterdir())
                if "exec" in args:
                    assert "--json" in args and "--ephemeral" in args
                    if treatment:
                        print(json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "cat .agents/skills/nalcos/SKILL.md", "exit_code": 0}}))
                    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Found the change with a supporting hunk"}}))
                    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 5, "output_tokens": 3}}))
                else:
                    assert "--max-turns" in args
                    assert json.loads((Path.cwd() / ".claude/settings.json").read_text()) == {}
                    assert (Path.cwd() / ".claude/skills").is_dir()
                    allowed = args[args.index("--allowed-tools") + 1]
                    assert ("Bash(nalcos *)" in allowed) == treatment
                    assert "Bash(git *)" not in allowed
                    messages = []
                    if treatment:
                        messages.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Skill", "input": {"skill": "nalcos"}}]}})
                    messages.append({"type": "result", "result": "Found the change with a supporting hunk", "usage": {"input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 2}})
                    print(json.dumps(messages))
            """
            claude = executable(work / "fake-claude", agent_source)
            codex = executable(work / "fake-codex", agent_source)
            output = work / "trials"
            command = [
                sys.executable,
                str(BENCHMARKS / "agent_trials.py"),
                "--tasks",
                str(tasks),
                "--repos",
                str(repos),
                "--clones",
                str(clones),
                "--skill",
                str(skill),
                "--binary",
                str(binary),
                "--claude",
                str(claude),
                "--codex",
                str(codex),
                "--budget-seconds",
                "30",
                "--budget-tokens",
                "100",
                "--output",
                str(output),
            ]
            planned = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            self.assertEqual(len(planned.stdout.splitlines()), 8)
            self.assertFalse(output.exists())
            missing_indexes = subprocess.run(
                [*command, "--run"], capture_output=True, text=True, check=False
            )
            self.assertEqual(missing_indexes.returncode, 2)
            self.assertIn(
                "nalcos_git_rg requires --indexes PATH", missing_indexes.stderr
            )
            self.assertFalse(output.exists())
            with patch.dict(
                os.environ,
                {
                    "GITHUB_TOKEN": "secret",
                    "AWS_SECRET_ACCESS_KEY": "secret",
                    "HF_HOME": "/shared/hf",
                    "HF_HUB_CACHE": "/shared/hf/hub",
                },
            ):
                collected = subprocess.run(
                    [
                        *command,
                        "--indexes",
                        str(indexes),
                        "--runtime-cache",
                        str(runtime),
                        "--run",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            diagnostics = collected.stdout + collected.stderr
            if collected.returncode:
                diagnostics += "\n" + "\n".join(
                    path.read_text() for path in output.glob("*/trial-*/*.txt")
                )
                diagnostics += "\n" + "\n".join(
                    path.read_text() for path in output.glob("*/trials.jsonl")
                )
            self.assertEqual(collected.returncode, 0, diagnostics)
            manifest = read_jsonl(output / "manifest.jsonl")
            all_rows = []
            for agent in ("claude", "codex"):
                rows = read_jsonl(output / agent / "trials.jsonl")
                self.assertEqual(len(rows), 4)
                self.assertTrue(
                    all(
                        r["review_status"] == "pending_review"
                        and r["outcome"] == "completed"
                        for r in rows
                    )
                )
                self.assertTrue(
                    all(
                        r["tokens_used"] == (15 if agent == "claude" else 13)
                        for r in rows
                    )
                )
                self.assertEqual(
                    {r["order"] for r in rows if r["condition"] == "nalcos_git_rg"},
                    {0, 1},
                )
                self.assertTrue(
                    all((output / r["transcript_path"]).is_file() for r in rows)
                )
                result = evaluate_trials(manifest, rows, split="development")
                self.assertEqual(result["paired_questions"], 2)
                self.assertEqual(result["incomparable_pairs"], 0)
                self.assertEqual(result["conditions"]["git_rg"]["successes"], 0)
                all_rows.extend(rows)
            self.assertEqual(len({r["isolation_id"] for r in all_rows}), 8)
            for row in all_rows:
                self.assertEqual(row["skill_used"], row["condition"] == "nalcos_git_rg")
                self.assertEqual(row["cleanup_errors"], [])
                self.assertEqual(
                    row["data_size_bytes"] > 0, row["condition"] == "nalcos_git_rg"
                )
                trial_dir = (output / row["transcript_path"]).parent
                for name in ("repo", "data", "runtime"):
                    self.assertFalse((trial_dir / name).exists())
            self.assertEqual((index / "seed").read_text(), "original index")
            self.assertEqual({p.name for p in index.iterdir()}, {"seed", "common"})
            self.assertEqual((runtime / "device").read_text(), "calibrated")
            self.assertEqual(
                (clones / "small/.git/config").read_bytes(), original_config
            )
            self.assertFalse((clones / "small/.git/worktrees").exists())
            self.assertFalse((clones / "small" / ".agents").exists())
            blind = work / "blind"
            exported = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARKS / "agent_trials.py"),
                    "--blind-export",
                    str(output),
                    "--output",
                    str(blind),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(exported.returncode, 0, exported.stderr)
            answers = read_jsonl(blind / "answers.jsonl")
            self.assertEqual(len(answers), 8)
            self.assertEqual(list(blind.iterdir()), [blind / "answers.jsonl"])
            self.assertEqual(answers, blind_answers(manifest, all_rows))
            for answer in answers:
                self.assertFalse(
                    set(answer).intersection(
                        {
                            "condition",
                            "agent",
                            "order",
                            "transcript_path",
                            "elapsed_seconds",
                        }
                    )
                )
            reviews = work / "reviews.jsonl"
            reviews.write_text(
                "".join(
                    json.dumps(
                        {
                            "review_id": a["review_id"],
                            "passed": True,
                            "blind_to_condition": True,
                            "reviewer_id": "independent",
                            "references": ["review.txt"],
                            "verified_commits": [head],
                        }
                    )
                    + "\n"
                    for a in answers
                )
            )
            reviewed = work / "reviewed"
            imported = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARKS / "agent_trials.py"),
                    "--import-reviews",
                    str(reviews),
                    "--trials",
                    str(output),
                    "--output",
                    str(reviewed),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(imported.returncode, 0, imported.stderr)
            for agent in ("claude", "codex"):
                result = evaluate_trials(
                    manifest,
                    read_jsonl(reviewed / agent / "trials.jsonl"),
                    split="development",
                )
                self.assertEqual(result["conditions"]["nalcos_git_rg"]["successes"], 2)

    def test_review_rejects_unknown_or_duplicate_ids_and_task_labels(self):
        row = {
            "trial_id": "one",
            "outcome": "failed",
            "review_status": "pending_review",
        }
        review = {
            "review_id": "one",
            "passed": True,
            "blind_to_condition": True,
            "reviewer_id": "judge",
            "references": ["review"],
            "verified_commits": [],
        }
        merged = import_reviews([row], [review])[0]
        self.assertEqual(merged["outcome"], "failed")
        self.assertEqual(row["review_status"], "pending_review")
        for reviews in (
            [review, review],
            [review | {"review_id": "unknown"}],
            [review | {"verified_commits": ["abc"]}],
        ):
            with self.assertRaises(ValueError):
                import_reviews([row], reviews)
        repo = {"r": {"url": "local", "snapshot": "a" * 40, "split": "heldout"}}
        task = {
            "id": "q",
            "repo_key": "r",
            "prompt": "Was this ever implemented?",
            "answer_type": "absence",
            "gold_commits": [],
        }
        self.assertFalse(task_manifest([task], repo)[0]["answerable"])
        with self.assertRaisesRegex(ValueError, "gold"):
            task_manifest([task | {"gold_commits": ["b" * 40]}], repo)

    def test_token_overrun_is_failed_and_wall_timeout_is_workflow_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            clones = work / "clones"
            clones.mkdir()
            head, base = local_repo(clones / "small")
            repos = work / "repos.json"
            repo_manifest(repos, head, base, keys=("small",))
            tasks = work / "tasks.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "id": "q",
                        "repo_key": "small",
                        "prompt": "Find change",
                        "gold_commits": [head],
                        "answer_type": "explain",
                    }
                )
                + "\n"
            )
            fake = executable(
                work / "fake",
                """
                import json, time
                print(json.dumps({"result": "answer", "usage": {"input_tokens": 100, "output_tokens": 1}}), flush=True)
            """,
            )
            base_command = [
                sys.executable,
                str(BENCHMARKS / "agent_trials.py"),
                "--tasks",
                str(tasks),
                "--repos",
                str(repos),
                "--clones",
                str(clones),
                "--conditions",
                "git_rg",
                "--agents",
                "claude",
                "--claude",
                str(fake),
                "--budget-tokens",
                "10",
                "--run",
            ]
            overrun = work / "overrun"
            result = subprocess.run(
                [*base_command, "--output", str(overrun)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertEqual(
                read_jsonl(overrun / "claude/trials.jsonl")[0]["outcome"], "failed"
            )
            executable(fake, "import time\ntime.sleep(20)")
            timeout = work / "timeout"
            result = subprocess.run(
                [*base_command, "--budget-seconds", "0.1", "--output", str(timeout)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 3, result.stderr)
            row = read_jsonl(timeout / "claude/trials.jsonl")[0]
            self.assertEqual(row["outcome"], "timeout")
            self.assertFalse(row["usage_complete"])
            self.assertIn("transcript", row["parse_error"])


if __name__ == "__main__":
    unittest.main()
