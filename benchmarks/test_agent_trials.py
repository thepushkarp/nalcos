"""Agent collection and blind review with fake executables; no model calls."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trials import (
    parse_agent_output,
    real_git,
    write_git_guard,
)
from evaluate import read_jsonl
from test_index_matrix import BENCHMARKS, executable, local_repo, repo_manifest
from workflows import evaluate_trials


def collection_inputs(work):
    clones = work / "clones"
    clones.mkdir()
    head, base = local_repo(clones / "small")
    repos, tasks = work / "repos.json", work / "tasks.jsonl"
    repo_manifest(repos, head, base, keys=("small",))
    tasks.write_text(
        json.dumps(
            {
                "id": "q",
                "repo_key": "small",
                "prompt": "Find change",
                "gold_commits": [head],
                "answer_type": "find",
            }
        )
        + "\n"
    )
    return head, [
        sys.executable,
        str(BENCHMARKS / "agent_trials.py"),
        "--tasks",
        str(tasks),
        "--repos",
        str(repos),
        "--clones",
        str(clones),
        "--budget-tokens",
        "100",
    ]


class AgentTrialTests(unittest.TestCase):
    def test_git_guard_rejects_writes_and_exec_and_allows_history_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            head, _ = local_repo(work / "repo")
            guard = write_git_guard(work, real_git())

            def run(*args):
                return subprocess.run(
                    [str(guard), *args],
                    cwd=work / "repo",
                    capture_output=True,
                    text=True,
                    check=False,
                )

            self.assertEqual(run("rev-parse", "HEAD").stdout.strip(), head)
            self.assertEqual(run("log", "-1", "--format=%H").stdout.strip(), head)
            self.assertIn("two", run("show", "HEAD:file.txt").stdout)
            target = work / "written"
            for args in (
                ("grep", "-Otouch", "x"),
                ("-c", "core.pager=touch", "log"),
                ("log", f"--output={target}"),
                ("log", "--output", str(target)),
                ("diff", "--ext-diff"),
                ("show", "--textconv"),
                ("log", "--exec-path=/tmp"),
                ("config", "alias.x", "!touch"),
            ):
                with self.subTest(args=args):
                    self.assertEqual(run(*args).returncode, 126)
            self.assertFalse(target.exists())

    def test_usage_counts_claude_cache_tokens_and_codex_cache_once(self):
        result = {
            "type": "result",
            "result": "answer",
            "usage": {
                "input_tokens": 2,
                "output_tokens": 3,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 7,
            },
        }
        self.assertEqual(
            parse_agent_output("claude", json.dumps([result]))["tokens_used"], 17
        )
        events = [
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
            parse_agent_output("codex", "\n".join(map(json.dumps, events)))[
                "tokens_used"
            ],
            37,
        )
        for agent, text in (("claude", "{}"), ("codex", "{}"), ("codex", "not json")):
            with self.subTest(agent=agent), self.assertRaises(ValueError):
                parse_agent_output(agent, text)
        for usage in (
            {"input_tokens": -1, "output_tokens": 1},
            result["usage"] | {"cache_read_input_tokens": -1},
        ):
            with self.assertRaisesRegex(ValueError, "token usage"):
                parse_agent_output("claude", json.dumps(result | {"usage": usage}))

    def test_skill_use_requires_invocation_or_successful_read(self):
        result = {
            "type": "result",
            "result": "I used the nalcos skill",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "skills_used": ["nalcos"],
        }
        for name, inputs, expected in (
            ("Skill", {"skill": "nalcos"}, True),
            ("Read", {"file_path": "/repo/.claude/skills/nalcos/SKILL.md"}, True),
            ("Skill", {"skill": "other"}, False),
        ):
            event = {
                "message": {
                    "content": [{"type": "tool_use", "name": name, "input": inputs}]
                }
            }
            self.assertIs(
                parse_agent_output("claude", json.dumps([event, result]))["skill_used"],
                expected,
            )
        available = {"type": "system", "skills": ["nalcos"]}
        self.assertIs(
            parse_agent_output("claude", json.dumps([available, result]))["skill_used"],
            False,
        )
        self.assertEqual(
            parse_agent_output("claude", json.dumps(result))["skill_used"], "unknown"
        )
        completed = {
            "type": "turn.completed",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        for command, code, expected in (
            ("cat .agents/skills/nalcos/SKILL.md", 0, True),
            ("/bin/zsh -lc 'head .agents/skills/nalcos/SKILL.md'", 0, True),
            ("sed -n '1,80p' .agents/skills/nalcos/SKILL.md", 0, True),
            ("cat .agents/skills/nalcos/SKILL.md", 1, False),
            ("echo cat .agents/skills/nalcos/SKILL.md", 0, False),
            ("cat other; echo .agents/skills/nalcos/SKILL.md", 0, False),
            ("cat other # .agents/skills/nalcos/SKILL.md", 0, False),
            ("cat --help .agents/skills/nalcos/SKILL.md", 0, False),
            ("cat .agents/skills/other/SKILL.md", 0, False),
        ):
            event = {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": command,
                    "exit_code": code,
                },
            }
            with self.subTest(command=command, code=code):
                self.assertIs(
                    parse_agent_output(
                        "codex", "\n".join(map(json.dumps, [event, completed]))
                    )["skill_used"],
                    expected,
                )

    def test_collection_isolates_indexes_and_secrets_and_exports_reviewable_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            head, command = collection_inputs(work)
            skill = work / "nalcos"
            skill.mkdir()
            (skill / "SKILL.md").write_text("Use nalcos.\n")
            index, runtime = work / "index", work / "runtime"
            index.mkdir()
            runtime.mkdir()
            (index / "seed").write_text("original index")
            (runtime / "seed").write_text("original runtime")
            indexes = work / "indexes.json"
            indexes.write_text(json.dumps({"small": "index"}))
            binary = executable(work / "binary", 'print("fake nalcos")')
            agent_source = """
                import json, os, subprocess, sys
                from pathlib import Path
                treatment = Path(".agents/skills/nalcos/SKILL.md").is_file()
                for key in ("GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "SSH_AUTH_SOCK"):
                    assert key not in os.environ, key
                data = Path(os.environ["NALCOS_DATA_DIR"])
                runtime = Path(os.environ["NALCOS_RUNTIME_CACHE"])
                if treatment:
                    assert (data / "seed").read_text() == "original index"
                    assert (runtime / "seed").read_text() == "original runtime"
                    (data / "seed").write_text("agent write")
                    (runtime / "seed").write_text("agent write")
                else:
                    assert not list(data.iterdir()) and not list(runtime.iterdir())
                    assert subprocess.run(["nalcos"], capture_output=True).returncode == 127
                if sys.argv[0].endswith("codex"):
                    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Evidence answer"}}))
                    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 3}}))
                else:
                    print(json.dumps([{"type": "result", "result": "Evidence answer", "usage": {"input_tokens": 10, "output_tokens": 3}}]))
            """
            claude = executable(work / "fake-claude", agent_source)
            codex = executable(work / "fake-codex", agent_source)
            output = work / "out"
            command += [
                "--skill",
                str(skill),
                "--binary",
                str(binary),
                "--claude",
                str(claude),
                "--codex",
                str(codex),
                "--output",
                str(output),
            ]
            planned = subprocess.run(
                command + ["--dry-run"], capture_output=True, text=True, check=False
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            self.assertFalse(output.exists())
            for condition in ("nalcos_git_rg", "nalcos_hint"):
                missing = subprocess.run(
                    command + ["--run", "--conditions", condition],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(missing.returncode, 2)
                self.assertIn("requires --indexes", missing.stderr)
                self.assertIn("repo_key", missing.stderr)
                self.assertFalse(output.exists())
            planted = {
                key: "planted-secret"
                for key in (
                    "GITHUB_TOKEN",
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY",
                    "AWS_SESSION_TOKEN",
                    "SSH_AUTH_SOCK",
                )
            }
            with patch.dict(os.environ, planted):
                collected = subprocess.run(
                    command
                    + [
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
            diagnostics = collected.stderr + "\n".join(
                p.read_text() for p in output.glob("*/trial-*/*.txt")
            )
            self.assertEqual(collected.returncode, 0, diagnostics)
            self.assertEqual((index / "seed").read_text(), "original index")
            self.assertEqual((runtime / "seed").read_text(), "original runtime")
            manifest = read_jsonl(output / "manifest.jsonl")
            for agent in ("claude", "codex"):
                rows = read_jsonl(output / agent / "trials.jsonl")
                result = evaluate_trials(manifest, rows, split="development")
                self.assertEqual(result["paired_questions"], 1)
                self.assertEqual(result["incomparable_pairs"], 0)
                self.assertEqual(result["conditions"]["git_rg"]["successes"], 0)
                for row in rows:
                    self.assertEqual(row["outcome"], "completed")
                    for name in ("repo", "data", "runtime"):
                        self.assertFalse(
                            ((output / row["transcript_path"]).parent / name).exists()
                        )
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
            self.assertEqual(len(answers), 4)
            for answer in answers:
                self.assertEqual(answer["answer"], "Evidence answer")
                for hidden in (
                    "condition",
                    "agent",
                    "order",
                    "transcript_path",
                    "elapsed_seconds",
                ):
                    self.assertNotIn(hidden, answer)
            reviews = work / "reviews.jsonl"
            reviews.write_text(
                "".join(
                    json.dumps(
                        {
                            "review_id": a["review_id"],
                            "passed": True,
                            "blind_to_condition": True,
                            "reviewer_id": "judge",
                            "references": ["review"],
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
                self.assertEqual(result["conditions"]["nalcos_git_rg"]["successes"], 1)

    def test_token_overrun_and_timeout_fail_and_remain_workflow_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            _, command = collection_inputs(work)
            fake = executable(
                work / "fake",
                'import json\nprint(json.dumps({"result": "answer", "usage": {"input_tokens": 100, "output_tokens": 1}}), flush=True)',
            )
            command += [
                "--conditions",
                "git_rg",
                "--agents",
                "claude",
                "--claude",
                str(fake),
                "--run",
            ]
            for outcome in ("failed", "timeout"):
                output = work / outcome
                if outcome == "timeout":
                    executable(fake, "import time\ntime.sleep(20)")
                result = subprocess.run(
                    command
                    + [
                        "--budget-seconds",
                        "0.2" if outcome == "timeout" else "30",
                        "--output",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 3, result.stderr)
                row = read_jsonl(output / "claude/trials.jsonl")[0]
                self.assertEqual(row["outcome"], outcome)
                evaluate_trials(read_jsonl(output / "manifest.jsonl"), [row])
                self.assertFalse((output / "claude/trial-000-0/repo").exists())
                if outcome == "timeout":
                    self.assertFalse(row["usage_complete"])
                    self.assertIn("transcript", row["parse_error"])


if __name__ == "__main__":
    unittest.main()
