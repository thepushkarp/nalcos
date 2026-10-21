"""Agent collection and blind review with fake executables; no model calls."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent_trials import (
    agent_command,
    blind_answers,
    import_reviews,
    parse_agent_output,
    task_manifest,
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
            self.assertIn("Bash(git *)", allowed)
            self.assertEqual("Bash(nalcos *)" in allowed, condition == "nalcos_git_rg")
        command = agent_command(
            "codex", "/fake/codex", "prompt", "git_rg", model="pinned-model"
        )
        self.assertEqual(command[1:3], ["exec", "--json"])
        self.assertIn("--ephemeral", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(command[-2:], ["--model", "pinned-model"])

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
            agent_source = """
                import json, os, subprocess, sys
                from pathlib import Path
                args = sys.argv[1:]
                treatment = (Path.cwd() / ".agents/skills/nalcos/SKILL.md").is_file()
                assert (Path.cwd() / ".claude/skills/nalcos/SKILL.md").is_file() == treatment
                probe = subprocess.run(["nalcos", "--version"], capture_output=True)
                assert probe.returncode == (0 if treatment else 127)
                assert os.environ["GIT_NO_LAZY_FETCH"] == "1"
                assert not (Path.cwd() / ".git/objects/info/alternates").exists()
                if "exec" in args:
                    assert "--json" in args and "--ephemeral" in args
                    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Found the change with a supporting hunk"}}))
                    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 5, "output_tokens": 3}}))
                else:
                    assert "--max-turns" in args
                    allowed = args[args.index("--allowed-tools") + 1]
                    assert ("Bash(nalcos *)" in allowed) == treatment
                    print(json.dumps({"result": "Found the change with a supporting hunk", "usage": {"input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 2}}))
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
            collected = subprocess.run(
                [*command, "--run"], capture_output=True, text=True, check=False
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
