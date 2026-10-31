"""Preview entry-point checks in a subprocess with site packages disabled."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_cases import case_data


CODE_DIR = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cases_dir = self.root / "cases"
        for folder, case_id, category in (
            ("a", "case_a", "recall"), ("b", "case_b", "strategy"),
        ):
            case_dir = self.cases_dir / folder
            case_dir.mkdir(parents=True)
            data = case_data(case_id, category)
            if case_id == "case_a":
                data["rounds"] = 3
            (case_dir / "case.json").write_text(json.dumps(data), encoding="utf-8")
            (case_dir / "reference_dialogue.json").write_text("not parsed by the simulator", encoding="utf-8")
        config = json.loads((CODE_DIR / "config.json").read_text(encoding="utf-8"))
        config["rounds"] = 10
        config["therapist_prompt"] = "therapist.txt"
        config["opening_prompt"] = "opening.txt"
        config["patient_prompt"] = "patient.txt"
        config["patient_agents"] = {
            "response_prompt": str(CODE_DIR / "prompts/patient_agents/response.txt"),
            "policy_path": str(CODE_DIR / "patient_agents/policy.json"),
        }
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        (self.root / "therapist.txt").write_text("Generic therapist instructions.", encoding="utf-8")
        (self.root / "opening.txt").write_text("First question: {opening_question}", encoding="utf-8")
        (self.root / "patient.txt").write_text("PATIENT_PROMPT_PRIVATE_SENTINEL", encoding="utf-8")

    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, "-S", str(CODE_DIR / "run_dialogue.py"),
             "--config", str(self.config_path), "--cases-dir", str(self.cases_dir), *args],
            cwd=self.root, text=True, capture_output=True, check=False,
        )

    def test_dry_run_composes_from_another_working_directory_without_gpu_runtime(self):
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        preview = json.loads(result.stdout)
        self.assertEqual([case["id"] for case in preview["cases"]], ["case_a", "case_b"])
        self.assertEqual([case["rounds"] for case in preview["cases"]], [3, 10])
        self.assertEqual(preview["patient_prompt_file"], str((self.root / "patient.txt").resolve()))
        self.assertNotIn("PATIENT_PROMPT_PRIVATE_SENTINEL", result.stdout)
        self.assertNotIn("OFFLINE_SUMMARY_DO_NOT_INCLUDE", result.stdout)
        self.assertFalse((self.root / "outputs").exists())

    def test_repeated_case_filter_round_and_seed_overrides(self):
        result = self.invoke("--dry-run", "--case", "case_b", "--case", "case_a", "--rounds", "5", "--seed", "17")
        self.assertEqual(result.returncode, 0, result.stderr)
        preview = json.loads(result.stdout)
        self.assertEqual([case["id"] for case in preview["cases"]], ["case_a", "case_b"])
        self.assertEqual([case["rounds"] for case in preview["cases"]], [5, 5])
        self.assertEqual(preview["seed"], 17)

    def test_proposed_preview_and_ablations_share_the_baseline_case_profile(self):
        baseline = self.invoke("--dry-run", "--show-patient")
        proposed = self.invoke("--patient", "patient_agents", "--dry-run", "--show-patient", "--no-attention", "--no-memory")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(proposed.returncode, 0, proposed.stderr)
        baseline_cases = json.loads(baseline.stdout)["cases"]
        proposed_cases = json.loads(proposed.stdout)["cases"]
        for left, right in zip(baseline_cases, proposed_cases):
            self.assertEqual(left["patient_profile_sha256"], right["patient_profile_sha256"])
            self.assertEqual(left["patient_information"], right["patient_information"])
            self.assertFalse(right["patient_setup"]["options"]["attention_enabled"])
            self.assertFalse(right["patient_setup"]["options"]["memory_enabled"])
            patient_only = json.dumps(right["patient_setup"])
            self.assertNotIn("Target name: Example Name", patient_only)
            self.assertNotIn("PRIVATE_SOURCE_SENTINEL", patient_only)
        self.assertNotIn("PATIENT_PROMPT_PRIVATE_SENTINEL", proposed.stdout)
        self.assertFalse((self.root / "outputs").exists())

    def test_invalid_ablation_and_private_preview_flags_fail_before_model_import(self):
        for args in (("--no-memory", "--dry-run"), ("--show-patient",)):
            result = self.invoke(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("require", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_default_case_directory_works_outside_the_code_directory(self):
        result = subprocess.run(
            [sys.executable, "-S", str(CODE_DIR / "run_dialogue.py"),
             "--config", str(self.config_path), "--list-cases"],
            cwd=self.root, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for case_id in ("name_recall", "work_task_compensation", "name_practice_review"):
            self.assertIn(case_id, result.stdout)

    def test_listing_filters_and_unknown_selection_has_actionable_error(self):
        result = self.invoke("--list-cases", "--category", "strategy")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("case_b", result.stdout)
        self.assertNotIn("case_a", result.stdout)
        result = self.invoke("--dry-run", "--case", "missing")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown case ids: missing", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
