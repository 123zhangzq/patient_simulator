"""Validate case discovery, filtering, and therapist-only prompt composition."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from core.cases import DialogueCase, compose_prompts, load_cases, select_cases


def case_data(case_id="test_case", category="recall"):
    return {
        "schema_version": 1,
        "id": case_id,
        "category": category,
        "title": "Offline reference title",
        "summary": "OFFLINE_SUMMARY_DO_NOT_INCLUDE",
        "therapist": {
            "context": "Review a previously practised task.",
            "objectives": ["Ask how the patient approached the task."],
            "materials": ["Target name: Example Name"],
            "opening_question": "How did you practise?",
            "conversation_plan": ["Invite an account, then ask about useful strategies."],
            "response_guidance": ["Follow the patient's reply rather than assuming success."],
            "completion_criteria": ["Summarise a patient-supported next step."],
        },
        "source": {
            "workbook": "OFFLINE_WORKBOOK_DO_NOT_INCLUDE.xlsx",
            "worksheet": "utterances",
            "excel_rows": [4, 30],
            "utt_ids": [3, 29],
            "dialog_ids": ["dg001"],
            "reference_note": "OFFLINE_PATIENT_REFERENCE_DO_NOT_INCLUDE",
        },
    }


class CaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_case(self, relative, data):
        path = self.root / relative / "case.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_recursive_discovery_ignores_reference_dialogue_and_preserves_path_order(self):
        self.write_case("z_category/second", case_data("second_case", "strategy"))
        first_path = self.write_case("a_category/first", case_data("first_case"))
        (first_path.parent / "reference_dialogue.json").write_text("invalid JSON", encoding="utf-8")
        (self.root / "notes.json").write_text("also invalid JSON", encoding="utf-8")
        cases = load_cases(self.root)
        self.assertEqual([case.case_id for case in cases], ["first_case", "second_case"])
        self.assertEqual(cases[0].path, first_path.resolve())
        self.assertEqual(load_cases(first_path)[0].case_id, "first_case")

    def test_duplicate_ids_and_path_like_ids_are_rejected(self):
        self.write_case("first", case_data("same_case"))
        second = self.write_case("second", case_data("same_case"))
        with self.assertRaisesRegex(ValueError, "Duplicate case id"):
            load_cases(self.root)
        for invalid_id in ("../escape", "folder/case", "..", "UPPER", "bad__id"):
            with self.subTest(case_id=invalid_id):
                data = case_data(invalid_id)
                second.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "id must use"):
                    load_cases(second)

    def test_selection_keeps_discovery_order_and_reports_invalid_filters(self):
        cases = tuple(DialogueCase(self.root / name, case_data(name, category)) for name, category in (
            ("case_a", "recall"), ("case_b", "strategy"), ("case_c", "recall"),
        ))
        selected = select_cases(cases, case_ids=["case_c", "case_a"])
        self.assertEqual([case.case_id for case in selected], ["case_a", "case_c"])
        self.assertEqual(select_cases(cases, category="strategy"), (cases[1],))
        with self.assertRaisesRegex(ValueError, "Unknown case ids"):
            select_cases(cases, case_ids=["missing"])
        with self.assertRaisesRegex(ValueError, "Unknown case category"):
            select_cases(cases, category="missing")
        with self.assertRaisesRegex(ValueError, "No cases match"):
            select_cases(cases, case_ids=["case_a"], category="strategy")

    def test_only_explicit_therapist_fields_enter_composed_prompts(self):
        data = case_data()
        original = copy.deepcopy(data)
        case = DialogueCase(self.root / "case.json", data)
        composed = compose_prompts(case, "BASE_THERAPIST", "Begin with: {opening_question}")
        self.assertEqual(set(composed), {"therapist", "opening"})
        self.assertEqual(composed["opening"], "Begin with: How did you practise?")
        self.assertIn("BASE_THERAPIST", composed["therapist"])
        self.assertIn("Target name: Example Name", composed["therapist"])
        self.assertIn("Follow the patient's reply", composed["therapist"])
        combined = "\n".join(composed.values())
        for offline in (data["summary"], data["source"]["workbook"], data["source"]["reference_note"]):
            self.assertNotIn(offline, combined)
        self.assertEqual(data, original)

    def test_opening_only_accepts_one_unformatted_opening_question_placeholder(self):
        case = DialogueCase(self.root / "case.json", case_data())
        for invalid in (
            "No placeholder", "{patient}", "{opening_question} {opening_question}",
            "{opening_question!r}", "{opening_question:>10}", "{opening_question.__class__}",
        ):
            with self.subTest(template=invalid):
                with self.assertRaises(ValueError):
                    compose_prompts(case, "BASE", invalid)

    def test_empty_directory_missing_path_and_malformed_case_have_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "No case.json"):
            load_cases(self.root)
        with self.assertRaises(FileNotFoundError):
            load_cases(self.root / "missing")
        path = self.write_case("bad", {"id": "incomplete"})
        with self.assertRaisesRegex(ValueError, "Invalid case file") as error:
            load_cases(path)
        self.assertIn(str(path), str(error.exception))


if __name__ == "__main__":
    unittest.main()
