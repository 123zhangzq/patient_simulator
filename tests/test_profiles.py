"""Validate the shared profile boundary without reading clinical workbooks."""

import copy
import json
from pathlib import Path
import sys
import unittest


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from core.profiles import (
    patient_information,
    profile_fingerprint,
    render_patient_profile,
    validate_patient_profile,
)
from helpers import profile_data


class ProfileTests(unittest.TestCase):
    def test_valid_profile_is_not_changed_by_validation_or_rendering(self):
        profile = profile_data()
        original = copy.deepcopy(profile)
        validate_patient_profile(profile)
        information = patient_information(profile)
        rendered = render_patient_profile(profile)
        self.assertIsInstance(information, dict)
        self.assertIsInstance(rendered, str)
        self.assertTrue(rendered.strip())
        self.assertEqual(profile, original)

    def test_observed_information_is_available_but_provenance_is_not(self):
        profile = profile_data()
        profile["known_facts"] = [{
            "id": "daily_activity",
            "text": "I keep a green notebook for shopping.",
            "source": "PRIVATE_SOURCE_WORKBOOK.xlsx / Z999",
        }]
        profile["provenance"] = {"private_note": "PRIVATE_PROVENANCE_SENTINEL"}
        profile["initial_memories"] = [{
            "id": "initial_exposure", "text": "The practice token was vensidor.",
            "source": "PRIVATE_INITIAL_SOURCE.xlsx / Q888", "cue_tags": ["practice token"],
        }]
        information = patient_information(profile)
        combined = json.dumps(information, ensure_ascii=False) + render_patient_profile(profile)
        self.assertIn("I keep a green notebook for shopping.", combined)
        self.assertEqual(information["assessments"], profile["assessments"])
        for sentinel in ("PRIVATE_SOURCE_WORKBOOK.xlsx", "Z999", "PRIVATE_PROVENANCE_SENTINEL",
                         "PRIVATE_INITIAL_SOURCE.xlsx", "Q888"):
            self.assertNotIn(sentinel, combined)
        self.assertNotIn("provenance", information)
        self.assertNotIn("source", information["known_facts"][0])
        self.assertNotIn("source", information["initial_memories"][0])

    def test_unknown_measurements_remain_unknown(self):
        profile = profile_data()
        for group in profile["assessments"].values():
            for key in group:
                group[key] = None
        profile["assessment_date"] = None
        profile["background"] = {"age": None, "language": None, "education_years": None}
        validate_patient_profile(profile)
        information = patient_information(profile)
        self.assertEqual(information["assessments"], profile["assessments"])
        self.assertEqual(information["background"], profile["background"])

    def test_caregiver_and_future_measurement_are_not_accepted_as_baseline_patient_data(self):
        profile = profile_data()
        profile["role"] = "caregiver"
        with self.assertRaisesRegex(ValueError, "role|caregiver"):
            validate_patient_profile(profile)
        profile = profile_data()
        profile["assessment_date"] = "2023-01-01"
        with self.assertRaisesRegex(ValueError, "assessment_date|simulation_date"):
            validate_patient_profile(profile)

    def test_method_visible_information_does_not_alias_the_source_profile(self):
        profile = profile_data()
        original = copy.deepcopy(profile)
        information = patient_information(profile)
        information["background"]["test_mutation"] = "must remain local"
        information["assessments"]["moca"]["total"] = 0
        self.assertEqual(profile, original)

    def test_missing_required_fields_and_unexpected_answer_fields_fail_clearly(self):
        for missing in ("schema_version", "participant_id", "role", "background", "assessments"):
            with self.subTest(missing=missing):
                profile = profile_data()
                del profile[missing]
                with self.assertRaisesRegex(ValueError, missing):
                    validate_patient_profile(profile)
        profile = profile_data()
        profile["therapist_answers"] = {"target_name": "HIDDEN_TARGET"}
        with self.assertRaisesRegex(ValueError, "therapist_answers|unknown|Unknown"):
            validate_patient_profile(profile)

    def test_invalid_assessment_values_do_not_silently_become_patient_traits(self):
        for field, value in (
            ("total", -1), ("total", 31), ("total", True),
            ("total", "23"), ("total", float("nan")),
            ("attention", 7), ("delayed_recall", 6),
        ):
            with self.subTest(field=field, value=value):
                profile = profile_data()
                profile["assessments"]["moca"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_patient_profile(profile)

    def test_profile_containers_and_fact_text_must_have_the_expected_types(self):
        for field, value in (
            ("background", []), ("assessments", []),
            ("known_facts", "not a list"), ("initial_memories", "not a list"),
        ):
            with self.subTest(field=field):
                profile = profile_data()
                profile[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_patient_profile(profile)
        profile = profile_data()
        profile["known_facts"] = [{"id": "invalid_fact", "text": "", "source": "authored fixture"}]
        with self.assertRaisesRegex(ValueError, "text|known_facts"):
            validate_patient_profile(profile)

    def test_fingerprint_ignores_dictionary_order_but_detects_profile_changes(self):
        profile = profile_data()
        reordered = {key: profile[key] for key in reversed(profile)}
        self.assertEqual(profile_fingerprint(profile), profile_fingerprint(reordered))
        changed = copy.deepcopy(profile)
        changed["assessments"]["moca"]["total"] = (
            0 if profile["assessments"]["moca"]["total"] else 1
        )
        self.assertNotEqual(profile_fingerprint(profile), profile_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
