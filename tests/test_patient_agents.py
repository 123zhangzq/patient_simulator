"""Behavioral checks of patient-visible context using a recording model.

Nonce facts let these tests distinguish actual information access from plausible
language. No source workbooks, model weights, network or GPU are required.
"""

import builtins
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from baselines.single_prompt.patient import SinglePromptPatient
from core.interfaces import DialogueMessage
from core.profiles import profile_fingerprint
from helpers import profile_data
from patient_agents.patient import PatientAgents


class RecordingModel:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])

    def reply(self, messages):
        self.calls.append(copy.deepcopy(messages))
        content = self.responses.pop(0) if self.responses else "Could you tell me a little more?"
        return {"content": content, "input_tokens": 10, "output_tokens": 5, "finish_reason": "eos"}

    def visible_text(self):
        return json.dumps(self.calls[-1], ensure_ascii=False)


def controlled_options(**overrides):
    """Deterministic information loss/retrieval, independent of model sampling."""
    options = {
        "traits": {"attention": "markedly_reduced", "memory": "preserved", "language": "preserved"},
        "parameters": {
            "attention_units_markedly_reduced": 1,
            "encoding_probability_preserved": 1.0,
            "encoding_probability_markedly_reduced": 1.0,
            "retrieval_probability_preserved": 1.0,
            "retention_per_turn_preserved": 1.0,
        },
    }
    options.update(overrides)
    return options


class PatientAgentTests(unittest.TestCase):
    @staticmethod
    def turn(patient, history, text):
        history.append(DialogueMessage("Therapist", text))
        result = patient.respond(tuple(history))
        history.append(DialogueMessage("Patient", result["content"]))
        return result

    def patient(self, options=None, profile=None, responses=None, seed=42):
        model = RecordingModel(responses)
        patient = PatientAgents(
            model, "Speak only as the patient.",
            patient_profile=profile or profile_data(), seed=seed,
            options=controlled_options() if options is None else options,
        )
        return patient, model

    def test_output_model_gets_visible_profile_but_no_scores_sources_or_answer_key(self):
        profile = profile_data()
        profile["provenance"]["answer_key"] = "UNSEEN_THERAPIST_ANSWER_QZV"
        original = copy.deepcopy(profile)
        patient, model = self.patient(profile=profile)
        self.turn(patient, [], "Who do you live with?")
        visible = model.visible_text()
        self.assertIn("I live with my spouse.", visible)
        for hidden in ("PRIVATE_SOURCE_SENTINEL", "UNSEEN_THERAPIST_ANSWER_QZV",
                       '"moca"', '"sdmt"', '"dass21"', "category_cue", "multiple_choice_cue"):
            self.assertNotIn(hidden, visible)
        self.assertEqual(profile, original)

    def test_unattended_fact_is_not_restored_by_a_question_but_restatement_can_encode_it(self):
        patient, model = self.patient()
        history = []
        self.turn(patient, history, "Please take your time. The password is ZORPENQ.")
        self.assertNotIn("ZORPENQ", model.visible_text())
        self.turn(patient, history, "What was the password?")
        self.assertNotIn("ZORPENQ", model.visible_text())
        self.turn(patient, history, "The password is ZORPENQ.")
        self.assertIn("ZORPENQ", model.visible_text())
        self.turn(patient, history, "What was the password?")
        self.assertIn("ZORPENQ", model.visible_text())

    def test_audit_trace_records_omissions_but_the_response_input_cannot_see_them(self):
        patient, model = self.patient()
        self.turn(patient, [], "Look at the card. Its hidden code is QUENLAX.")
        trace = patient.last_trace
        self.assertIn("QUENLAX", json.dumps(trace["input_units"]))
        self.assertTrue(trace["omitted_ids"])
        payload = json.loads(model.calls[-1][-1]["content"])
        self.assertEqual(payload["turn_context"], trace["turn_context"])
        self.assertNotIn("QUENLAX", json.dumps(payload))
        self.assertNotIn("input_units", payload["turn_context"])
        self.assertNotIn("omitted_ids", payload["turn_context"])

    def test_unencoded_but_heard_fact_does_not_reappear_from_the_public_transcript(self):
        options = controlled_options(attention_enabled=False)
        options["parameters"]["encoding_probability_markedly_reduced"] = 0.0
        patient, model = self.patient(options=options)
        history = []
        self.turn(patient, history, "The parcel label is NAVLORX.")
        self.assertIn("NAVLORX", model.visible_text(), "The current stimulus was heard.")
        self.turn(patient, history, "What was the parcel label?")
        self.assertNotIn("NAVLORX", model.visible_text(), "A retrieval cue cannot create an unencoded trace.")

    def test_generated_self_report_can_be_remembered_without_becoming_a_personal_fact(self):
        patient, model = self.patient(responses=["My personal code is FABRILOX."])
        history = []
        self.turn(patient, history, "How are you today?")
        self.turn(patient, history, "What personal code do you remember?")
        payload = json.loads(model.calls[-1][-1]["content"])
        context = payload["turn_context"]
        self.assertNotIn("FABRILOX", json.dumps(context["available_personal_facts"]))
        self.assertNotIn("FABRILOX", json.dumps(payload["patient"]))
        self.assertNotIn("FABRILOX", json.dumps(patient.describe()["compiled_profile"]["known_facts"]))
        self_reports = [item for item in context["retrieved_memories"] if "FABRILOX" in item["text"]]
        self.assertTrue(self_reports, "A prior utterance may be recalled as something the patient said.")
        self.assertTrue(all(item["origin"] == "patient_utterance" for item in self_reports))

    def test_immediate_follow_up_can_use_working_memory_without_restoring_older_episodes(self):
        options = controlled_options()
        options["parameters"]["retrieval_probability_preserved"] = 0.0
        patient, model = self.patient(options=options, responses=["I preferred the VELNORA card."])
        history = []
        self.turn(patient, history, "Choose an item.")
        self.turn(patient, history, "Why?")
        payload = json.loads(model.calls[-1][-1]["content"])
        recent = [item for item in payload["turn_context"]["retrieved_memories"] if "VELNORA" in item["text"]]
        self.assertTrue(recent, "An immediate elliptical follow-up should retain the patient's last statement.")
        self.assertTrue(all(item["origin"] == "patient_utterance" for item in recent))
        self.turn(patient, history, "Let's discuss something else.")
        self.turn(patient, history, "Which card did you prefer earlier?")
        self.assertNotIn("VELNORA", model.visible_text(), "Working memory must not become an unlimited history bypass.")

    def test_preexisting_episode_is_not_treated_as_just_heard_working_memory(self):
        profile = profile_data()
        profile["initial_memories"] = [{
            "id": "previous_session", "text": "The badge from the previous session was DEMVORQ.",
            "source": "authored prior-session fixture", "cue_tags": ["badge"],
        }]
        options = controlled_options()
        options["parameters"]["retrieval_probability_preserved"] = 0.0
        patient, model = self.patient(options=options, profile=profile)
        self.turn(patient, [], "What was the badge from the previous session?")
        self.assertNotIn("DEMVORQ", model.visible_text(), "Preexisting episodes must pass retrieval, even on turn one.")

    def test_source_workbooks_are_not_read_during_construction_or_conversation(self):
        profile = profile_data()
        profile["known_facts"][0]["source"] = "/does/not/exist/participant_profile.xlsx / Sheet1!A2"
        real_builtin_open, real_io_open = builtins.open, io.open

        def guard(original):
            def checked(file, *args, **kwargs):
                if isinstance(file, (str, bytes, Path)) and str(file).lower().endswith((".xlsx", ".xls")):
                    raise AssertionError("Runtime attempted to open a source workbook.")
                return original(file, *args, **kwargs)
            return checked

        with mock.patch("builtins.open", side_effect=guard(real_builtin_open)), \
                mock.patch("io.open", side_effect=guard(real_io_open)):
            patient, model = self.patient(profile=profile)
            self.turn(patient, [], "Who do you live with?")
        self.assertEqual(len(model.calls), 1)

    def test_fixed_seed_reproduces_context_and_trace_independently_of_other_instances(self):
        options = controlled_options(attention_enabled=False)
        options["parameters"].update({
            "encoding_probability_preserved": 0.5,
            "encoding_probability_markedly_reduced": 0.5,
            "retrieval_probability_preserved": 0.5,
            "retention_per_turn_preserved": 0.8,
        })
        first, first_model = self.patient(options=options, seed=712)
        noise, _ = self.patient(options=options, seed=712)
        second, second_model = self.patient(options=options, seed=712)
        first_history, second_history, noise_history = [], [], []
        for stimulus in ("The badge is VESPARX.", "The folder is blue.", "What was the badge?", "What was the folder?"):
            self.turn(first, first_history, stimulus)
            self.turn(noise, noise_history, "Unrelated distracting exchange.")
            self.turn(second, second_history, stimulus)
            self.assertEqual(first.last_trace, second.last_trace)
        self.assertEqual(first_model.calls, second_model.calls)

    def test_attention_and_memory_ablations_preserve_information_access(self):
        patient, model = self.patient(options=controlled_options(attention_enabled=False, memory_enabled=False))
        history = []
        self.turn(patient, history, "The first token is LARVEX. The second token is MOVNIR.")
        self.assertIn("LARVEX", model.visible_text())
        self.assertIn("MOVNIR", model.visible_text())
        self.turn(patient, history, "Which tokens were presented?")
        self.assertIn("LARVEX", model.visible_text())
        self.assertIn("MOVNIR", model.visible_text())

    def test_baseline_and_proposed_method_describe_the_same_source_profile(self):
        profile = profile_data()
        baseline_model = RecordingModel()
        baseline = SinglePromptPatient(baseline_model, "Speak as the patient.", patient_profile=profile)
        proposed, proposed_model = self.patient(profile=profile)
        expected = profile_fingerprint(profile)
        self.assertEqual(baseline.describe()["profile_sha256"], expected)
        self.assertEqual(proposed.describe()["profile_sha256"], expected)
        self.turn(baseline, [], "Who do you live with?")
        self.turn(proposed, [], "Who do you live with?")
        for model in (baseline_model, proposed_model):
            self.assertIn("I live with my spouse.", model.visible_text())
            self.assertNotIn("PRIVATE_SOURCE_SENTINEL", model.visible_text())

    def test_missing_domains_remain_unknown_even_when_a_total_score_exists(self):
        profile = profile_data()
        profile["assessments"]["moca"].update({
            "total": 30, "attention": None, "delayed_recall": None, "language": None,
        })
        patient, _ = self.patient(options={}, profile=profile)
        traits = patient.describe()["compiled_profile"]["traits"]
        self.assertEqual(traits, {"attention": "unknown", "memory": "unknown", "language": "unknown"})

    def test_reusing_or_rewriting_a_processed_history_fails_before_another_generation(self):
        patient, model = self.patient()
        first = (DialogueMessage("Therapist", "The item is red."),)
        patient.respond(first)
        with self.assertRaises(ValueError):
            patient.respond(first)
        with self.assertRaises(ValueError):
            patient.respond((DialogueMessage("Therapist", "The item is blue."),))
        self.assertEqual(len(model.calls), 1)

    def test_invalid_options_and_profiles_fail_before_any_model_call(self):
        invalid_options = [
            [],
            {"parameters": {"encoding_probability_preserved": -0.1}},
            {"parameters": {"retrieval_probability_preserved": 1.1}},
            {"parameters": {"retention_per_turn_preserved": float("nan")}},
            {"parameters": {"attention_preserved_min": 2, "attention_reduced_min": 4}},
            {"parameters": {"language_preserved_min": 99}},
            {"traits": {"attention": "invented_level"}},
            {"unexpected_option": True},
        ]
        for options in invalid_options:
            with self.subTest(options=options):
                model = RecordingModel()
                with self.assertRaises(ValueError):
                    PatientAgents(model, "Speak as the patient.", patient_profile=profile_data(), options=options)
                self.assertEqual(model.calls, [])
        model = RecordingModel()
        with self.assertRaises(ValueError):
            PatientAgents(model, "Speak as the patient.", patient_profile={})
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
