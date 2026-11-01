"""Therapist control behavior, without model weights or inference dependencies."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from core.interfaces import DialogueMessage
from core.therapist import Therapist, load_control


class ScriptedModel:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.calls = []

    def reply(self, messages):
        self.calls.append(messages)
        return {"content": next(self.replies), "input_tokens": 20, "output_tokens": 8, "finish_reason": "eos"}


def response(utterance, action="continue"):
    return json.dumps({"utterance": utterance, "action": action})


class TherapistTests(unittest.TestCase):
    def therapist(self, model, **kwargs):
        return Therapist(model, "Discuss the activity.", "Start at this opening.", **kwargs)

    def test_prefix_does_not_replace_deterministic_first_new_opening(self):
        model = ScriptedModel(response("What was difficult about that?"))
        therapist = self.therapist(model, opening_utterance="How did you use your notes?")
        history = [DialogueMessage("Therapist", "What do you use?"),
                   DialogueMessage("Patient", "I keep a spreadsheet.")]
        result = therapist.respond(history)
        self.assertEqual(result["content"], "How did you use your notes?")
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(therapist.last_trace["source"], "case_opening")
        history += [DialogueMessage("Therapist", result["content"]), DialogueMessage("Patient", "I check the steps.")]
        therapist.respond(history)
        self.assertEqual(len(model.calls), 1)
        self.assertIn("I keep a spreadsheet.", json.dumps(model.calls[0]))

    def test_control_fields_are_not_spoken_and_closing_prevents_further_turns(self):
        therapist = self.therapist(ScriptedModel(response("Let's stop here.", "close")))
        result = therapist.respond([])
        self.assertEqual(result["content"], "Let's stop here.")
        self.assertEqual(result["dialogue_action"], "close")
        self.assertEqual(result["stop_reason"], "therapist_close")
        self.assertNotIn("utterance", result["content"])
        with self.assertRaisesRegex(ValueError, "ended"):
            therapist.respond([])

    def test_third_consecutive_clarification_closes_even_if_model_ignores_budget(self):
        model = ScriptedModel(*(response(f"Could you explain question {i}?", "clarify") for i in range(3)))
        therapist = self.therapist(model)
        self.assertEqual(therapist.respond([])["dialogue_action"], "clarify")
        self.assertEqual(therapist.respond([])["dialogue_action"], "clarify")
        result = therapist.respond([])
        self.assertIn(therapist.control["repair_limit"], model.calls[-1][0]["content"])
        self.assertEqual(result["content"], therapist.control["stall_closing"])
        self.assertEqual(result["stop_reason"], "clarification_limit")
        self.assertEqual(therapist.last_trace["requested_action"], "clarify")
        self.assertEqual(therapist.last_trace["override"], "clarification_limit")

    def test_a_concrete_task_step_resets_consecutive_clarification_budget(self):
        model = ScriptedModel(response("Which part?", "clarify"), response("Here is one example."),
                              response("Which step do you mean?", "clarify"))
        therapist = self.therapist(model, max_clarifications=1)
        for _ in range(3):
            result = therapist.respond([])
        self.assertEqual(result["dialogue_action"], "clarify")
        self.assertEqual(therapist.consecutive_clarifications, 1)

    def test_repeated_opening_question_counts_as_repair_despite_continue_label(self):
        model = ScriptedModel(
            response("  HOW did you use your notes ??  "),
            response("How  did you use your notes？"),
            response("How did you use your notes?"),
        )
        therapist = self.therapist(model, opening_utterance="How did you use your notes?")
        therapist.respond([])
        self.assertFalse(therapist.last_trace["repeated_question"])
        for expected_count in (1, 2):
            result = therapist.respond([])
            self.assertEqual(result["dialogue_action"], "clarify")
            self.assertEqual(therapist.consecutive_clarifications, expected_count)
            self.assertTrue(therapist.last_trace["repeated_question"])
            self.assertEqual(therapist.last_trace["requested_action"], "continue")
        result = therapist.respond([])
        self.assertEqual(result["dialogue_action"], "close")
        self.assertEqual(result["stop_reason"], "clarification_limit")
        self.assertEqual(therapist.last_trace["requested_action"], "continue")
        self.assertTrue(therapist.last_trace["repeated_question"])

    def test_repeated_acknowledgments_do_not_use_clarification_budget(self):
        therapist = self.therapist(ScriptedModel(*(response("Okay.") for _ in range(3))),
                                  opening_utterance="Okay.", max_clarifications=1)
        for _ in range(4):
            result = therapist.respond([])
            self.assertEqual(result["dialogue_action"], "continue")
            self.assertFalse(therapist.last_trace["repeated_question"])
            self.assertEqual(therapist.consecutive_clarifications, 0)

    def test_repeat_detection_tracks_last_model_utterance_not_all_prior_questions(self):
        model = ScriptedModel(response("Which step came first?"),
                              response("Which step came first?"),
                              response("The first step was to open the file."),
                              response("Which step came first?"))
        therapist = self.therapist(model)
        therapist.respond([])
        self.assertEqual(therapist.respond([])["dialogue_action"], "clarify")
        therapist.respond([])
        self.assertEqual(therapist.respond([])["dialogue_action"], "continue")
        self.assertFalse(therapist.last_trace["repeated_question"])

    def test_round_limit_closes_instead_of_starting_another_question(self):
        model = ScriptedModel(response("Should we start another task?"))
        therapist = self.therapist(model, opening_utterance="Let's try one name.", max_rounds=2)
        therapist.respond([])
        result = therapist.respond([DialogueMessage("Patient", "Okay.")])
        self.assertEqual(result["dialogue_action"], "close")
        self.assertEqual(result["stop_reason"], "round_limit")
        self.assertEqual(result["content"], therapist.control["budget_closing"])
        self.assertIn(therapist.control["last_turn"], model.calls[0][0]["content"])

    def test_single_round_budget_still_delivers_the_opening(self):
        model = ScriptedModel()
        therapist = self.therapist(model, opening_utterance="Try this name.", max_rounds=1)
        self.assertEqual(therapist.respond([])["content"], "Try this name.")
        self.assertEqual(model.calls, [])
        with self.assertRaisesRegex(ValueError, "ended"):
            therapist.respond([])

    def test_malformed_control_fails_with_raw_trace_instead_of_guessing(self):
        for raw in ("Some ordinary prose.", '{"utterance":"hello","action":"unknown"}',
                    '{"utterance":"hello","action":"continue","extra":true}'):
            with self.subTest(raw=raw):
                therapist = self.therapist(ScriptedModel(raw))
                with self.assertRaisesRegex(ValueError, "Therapist"):
                    therapist.respond([])
                self.assertEqual(therapist.last_trace["status"], "failed")
                self.assertEqual(therapist.last_trace["raw_result"]["content"], raw)
                self.assertEqual(therapist.generated_turns, 0)

    def test_one_json_fence_is_accepted_but_external_prose_is_not(self):
        therapist = self.therapist(ScriptedModel('```json\n' + response("Take your time.") + '\n```'))
        self.assertEqual(therapist.respond([])["content"], "Take your time.")
        therapist = self.therapist(ScriptedModel('Here is my reply: ' + response("Take your time.")))
        with self.assertRaises(ValueError):
            therapist.respond([])

    def test_invalid_control_resource_fails_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.json"
            path.write_text('{}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "control"):
                load_control(path)


if __name__ == "__main__":
    unittest.main()
