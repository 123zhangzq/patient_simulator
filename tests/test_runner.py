"""Batch orchestration checks with a fake local model; no GPU packages needed."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from baselines.single_prompt.patient import SinglePromptPatient
from core.cases import DialogueCase, compose_prompts, load_cases
from core.therapist import Therapist
from helpers import profile_data


class FakeModel:
    instances = []
    fail_on_reply = None
    therapist_action = "continue"
    invalid_therapist_json = False

    def __init__(self, config):
        self.config = dict(config)
        self.dtype = "fake-float16"
        self.calls = []
        self.instances.append(self)

    def reply(self, messages):
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == self.fail_on_reply:
            raise RuntimeError("simulated generation failure")
        is_therapist = messages[0]["content"].startswith(("THERAPIST_", "You are a therapist"))
        content = f"reply-{len(self.calls)}"
        if is_therapist:
            content = "invalid control reply" if self.invalid_therapist_json else json.dumps({
                "utterance": content, "action": self.therapist_action})
        return {
            "content": content,
            "input_tokens": 10,
            "output_tokens": 2,
            "finish_reason": "eos",
        }


class TrackingTherapist(Therapist):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.histories = []
        self.instances.append(self)

    def respond(self, history):
        self.histories.append(tuple(history))
        return super().respond(history)


class TrackingPatient(SinglePromptPatient):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.histories = []
        self.instances.append(self)

    def respond(self, history):
        self.histories.append(tuple(history))
        return super().respond(history)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = {
            "model_id": "test/model", "revision": "test-revision",
            "model_dir": str(self.root / "model"),
            "output_dir": str(self.root / "outputs"), "rounds": 2, "seed": 42,
        }
        model_dir = Path(self.config["model_dir"])
        model_dir.mkdir()
        (model_dir / "download_manifest.json").write_text(json.dumps({
            "model_id": self.config["model_id"],
            "requested_revision": self.config["revision"],
        }), encoding="utf-8")
        for field in ("therapist", "opening", "patient"):
            path = self.root / f"{field}.txt"
            path.write_text(f"BASE_{field.upper()}", encoding="utf-8")
            self.config[f"{field}_prompt"] = str(path)

        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "test"
        fake_torch.cuda = types.SimpleNamespace(get_device_name=lambda index: "fake GPU")
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.__version__ = "test"
        fake_transformers.set_seed = mock.Mock()
        fake_llm_module = types.ModuleType("core.llm")
        fake_llm_module.LocalQwen = FakeModel
        self.modules = mock.patch.dict(sys.modules, {
            "torch": fake_torch,
            "transformers": fake_transformers,
            "core.llm": fake_llm_module,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        sys.modules.pop("core.runner", None)
        self.runner = importlib.import_module("core.runner")
        self.addCleanup(lambda: sys.modules.pop("core.runner", None))
        FakeModel.instances = []
        FakeModel.fail_on_reply = None
        FakeModel.therapist_action = "continue"
        FakeModel.invalid_therapist_json = False
        TrackingTherapist.instances = []
        TrackingPatient.instances = []

        self.patches = mock.patch.multiple(
            self.runner,
            LocalQwen=FakeModel,
            Therapist=TrackingTherapist,
            load_patient_class=mock.Mock(return_value=TrackingPatient),
            version=lambda package: "test",
            compose_prompts=lambda case, therapist, opening: {
                "therapist": f"THERAPIST_{case.case_id}",
                "opening": f"OPENING_{case.case_id}",
            },
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.seed = fake_transformers.set_seed

    def case(self, case_id, rounds=None):
        data = {"id": case_id, "patient_profile": profile_data(case_id),
                "therapist": {"opening_question": f"Opening for {case_id}?"}}
        if rounds is not None:
            data["rounds"] = rounds
        return DialogueCase(self.root / case_id / "case.json", data)

    def run_batch(self, cases, **kwargs):
        with mock.patch("builtins.print"):
            return self.runner.run(self.config, self.root / "config.json", cases=cases, **kwargs)

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_model_is_shared_but_agents_histories_and_seeds_restart_per_case(self):
        output = self.run_batch([self.case("case_a", 1), self.case("case_b")])
        self.assertEqual(len(FakeModel.instances), 1)
        self.assertEqual(len(TrackingTherapist.instances), 2)
        self.assertEqual(len(TrackingPatient.instances), 2)
        self.assertEqual([len(agent.histories) for agent in TrackingTherapist.instances], [1, 2])
        for therapist, patient in zip(TrackingTherapist.instances, TrackingPatient.instances):
            self.assertEqual(therapist.histories[0], ())
            self.assertEqual(len(patient.histories[0]), 1)
            self.assertEqual(patient.histories[0][0].speaker, "Therapist")
            self.assertTrue(patient.system_prompt.startswith("BASE_PATIENT"))
            self.assertIn("Case patient information", patient.system_prompt)
            self.assertNotIn("PRIVATE_SOURCE_SENTINEL", patient.system_prompt)
        self.assertEqual(self.seed.call_args_list, [mock.call(42)] * 3)

        batch = self.read_json(output / "batch_metadata.json")
        self.assertEqual(batch["status"], "completed")
        self.assertEqual(batch["completed_cases"], 2)
        self.assertEqual([case["effective_rounds"] for case in batch["cases"]], [1, 2])
        for case_id, rounds in (("case_a", 1), ("case_b", 2)):
            metadata = self.read_json(output / case_id / "metadata.json")
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["completed_rounds"], rounds)
            self.assertEqual(metadata["prompts"]["patient"], "BASE_PATIENT")
            records = (output / case_id / "dialogue.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 2 * rounds)

    def test_cli_round_override_wins_over_case_and_configuration_rounds(self):
        output = self.run_batch([self.case("case_a", 1), self.case("case_b")], rounds_override=3)
        batch = self.read_json(output / "batch_metadata.json")
        self.assertEqual([case["effective_rounds"] for case in batch["cases"]], [3, 3])
        self.assertEqual([len(agent.histories) for agent in TrackingPatient.instances], [3, 3])
        self.assertEqual(self.config["rounds"], 2)

    def test_partial_generation_failure_is_persisted_and_later_case_stays_pending(self):
        FakeModel.fail_on_reply = 3
        with self.assertRaisesRegex(RuntimeError, "simulated generation failure"):
            self.run_batch([self.case("case_a", 2), self.case("case_b", 1)])
        output, = Path(self.config["output_dir"]).iterdir()
        metadata = self.read_json(output / "case_a" / "metadata.json")
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["completed_rounds"], 1)
        self.assertEqual(metadata["utterances"], 3)
        self.assertIn("simulated generation failure", metadata["error"])
        records = (output / "case_a" / "dialogue.jsonl").read_text().splitlines()
        self.assertEqual(len(records), 3)
        batch = self.read_json(output / "batch_metadata.json")
        self.assertEqual(batch["status"], "failed")
        self.assertEqual([case["status"] for case in batch["cases"]], ["failed", "pending"])
        self.assertEqual(batch["cases"][0]["utterances"], 3)
        self.assertFalse((output / "case_b").exists())
        traces = [json.loads(line) for line in (output / "case_a/patient_trace.jsonl").read_text().splitlines()]
        self.assertEqual([record["status"] for record in traces], ["completed", "failed"])

    def test_proposed_method_runs_multiple_cases_with_private_traces_and_fresh_memory(self):
        from patient_agents.patient import PatientAgents
        self.runner.load_patient_class.return_value = PatientAgents
        output = self.run_batch([self.case("case_a", 2), self.case("case_b", 1)], patient_method="patient_agents")
        self.assertEqual(len(FakeModel.instances), 1)
        self.assertEqual(len(FakeModel.instances[0].calls), 4)
        for case_id, rounds in (("case_a", 2), ("case_b", 1)):
            metadata = self.read_json(output / case_id / "metadata.json")
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["patient_initialization"]["method"], "patient_agents")
            self.assertEqual(len(metadata["patient_profile_sha256"]), 64)
            traces = [json.loads(line) for line in (output / case_id / "patient_trace.jsonl").read_text().splitlines()]
            self.assertEqual(len(traces), rounds)
            self.assertEqual(traces[0]["trace"]["turn"], 1)
            dialogue = (output / case_id / "dialogue.jsonl").read_text()
            self.assertNotIn("turn_context", dialogue)
            self.assertNotIn("PRIVATE_SOURCE_SENTINEL", dialogue)
        for messages in FakeModel.instances[0].calls:
            serialized = json.dumps(messages)
            self.assertNotIn("PRIVATE_SOURCE_SENTINEL", serialized)

    def test_model_load_failure_leaves_batch_metadata(self):
        with mock.patch.object(self.runner, "LocalQwen", side_effect=RuntimeError("model unavailable")):
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                self.run_batch([self.case("case_a")])
        output, = Path(self.config["output_dir"]).iterdir()
        batch = self.read_json(output / "batch_metadata.json")
        self.assertEqual(batch["status"], "failed")
        self.assertEqual(batch["cases"][0]["status"], "pending")
        self.assertIn("model unavailable", batch["error"])

    def test_supplied_prefix_is_shared_by_methods_and_never_counted_as_new_speech(self):
        from patient_agents.patient import PatientAgents
        case = self.case("continuation", 1)
        case.data["session"] = {
            "mode": "continuation", "shared_context": "We are continuing a practice; you reported using notes.",
            "initial_history": [
                {"speaker": "Therapist", "content": "The practice label was XYLORP.", "source": "PRIVATE_PREFIX_FILE"},
                {"speaker": "Patient", "content": "I use notes to help.", "source": "PRIVATE_PREFIX_FILE"},
            ], "evaluation_note": "HIDDEN_SUFFIX_EXCLUDED_FROM_EVALUATION",
        }
        outputs = []
        for method, patient_class in (("single_prompt", TrackingPatient), ("patient_agents", PatientAgents)):
            self.runner.load_patient_class.return_value = patient_class
            output = self.run_batch([case], patient_method=method) / "continuation"
            outputs.append(output)
            metadata = self.read_json(output / "metadata.json")
            self.assertEqual(metadata["initial_history_utterances"], 2)
            self.assertEqual(metadata["completed_rounds"], 1)
            self.assertEqual(metadata["utterances"], 2)
            records = [json.loads(line) for line in (output / "dialogue.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["content"], case.data["therapist"]["opening_question"])
            self.assertEqual(len(records), 2)
            self.assertEqual(len(self.read_json(output / "initial_history.json")), 2)
            traces = [json.loads(line) for line in (output / "patient_trace.jsonl").read_text().splitlines()]
            visible = json.dumps(traces[0]["trace"]["model_input"])
            self.assertIn("XYLORP", visible)
            self.assertIn(case.data["session"]["shared_context"], visible)
            self.assertNotIn("PRIVATE_PREFIX_FILE", visible)
            self.assertNotIn("HIDDEN_SUFFIX_EXCLUDED_FROM_EVALUATION", visible)
            if method == "patient_agents":
                self.assertEqual(traces[0]["trace"]["turn"], 2)
        left, right = [self.read_json(output / "metadata.json") for output in outputs]
        self.assertEqual(left["patient_profile_sha256"], right["patient_profile_sha256"])
        self.assertEqual(left["session_sha256"], right["session_sha256"])

    def test_close_stops_after_patient_reply_and_keeps_control_out_of_public_speech(self):
        FakeModel.therapist_action = "close"
        output = self.run_batch([self.case("case_a", 12)]) / "case_a"
        metadata = self.read_json(output / "metadata.json")
        self.assertEqual(metadata["completed_rounds"], 2)
        self.assertEqual(metadata["stop_reason"], "therapist_close")
        records = [json.loads(line) for line in (output / "dialogue.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 4)
        self.assertEqual(records[-1]["speaker"], "Patient")
        self.assertNotIn('"action"', json.dumps(records))
        traces = [json.loads(line) for line in (output / "therapist_trace.jsonl").read_text().splitlines()]
        self.assertEqual(len(traces), 2)
        for history in TrackingPatient.instances[-1].histories:
            self.assertTrue(all('"action"' not in message.content for message in history))

    def test_repeated_clarification_stops_without_filling_twelve_rounds(self):
        FakeModel.therapist_action = "clarify"
        output = self.run_batch([self.case("case_a", 12)]) / "case_a"
        metadata = self.read_json(output / "metadata.json")
        self.assertEqual(metadata["completed_rounds"], 4)
        self.assertEqual(metadata["stop_reason"], "clarification_limit")

    def test_invalid_therapist_control_persists_raw_trace_and_partial_dialogue(self):
        FakeModel.invalid_therapist_json = True
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            self.run_batch([self.case("case_a", 3)])
        root, = Path(self.config["output_dir"]).iterdir()
        output = root / "case_a"
        metadata = self.read_json(output / "metadata.json")
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["completed_rounds"], 1)
        self.assertEqual(metadata["utterances"], 2)
        traces = [json.loads(line) for line in (output / "therapist_trace.jsonl").read_text().splitlines()]
        self.assertEqual(traces[-1]["status"], "failed")
        self.assertEqual(traces[-1]["trace"]["raw_result"]["content"], "invalid control reply")

    def test_repository_cases_run_with_the_real_composition_for_both_methods(self):
        from patient_agents.patient import PatientAgents
        for key in ("therapist", "opening", "patient"):
            name = "baselines/single_prompt/patient" if key == "patient" else key
            self.config[f"{key}_prompt"] = str(CODE_DIR / "prompts" / f"{name}.txt")
        cases = load_cases(CODE_DIR / "cases")
        snapshots = {}
        with mock.patch.object(self.runner, "compose_prompts", compose_prompts):
            for method, patient_class in (("single_prompt", SinglePromptPatient), ("patient_agents", PatientAgents)):
                self.runner.load_patient_class.return_value = patient_class
                output = self.run_batch(cases, patient_method=method)
                for case in cases:
                    metadata = self.read_json(output / case.case_id / "metadata.json")
                    prefix = self.read_json(output / case.case_id / "initial_history.json")
                    records = [json.loads(line) for line in (output / case.case_id / "dialogue.jsonl").read_text().splitlines()]
                    self.assertEqual(metadata["status"], "completed")
                    self.assertEqual(metadata["completed_rounds"], case.data["rounds"])
                    self.assertEqual(metadata["stop_reason"], "round_limit")
                    self.assertEqual(len(prefix), len(case.data["session"]["initial_history"]))
                    self.assertEqual(records[0]["content"], case.data["therapist"]["opening_question"])
                    self.assertEqual(len(records), 2 * case.data["rounds"])
                    exposure = (metadata["patient_profile_sha256"], metadata["session_sha256"], prefix)
                    if method == "single_prompt":
                        snapshots[case.case_id] = exposure
                    else:
                        self.assertEqual(exposure, snapshots[case.case_id])


if __name__ == "__main__":
    unittest.main()
