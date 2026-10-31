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
from core.cases import DialogueCase
from core.therapist import Therapist


class FakeModel:
    instances = []
    fail_on_reply = None

    def __init__(self, config):
        self.config = dict(config)
        self.dtype = "fake-float16"
        self.calls = []
        self.instances.append(self)

    def reply(self, messages):
        self.calls.append([dict(message) for message in messages])
        if len(self.calls) == self.fail_on_reply:
            raise RuntimeError("simulated generation failure")
        return {
            "content": f"reply-{len(self.calls)}",
            "input_tokens": 10,
            "output_tokens": 2,
            "finish_reason": "eos",
        }


class TrackingTherapist(Therapist):
    instances = []

    def __init__(self, *args):
        super().__init__(*args)
        self.histories = []
        self.instances.append(self)

    def respond(self, history):
        self.histories.append(tuple(history))
        return super().respond(history)


class TrackingPatient(SinglePromptPatient):
    instances = []

    def __init__(self, *args):
        super().__init__(*args)
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
        data = {"id": case_id}
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
            self.assertEqual(patient.system_prompt, "BASE_PATIENT")
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
        FakeModel.fail_on_reply = 4
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

    def test_model_load_failure_leaves_batch_metadata(self):
        with mock.patch.object(self.runner, "LocalQwen", side_effect=RuntimeError("model unavailable")):
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                self.run_batch([self.case("case_a")])
        output, = Path(self.config["output_dir"]).iterdir()
        batch = self.read_json(output / "batch_metadata.json")
        self.assertEqual(batch["status"], "failed")
        self.assertEqual(batch["cases"][0]["status"], "pending")
        self.assertIn("model unavailable", batch["error"])


if __name__ == "__main__":
    unittest.main()
