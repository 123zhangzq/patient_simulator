"""Own attention, encoded episodes, retrieval, and a small interaction state.

Punctuation segmentation and lexical cue matching are transparent approximations;
this module does not claim a semantic parser or clinically calibrated memory.
"""

from copy import deepcopy
import random
import re


def split_units(text: str) -> list[str]:
    """Split clauses without inventing or summarizing information."""
    return [part.strip() for part in re.split(
        r"(?<=[.!?;。！？；])\s*|\n+|,(?=\s+(?:and|then|also|but|after|before)\b)", text
    ) if part.strip()]


class ContextController:
    def __init__(self, compiled: dict, seed: int = 42, *, attention_enabled: bool = True,
                 memory_enabled: bool = True):
        self.compiled = deepcopy(compiled)
        self.policy = self.compiled["policy"]
        self.parameters = self.policy["parameters"]
        self.rng = random.Random(seed)
        self.attention_enabled = attention_enabled
        self.memory_enabled = memory_enabled
        self.turn = 0
        self.frustration = 0
        self.memories = [
            {"id": f"initial:{item['id']}", "text": item["text"], "encoded_turn": 0,
             "origin": "initial_memory", "cue_tags": item.get("cue_tags", [])}
            for item in self.compiled["initial_memories"]
        ]
        self.last_trace: dict = {}

    def record_patient_reply(self, text: str) -> list[str]:
        """Remember the act of speaking, without asserting the content is true."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A nonempty patient reply is required.")
        ids = []
        for index, unit in enumerate(split_units(text)):
            identifier = f"t{self.turn}:p{index + 1}"
            self.memories.append({"id": identifier, "text": unit, "encoded_turn": self.turn,
                                  "origin": "patient_utterance", "cue_tags": []})
            ids.append(identifier)
        return ids

    def _tokens(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        # Character bigrams provide basic matching for unsegmented CJK text.
        for run in re.findall(r"[\u3400-\u9fff]+", text):
            tokens.update(run[index:index + 2] for index in range(max(1, len(run) - 1)))
        return tokens - set(self.policy["stopwords"])

    def _markers(self, text: str, key: str) -> bool:
        return any(marker.lower() in text.lower() for marker in self.policy[key])

    def advance(self, therapist_text: str) -> dict:
        if not isinstance(therapist_text, str) or not therapist_text.strip():
            raise ValueError("A nonempty new therapist utterance is required.")
        self.turn += 1
        units = [{"id": f"t{self.turn}:u{index + 1}", "text": text}
                 for index, text in enumerate(split_units(therapist_text))]
        attention = self.compiled["traits"]["attention"]
        memory_trait = self.compiled["traits"]["memory"]
        capacity = self.parameters[f"attention_units_{attention}"]
        attended = units[:capacity] if self.attention_enabled else units
        omitted = units[len(attended):]
        accessible_text = " ".join(unit["text"] for unit in attended)
        query = self._tokens(accessible_text)
        recall_requested = self._markers(accessible_text, "recall_markers")
        explicit_cue = self._markers(accessible_text, "cue_markers")
        retrieval_events, retrieved = [], []
        recent = [item for item in self.memories
                  if item["origin"] != "initial_memory"
                  and self.turn - item["encoded_turn"] <= self.parameters["working_memory_turns"]]
        # A small recent workspace is distinct from delayed episodic retrieval.
        # Missing/failed encoding is never reconstructed from the transcript.
        working_ids = {item["id"] for item in (recent[-capacity:] if self.attention_enabled else recent)}
        # Only previously encoded episodes are candidates. Neither the public
        # transcript nor an omitted input unit is accessible to this retrieval.
        ranked = []
        for item in self.memories:
            overlap = len(query & self._tokens(item["text"]))
            tag_match = any(self._tokens(tag) & query for tag in item["cue_tags"])
            if not self.memory_enabled or item["id"] in working_ids or overlap or tag_match or recall_requested:
                ranked.append((overlap + int(tag_match), item))
        ranked.sort(key=lambda pair: (pair[1]["id"] in working_ids, pair[0], pair[1]["encoded_turn"]), reverse=True)
        limit = self.parameters["retrieval_limit"] if self.memory_enabled else len(ranked)
        for _, item in ranked[:limit]:
            age = max(0, self.turn - item["encoded_turn"] - 1)
            probability = 1.0
            cue_helped = False
            in_working_memory = item["id"] in working_ids
            if self.memory_enabled and not in_working_memory:
                probability = (self.parameters[f"retrieval_probability_{memory_trait}"]
                               * self.parameters[f"retention_per_turn_{memory_trait}"] ** age)
                # Cues improve access only to an encoded, relevant episode.
                cue_helped = explicit_cue and bool(query & self._tokens(
                    item["text"] + " " + " ".join(item["cue_tags"])
                ))
                if cue_helped:
                    probability = min(1.0, probability + self.parameters["cue_bonus"])
            success = probability >= 1 or self.rng.random() < probability
            retrieval_events.append({"id": item["id"], "probability": probability,
                                     "cue_applied": cue_helped, "retrieved": success,
                                     "access": "working_memory" if in_working_memory else "episodic_retrieval"})
            if success:
                retrieved.append({"id": item["id"], "text": item["text"],
                                  "origin": item["origin"], "learned_turn": item["encoded_turn"],
                                  "access": "working_memory" if in_working_memory else "episodic_retrieval"})
        # Encoding takes place after retrieval: a newly supplied answer is new
        # information, never evidence that it was remembered beforehand.
        encoding_events = []
        # Disabling the attention gate presents all units, but does not turn
        # off encoding failure in the separate memory stage.
        probability = self.parameters[f"encoding_probability_{attention}"] if self.memory_enabled else 1.0
        for unit in attended:
            success = probability >= 1 or self.rng.random() < probability
            encoding_events.append({"id": unit["id"], "probability": probability, "encoded": success})
            if success:
                self.memories.append({**unit, "encoded_turn": self.turn,
                                      "origin": "therapist_utterance", "cue_tags": []})
        facts = []
        candidates = [(len(query & self._tokens(
            item["text"] + " " + " ".join(item.get("cue_tags", []))
        )), item)
                      for item in self.compiled["known_facts"]]
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        for overlap, item in candidates[:self.parameters["known_fact_limit"]]:
            if overlap:
                facts.append({"id": item["id"], "text": item["text"]})
        difficulty = bool(omitted) or (recall_requested and not retrieved)
        self.frustration = min(2, self.frustration + 1) if difficulty else max(0, self.frustration - 1)
        context = {
            "turn": self.turn,
            "accessible_current_units": deepcopy(attended),
            "retrieved_memories": retrieved,
            "available_personal_facts": facts,
            "interaction_state": {"affect": "mild_frustration" if self.frustration >= 2 else "neutral",
                                  "some_input_was_missed": bool(omitted)},
            "response_policy": {
                "act": "answer_accessible_then_request_repeat" if omitted else "answer_accessible",
                "max_reply_units": self.parameters[f"max_reply_units_{attention if self.attention_enabled else 'preserved'}"],
                "recall_requested": recall_requested,
                "prior_recall_available": bool(retrieved),
            },
        }
        self.last_trace = {
            "turn": self.turn, "input_units": units, "attended_ids": [u["id"] for u in attended],
            "omitted_ids": [u["id"] for u in omitted], "encoding": encoding_events,
            "retrieval": retrieval_events, "encoded_memory_count": len(self.memories),
            "state": {"frustration": self.frustration}, "turn_context": deepcopy(context),
        }
        return context
