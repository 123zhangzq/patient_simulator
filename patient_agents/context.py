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
             "origin": "initial_memory", "speaker": None, "addressee": None,
             "cue_tags": item.get("cue_tags", [])}
            for item in self.compiled["initial_memories"]
        ]
        self.last_trace: dict = {}

    def record_patient_reply(self, text: str, *, from_initial_history: bool = False) -> list[str]:
        """Remember the act of speaking, without asserting the content is true."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A nonempty patient reply is required.")
        ids = []
        for index, unit in enumerate(split_units(text)):
            identifier = f"t{self.turn}:p{index + 1}"
            self.memories.append({"id": identifier, "text": unit, "encoded_turn": self.turn,
                                  "origin": "patient_utterance", "cue_tags": [],
                                  "speaker": "Patient", "addressee": "Therapist",
                                  "from_initial_history": from_initial_history})
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
        units = [{"id": f"t{self.turn}:u{index + 1}", "text": text,
                  "speaker": "Therapist", "addressee": "Patient"}
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
        # Keep the whole last self-report in the immediate workspace. Its
        # opening choice or plan must not vanish merely because it was followed
        # by several short clauses. It expires with the recent workspace and
        # never becomes an unlimited transcript or a verified personal fact.
        working_ids.update(item["id"] for item in recent if item["origin"] == "patient_utterance")
        # Encoded dialogue stays eligible even when a follow-up uses only a
        # pronoun or a new phrase. Relevance and recency rank these candidates;
        # the older-episode limit and probabilistic gate still control access.
        # Preloaded memories require a matching cue or a recall request. Neither
        # the public transcript nor an omitted input is restored here.
        ranked = []
        for item in self.memories:
            overlap = len(query & self._tokens(item["text"]))
            tag_match = any(self._tokens(tag) & query for tag in item["cue_tags"])
            if (not self.memory_enabled or item["origin"] != "initial_memory"
                    or overlap or tag_match or recall_requested):
                ranked.append((overlap + int(tag_match), item))
        ranked.sort(key=lambda pair: (pair[1]["id"] in working_ids, pair[0], pair[1]["encoded_turn"]), reverse=True)
        # A retrieval limit applies to older episodes, not to a current turn's
        # conversational commitments. Keep each working item, then retrieve a
        # bounded number of older episodes through the probabilistic gate.
        working = [pair for pair in ranked if pair[1]["id"] in working_ids]
        older = [pair for pair in ranked if pair[1]["id"] not in working_ids]
        limit = self.parameters["retrieval_limit"] if self.memory_enabled else len(older)
        for _, item in working + older[:limit]:
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
                                  "speaker": item["speaker"], "addressee": item["addressee"],
                                  "from_initial_history": item.get("from_initial_history", False),
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
        # Empty history can mean missing scenario information, not impaired
        # comprehension or even a failed retrieval. Do not turn it into affect.
        difficulty = bool(omitted)
        self.frustration = min(2, self.frustration + 1) if difficulty else max(0, self.frustration - 1)
        context = {
            "turn": self.turn,
            "accessible_current_units": deepcopy(attended),
            "retrieved_memories": retrieved,
            "information_access": {
                "current_input": "partial" if omitted else "complete",
                "history": "earlier_statements_returned" if retrieved else "no_earlier_statements_returned",
            },
            "interaction_state": {"affect": "mild_frustration" if self.frustration >= 2 else "neutral",
                                  "some_input_was_missed": bool(omitted)},
        }
        self.last_trace = {
            "turn": self.turn, "input_units": units, "attended_ids": [u["id"] for u in attended],
            "omitted_ids": [u["id"] for u in omitted], "encoding": encoding_events,
            "retrieval": retrieval_events, "encoded_memory_count": len(self.memories),
            "recall_keyword_detected": recall_requested,
            "state": {"frustration": self.frustration}, "turn_context": deepcopy(context),
        }
        return context
