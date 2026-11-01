"""Bounded structured generation for private patient state, never public speech."""

from copy import deepcopy
import json
import re


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_object(text):
    if not isinstance(text, str):
        raise ValueError("Expected JSON text.")
    decoder = json.JSONDecoder(object_pairs_hook=_unique_fields)
    text = text.strip()
    try:
        result = decoder.decode(text)
    except json.JSONDecodeError:
        start = re.search(r"[\[{]", text)
        if start is None:
            raise ValueError("No JSON object found.") from None
        result, end = decoder.raw_decode(text, start.start())
        if re.search(r"[{}\[\]]", text[end:]):
            raise ValueError("Multiple or ambiguous JSON objects.")
    if not isinstance(result, dict):
        raise ValueError("Expected one JSON object.")
    return result


def generate_object(llm, messages, validator, repair_prompt, trace, max_new_tokens=None, json_schema=None):
    """Validate one result, allowing one recorded repair with the same evidence."""
    trace.update(generation_attempts=[], usage={"input_tokens": 0, "output_tokens": 0})
    if json_schema is not None:
        trace["json_schema"] = deepcopy(json_schema)
    request = deepcopy(messages)
    for attempt in range(2):
        record = {"model_input": deepcopy(request)}
        if json_schema is not None:
            record["json_schema"] = deepcopy(json_schema)
        trace["generation_attempts"].append(record)
        try:
            kwargs = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
            if json_schema is not None:
                kwargs["json_schema"] = deepcopy(json_schema)
            result = llm.reply(request, **kwargs)
            record["result"] = deepcopy(result)
            for key in trace["usage"]:
                trace["usage"][key] += result[key]
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        try:
            if result["finish_reason"] == "length":
                raise ValueError("Structured output reached its token limit; keep the JSON shorter.")
            parsed = validator(parse_object(result["content"]))
        except ValueError as error:
            record["error"] = str(error)
            if attempt:
                raise ValueError(
                    "Patient structured output remained invalid after one repair. "
                    "Inspect patient_initialization or patient_trace.jsonl. " + str(error)
                ) from error
            request = [*deepcopy(messages), {"role": "assistant", "content": result["content"]},
                       {"role": "user", "content": repair_prompt + "\nValidation error: " + str(error)}]
            continue
        trace["format_repaired"] = bool(attempt)
        return parsed, {**result, **trace["usage"]}

