"""Explicit patient method registration with imports deferred until selection."""

from importlib import import_module
from typing import cast

from core.interfaces import Patient


_PATIENT_CLASSES = {
    "single_prompt": "baselines.single_prompt.patient.SinglePromptPatient",
    "patient_agents": "patient_agents.patient.PatientAgents",
}


def available_patient_methods() -> tuple[str, ...]:
    """Return the names of the explicitly registered patient methods."""
    return tuple(_PATIENT_CLASSES)


def load_patient_class(name: str) -> type[Patient]:
    """Import and return only the patient implementation selected by name."""
    if name not in _PATIENT_CLASSES:
        available = ", ".join(available_patient_methods())
        raise ValueError(f"Unknown patient method {name!r}. Available methods: {available}.")
    module_name, _, class_name = _PATIENT_CLASSES[name].rpartition(".")
    module = import_module(module_name)
    return cast(type[Patient], getattr(module, class_name))
