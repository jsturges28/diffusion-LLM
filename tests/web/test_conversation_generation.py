"""Generation configuration codec and registry boundary.

The JSON vectors are also loaded by the browser codec tests. Passing
proves durable parsing is structural, while new writes require the
current device-qualified registry identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from src.web import conversation_generation


VECTORS_PATH = (
    Path(__file__).parent / "generation_configuration_vectors.json"
)


def _vectors() -> Dict[str, object]:
    raw = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_shared_valid_vectors_parse_structurally() -> None:
    vectors = _vectors()["valid"]
    assert isinstance(vectors, list)
    for vector in vectors:
        assert isinstance(vector, dict)
        wire = vector["wire"]
        assert isinstance(wire, dict)

        parser = (
            conversation_generation.parse_generation_configuration
        )
        parsed = parser(
            wire,
            expected_model_id=str(wire["model_id"]),
            expected_input_mode=str(wire["input_mode"]),
        )

        assert parsed == wire, vector["name"]


def test_shared_malformed_vectors_are_refused() -> None:
    vectors = _vectors()["invalid"]
    assert isinstance(vectors, list)
    for vector in vectors:
        assert isinstance(vector, dict)
        wire = vector["wire"]

        with pytest.raises(ValueError):
            conversation_generation.parse_generation_configuration(
                wire,
                expected_model_id="llada",
                expected_input_mode="chat",
            )


def test_current_registry_identity_is_required_for_writes() -> None:
    configuration = (
        conversation_generation.default_generation_configuration(
            model_id="llada",
            input_mode="chat",
        )
    )
    stale = dict(configuration)
    stale["schema_id"] = "0" * 64

    with pytest.raises(ValueError, match="current model/device"):
        conversation_generation.validate_generation_configuration(
            stale,
            expected_model_id="llada",
            expected_input_mode="chat",
        )

    assert (
        conversation_generation.validate_generation_configuration(
            configuration,
            expected_model_id="llada",
            expected_input_mode="chat",
        )
        == configuration
    )
