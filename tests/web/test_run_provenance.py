"""A saved run describes the worker that produced it.

Strategy: drive `_build_metadata` directly, with the supervisor's
global state monkeypatched to something *different* from what the run
attests. That difference is the whole test. If a field is read from
the manager, it shows up wrong; if it is read from the run's
provenance envelope, it shows up right.

The scenario being reproduced is not exotic. Two browser windows share
one supervisor. Window A finishes a run. Window B switches the model,
or the device. Window A clicks Save. Before this change the run was
saved with B's processor, context window, library versions and
tokenizer, and the reproducibility block said so with no hedging: a
record that is confidently wrong is worse than one that is missing,
because nothing about it looks suspect months later.

The worker half, that a backend attests where it actually loaded
rather than where it was told to load, is in
`tests/backends/test_worker_provenance.py`.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.web.server import (
    RunProvenance,
    SaveRunRequest,
    _build_metadata,
    _context_metadata,
    manager,
)

# What the run itself attests: an LLaDA worker that ran on CPU.
RUN_TOKENIZER: Dict[str, Any] = {
    "name_or_path": "GSAI-ML/LLaDA-8B-Instruct",
    "vocab_size": 126_464,
    "model_vocab_size": 126_464,
}
RUN_VERSIONS = {"torch": "2.4.0", "transformers": "4.38.2"}

# What the supervisor thinks is going on: a different model, on a
# different device, in a different environment.
OTHER_TOKENIZER: Dict[str, Any] = {
    "name_or_path": "HuggingFaceTB/SmolLM3-3B",
    "vocab_size": 128_256,
    "model_vocab_size": 128_256,
}
OTHER_VERSIONS = {"torch": "2.6.0", "transformers": "4.53.1"}


@pytest.fixture
def switched_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The state a second window leaves behind after switching."""
    monkeypatch.setattr(manager, "active_device", "cuda")
    monkeypatch.setattr(
        manager, "active_versions", OTHER_VERSIONS
    )
    monkeypatch.setattr(
        manager, "active_tokenizer", OTHER_TOKENIZER
    )
    monkeypatch.setattr(
        manager, "active_context_length", 65_536
    )


def _provenance(**overrides: Any) -> RunProvenance:
    base: Dict[str, Any] = {
        "model_id": "LLaDA-8B-Instruct",
        "checkpoint": "GSAI-ML/LLaDA-8B-Instruct",
        "device": "cpu",
        "versions": dict(RUN_VERSIONS),
        "tokenizer": dict(RUN_TOKENIZER),
        "context_length": 4_096,
    }
    base.update(overrides)
    return RunProvenance(**base)


def _request(**overrides: Any) -> SaveRunRequest:
    base: Dict[str, Any] = {
        "model": "LLaDA-8B-Instruct",
        "prompt": "hello",
        "final_text": "hello world",
        "frames": ["hello", "hello world"],
        "params": {"steps": 64, "seed": 7},
        "prompt_len": 12,
        "provenance": _provenance(),
    }
    base.update(overrides)
    return SaveRunRequest(**base)


# -- the run's facts beat the supervisor's --


def test_the_processor_is_the_run_s_not_the_current_one(
    switched_supervisor: None,
) -> None:
    """The headline case. The supervisor says cuda; the run ran on
    cpu and says so."""
    meta = _build_metadata(_request())

    assert meta["processor"] == "CPU"


def test_the_versions_are_the_run_s(
    switched_supervisor: None,
) -> None:
    meta = _build_metadata(_request())

    assert meta["reproducibility"]["versions"] == RUN_VERSIONS


def test_the_tokenizer_is_the_run_s(
    switched_supervisor: None,
) -> None:
    """Which vocabulary produced the run's token ids. Reading the
    resident model's would make every id in the run mean something
    else."""
    meta = _build_metadata(_request())

    assert meta["reproducibility"]["tokenizer"] == RUN_TOKENIZER


def test_the_context_window_is_the_run_s(
    switched_supervisor: None,
) -> None:
    """A prompt length is only meaningful against the window it
    competed for, so the window has to come from the same run."""
    meta = _build_metadata(_request())

    assert meta["context"]["context_length"] == 4_096
    assert meta["context"]["prompt_tokens"] == 12


def test_every_provenance_field_moves_together(
    switched_supervisor: None,
) -> None:
    """One assertion for the whole envelope, because a partial fix
    is the dangerous outcome: a record that is right about the
    tokenizer and wrong about the device reads as trustworthy."""
    meta = _build_metadata(_request())

    assert meta["processor"] == "CPU"
    assert meta["reproducibility"]["versions"] == RUN_VERSIONS
    assert meta["reproducibility"]["tokenizer"] == RUN_TOKENIZER
    assert meta["context"]["context_length"] == 4_096
    assert meta["model"] == "GSAI-ML/LLaDA-8B-Instruct"


def test_the_record_says_whether_it_was_attested(
    switched_supervisor: None,
) -> None:
    """A reader cannot otherwise tell a run that carried its own
    facts from one that borrowed the supervisor's."""
    attested = _build_metadata(_request())
    borrowed = _build_metadata(_request(provenance=None))

    assert attested["reproducibility"]["attested"] is True
    assert borrowed["reproducibility"]["attested"] is False


# -- the fallback still works --


def test_a_run_without_provenance_uses_the_supervisor(
    switched_supervisor: None,
) -> None:
    """Snapshots taken before this field existed still save, with
    exactly the behavior they had before."""
    meta = _build_metadata(_request(provenance=None))

    assert meta["processor"] == "GPU"
    assert meta["reproducibility"]["versions"] == OTHER_VERSIONS
    assert meta["reproducibility"]["tokenizer"] == OTHER_TOKENIZER
    assert meta["context"]["context_length"] == 65_536


def test_the_context_window_falls_back_when_unattested(
    switched_supervisor: None,
) -> None:
    assert _context_metadata(1240, None) == {
        "prompt_tokens": 1240,
        "context_length": 65_536,
    }


def test_an_attested_run_without_a_window_omits_it(
    switched_supervisor: None,
) -> None:
    """Absent, not borrowed. A checkpoint that reported no window
    must not inherit the resident model's ceiling, which is exactly
    the kind of plausible wrong number this finding is about."""
    block = _context_metadata(
        1240, _provenance(context_length=None)
    )

    assert block == {"prompt_tokens": 1240}


# -- the label follows the frames --


def test_the_attested_model_wins_over_the_claimed_one(
    switched_supervisor: None,
) -> None:
    """The client says one thing, the worker that generated the
    frames says another. The worker was there."""
    meta = _build_metadata(
        _request(
            model="SmolLM3-3B",
            provenance=_provenance(
                model_id="LLaDA-8B-Instruct"
            ),
        )
    )

    assert meta["backend"] == "LLaDA-8B-Instruct"


def test_a_mismatch_is_recorded_rather_than_refused(
    switched_supervisor: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The run is real and complete. Losing it over a disagreement
    about its label would be the worse outcome, but the
    disagreement is worth a line in the log."""
    with caplog.at_level("WARNING"):
        meta = _build_metadata(
            _request(
                model="SmolLM3-3B",
                provenance=_provenance(
                    model_id="LLaDA-8B-Instruct"
                ),
            )
        )

    assert meta["backend"] == "LLaDA-8B-Instruct"
    assert "SmolLM3-3B" in caplog.text
    assert "LLaDA-8B-Instruct" in caplog.text


def test_agreement_logs_nothing(
    switched_supervisor: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The negative space. Every ordinary save takes this path, so a
    warning here would be noise on every run.

    Scoped to the mismatch wording rather than asserting an empty
    log, because a GPU-less host warns about nvidia-smi on the way
    past and that is not what this test is about.
    """
    with caplog.at_level("WARNING"):
        _build_metadata(_request())

    assert "produced by" not in caplog.text


def test_the_checkpoint_comes_from_the_run(
    switched_supervisor: None,
) -> None:
    """The registry's checkpoint is what that model id points at
    today. The run's is what it pointed at when the run happened."""
    meta = _build_metadata(
        _request(
            provenance=_provenance(
                checkpoint="GSAI-ML/LLaDA-8B-Instruct@abc123"
            )
        )
    )

    assert meta["model"] == "GSAI-ML/LLaDA-8B-Instruct@abc123"


# -- the envelope's own shape --


def test_a_worker_may_attest_more_than_we_declare() -> None:
    """Deliberately not strict, unlike the rest of the save body.
    Workers gain fields ahead of the supervisor, and a save must not
    fail because one of them learned to report something new."""
    envelope = RunProvenance(
        model_id="LLaDA-8B-Instruct",
        device="cpu",
        artifact_revision="deadbeef",
    )

    assert envelope.model_id == "LLaDA-8B-Instruct"
    assert envelope.device == "cpu"


def test_an_envelope_needs_a_model_id() -> None:
    """The one field with nothing sensible to default to: an
    envelope that cannot say which model it describes is not
    provenance."""
    with pytest.raises(ValueError):
        RunProvenance(device="cpu")


def test_an_unattested_device_reads_as_unknown() -> None:
    """Not "GPU", and not the supervisor's guess. A worker that did
    not say must not be read as having said something."""
    meta = _build_metadata(
        _request(provenance=_provenance(device="unknown"))
    )

    assert meta["processor"] == "Unknown"
    assert meta["processor_name"] is None
