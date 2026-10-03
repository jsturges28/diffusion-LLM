"""A save is held to what one run can be, and says so when it is not.

Strategy: drive `/api/save` through the real app with Starlette's test
client, its results directory pointed at a temp directory, so every
case can also check what reached the disk. The byte ceiling is patched
down to a few kilobytes where a case needs a body past it, which
keeps those bodies cheap to build while exercising the same
middleware the real limit runs through.

What this pins is `A2-TRUST-02`. Starlette parses the whole JSON body
before any field is validated, and nothing bounded it, so a single
crafted POST could allocate and write far beyond any run the app can
make. Passing proves a body past its limit is refused with a 413
before it is parsed, whether it declared its length or arrived
chunked; that one at the limit is saved; that a refused save never
leaves anything under the results root; and that a refusal reaches
the page in words, where FastAPI's own answer read as "Save failed:
unknown".

Past the ceiling, each field is held to the most one run of its model
can hold. Those bounds are patched down the same way the ceiling is,
and the real ones are proved roomy enough from the other side: the
largest run each model can make passes them, and fits the ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
from starlette.testclient import TestClient

from src.backends.protocol import (
    CANDIDATE_BUDGET_RECORDS,
    CANDIDATES_PER_POSITION,
    PROMPT_CHARS_MAX,
)
from src.backends.registry import REGISTRY, RunBounds, run_bounds
from src.web import run_store, save_limits, server

# Small enough that a body past it costs nothing to build, large
# enough that an ordinary short run fits beneath it.
SMALL_LIMIT = 4096


@pytest.fixture()
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    return tmp_path


@pytest.fixture()
def client(results: Path) -> TestClient:
    return TestClient(server.app)


@pytest.fixture()
def small_limit(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setitem(
        save_limits.BODY_LIMITS, "/api/save", SMALL_LIMIT
    )
    return SMALL_LIMIT


def _payload(**overrides: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": "llada",
        "prompt": "explain REST",
        "frames": ["frame one", "frame two"],
        "final_text": "hello",
    }
    body.update(overrides)
    return body


def _padded(size: int) -> bytes:
    """An ordinary save as JSON, padded with spaces to ``size`` bytes.

    Trailing whitespace is valid JSON, so the body parses the same at
    any size and only its length differs from case to case.
    """
    raw = json.dumps(_payload()).encode()
    assert len(raw) <= size, "the payload must fit the size asked for"
    return raw + b" " * (size - len(raw))


def _post(client: TestClient, body: Any) -> Any:
    return client.post(
        "/api/save",
        content=body,
        headers={"content-type": "application/json"},
    )


def _written(results: Path) -> list:
    """Everything under the results root, the store's staging area
    included, so a refusal that began a save cannot hide in it."""
    return sorted(path.name for path in results.iterdir())


def _published(results: Path) -> list:
    """The runs a save published, counted as the store counts them,
    since the root also holds the store's own lock and scratch."""
    return sorted(run_store.list_run_ids(results))


# -- the ceiling, before anything is parsed --


def test_a_body_past_the_limit_is_refused(
    client: TestClient, results: Path, small_limit: int
) -> None:
    response = _post(client, _padded(small_limit + 1))

    assert response.status_code == 413
    assert response.json()["success"] is False
    assert "one run can take" in response.json()["message"]
    assert _written(results) == []


def test_a_chunked_body_past_the_limit_is_refused(
    client: TestClient, results: Path, small_limit: int
) -> None:
    """No length to refuse on in advance, so it is counted as it
    arrives and refused at the first byte over."""
    body = _padded(small_limit + 1)

    def chunks() -> Iterator[bytes]:
        for start in range(0, len(body), 1000):
            yield body[start:start + 1000]

    response = _post(client, chunks())

    assert response.status_code == 413
    assert _written(results) == []


def test_a_body_at_the_limit_is_saved(
    client: TestClient, results: Path, small_limit: int
) -> None:
    """The boundary from the permissive side, so a comparison flipped
    either way fails one of the two."""
    response = _post(client, _padded(small_limit))

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    assert len(_published(results)) == 1


def test_the_real_ceiling_holds_an_ordinary_run(
    client: TestClient, results: Path
) -> None:
    response = client.post("/api/save", json=_payload())

    assert response.status_code == 200, response.text


# -- a refusal the page can show --


def test_a_refused_save_says_why(
    client: TestClient, results: Path
) -> None:
    """FastAPI's own 422 carries no ``message``, which the page reads,
    so every refusal used to show as "Save failed: unknown"."""
    body = _payload()
    del body["prompt"]

    response = client.post("/api/save", json=body)

    assert response.status_code == 422
    assert response.json()["success"] is False
    assert response.json()["message"].startswith("prompt:")
    assert _written(results) == []


def test_the_reason_names_the_limit_it_broke(
    client: TestClient, results: Path
) -> None:
    """Candidates past their budget, a limit that predates this file,
    now read as one, and without the request echoed back."""
    candidate = {"h": 1, "c": [{"id": 1, "t": "a", "p": 0.5}]}
    sets = [candidate] * 20_481
    body = _payload(
        candidates={
            "k": 5,
            "stride": 1,
            "frames": [0],
            "segments": [0],
            "sets": [sets],
        }
    )

    response = client.post("/api/save", json=body)

    message = response.json()["message"]
    assert response.status_code == 422
    assert message.startswith("candidates:")
    assert "102400" in message
    assert len(message) < 200, message


# -- each field, against one run of its model --

# Bounds a short payload can pass and cheaply exceed.
TINY = RunBounds(frames_max=3, positions_max=4, frame_positions_max=4)

# What one position's alternatives may hold: the captured candidates,
# then the committed token where they missed it.
ROWS = CANDIDATES_PER_POSITION + 1

TEXT_MAX = save_limits.TOKEN_TEXT_CHARS_MAX
IDENTIFIER_MAX = save_limits.IDENTIFIER_CHARS_MAX
FREEFORM_MAX = save_limits.FREEFORM_JSON_CHARS_MAX

EDIT = {"frame_index": 0, "token_positions": []}


@pytest.fixture()
def tiny_bounds(monkeypatch: pytest.MonkeyPatch) -> RunBounds:
    monkeypatch.setattr(server, "run_bounds", lambda model_id: TINY)
    return TINY


def _record(text: str = " a") -> Dict[str, Any]:
    return {"t": text, "m": False, "id": 1, "c": 0.5, "e": 1.0}


def _alternatives(count: int) -> List[Dict[str, Any]]:
    return [
        {"id": rank, "t": " a", "p": 0.1} for rank in range(count)
    ]


def _no_preview(*args: Any, **kwargs: Any) -> None:
    return None


def _refusal(
    client: TestClient, results: Path, body: Dict[str, Any]
) -> str:
    """Why ``body`` was refused, once it is shown that it was, and
    that nothing of it reached the disk."""
    response = client.post("/api/save", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["success"] is False
    assert _written(results) == []
    return str(response.json()["message"])


PAST_TINY = [
    pytest.param(
        {"frames": ["f"] * 4}, "frames holds 4", id="frames"
    ),
    pytest.param(
        {"frames": ["f"], "frame_tokens": [[_record()] * 5]},
        "frame_tokens[0] holds 5",
        id="a-frame-wider-than-a-canvas",
    ),
    pytest.param(
        {
            "model": "smollm3",
            "frames": None,
            "frame_positions": [_record()] * 5,
        },
        "frame_positions holds 5",
        id="positions",
    ),
    pytest.param(
        {"alternatives": [_alternatives(ROWS + 1)]},
        f"alternatives[0] holds {ROWS + 1}",
        id="alternatives-at-one-position",
    ),
    pytest.param(
        {"remask_edits": [EDIT] * 4},
        "remask_edits holds 4",
        id="edits",
    ),
    pytest.param(
        {"final_text": "x" * (4 * TEXT_MAX + 1)},
        "final_text holds",
        id="final-text",
    ),
    pytest.param(
        {
            "candidates": {
                "k": 5,
                "stride": 1,
                "frames": [0, 1, 2, 3],
                "segments": [0],
                "sets": [[], [], [], []],
            }
        },
        "candidates.frames holds 4",
        id="candidate-frames",
    ),
]


@pytest.mark.parametrize(("overrides", "reason"), PAST_TINY)
def test_a_field_past_its_model_s_bounds_is_refused(
    client: TestClient,
    results: Path,
    tiny_bounds: RunBounds,
    overrides: Dict[str, Any],
    reason: str,
) -> None:
    body = _payload(**overrides)

    message = _refusal(client, results, body)

    assert message.startswith(reason), message
    assert body["model"] in message
    assert len(message) < 200, message


def test_a_run_at_its_model_s_bounds_is_saved(
    client: TestClient, results: Path, tiny_bounds: RunBounds
) -> None:
    """The same fields at their bounds, from the permissive side."""
    frames = ["f"] * TINY.frames_max
    body = _payload(
        frames=frames,
        frame_tokens=[
            [_record()] * TINY.frame_positions_max for _ in frames
        ],
        alternatives=[_alternatives(ROWS)] * TINY.positions_max,
        final_text="x" * (TINY.positions_max * TEXT_MAX),
    )

    response = client.post("/api/save", json=body)

    assert response.status_code == 200, response.text
    assert len(_published(results)) == 1


PAST_A_CAP = [
    pytest.param(
        {"prompt": "x" * (PROMPT_CHARS_MAX + 1)},
        "prompt:",
        id="prompt",
    ),
    pytest.param(
        {
            "frames": ["f"],
            "frame_tokens": [[_record("x" * (TEXT_MAX + 1))]],
        },
        "frame_tokens.0.0.t:",
        id="token-text",
    ),
    pytest.param(
        {"run_token": "x" * (IDENTIFIER_MAX + 1)},
        "run_token:",
        id="run-token",
    ),
    pytest.param(
        {"run_id": "x" * (IDENTIFIER_MAX + 1)},
        "run_id:",
        id="run-id",
    ),
    pytest.param(
        {"model": "x" * (IDENTIFIER_MAX + 1)},
        "model:",
        id="model",
    ),
    pytest.param(
        {"params": {"note": "x" * FREEFORM_MAX}},
        "params holds",
        id="params",
    ),
    pytest.param(
        {
            "provenance": {
                "model_id": "llada",
                "tokenizer": {"note": "x" * FREEFORM_MAX},
            }
        },
        "provenance holds",
        id="provenance",
    ),
]


@pytest.mark.parametrize(("overrides", "reason"), PAST_A_CAP)
def test_a_field_past_its_cap_is_refused(
    client: TestClient,
    results: Path,
    overrides: Dict[str, Any],
    reason: str,
) -> None:
    """Caps that do not depend on the model, at their real values."""
    message = _refusal(client, results, _payload(**overrides))

    assert message.startswith(reason), message
    assert len(message) < 200, message


def test_the_longest_smollm3_run_is_saved(
    client: TestClient, results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real bounds from the permissive side, end to end, on the
    model whose largest run is cheap enough to send whole. Its
    preview is skipped: drawn once the run is published, it is not
    what is bounded here, and it is most of what such a save costs."""
    monkeypatch.setattr(server, "_render_run_gif", _no_preview)
    positions = run_bounds("smollm3").positions_max
    body = _payload(
        model="smollm3",
        frames=None,
        frame_positions=[_record()] * positions,
        alternatives=[_alternatives(ROWS)] * positions,
        final_text=" a" * positions,
    )

    response = client.post("/api/save", json=body)

    assert response.status_code == 200, response.text
    assert len(_published(results)) == 1


def _largest(model_id: str) -> server.SaveRunRequest:
    """The largest edited run ``model_id`` can make, built without
    validation from shared rows, so it costs references rather than
    the hundred megabytes it would as JSON."""
    bounds = run_bounds(model_id)
    record = server.TokenRecord(t=" a", m=False, id=1)
    frame = [record] * bounds.frame_positions_max
    count = bounds.frames_max
    edit = server.RemaskEdit(
        frame_index=0,
        token_positions=list(range(bounds.frame_positions_max)),
    )
    return server.SaveRunRequest.model_construct(
        model=model_id,
        prompt="p",
        final_text=" a" * bounds.positions_max,
        frames=[" a" * bounds.frame_positions_max] * count,
        frame_tokens=[frame] * count,
        original_frame_tokens=[frame] * count,
        per_frame_elapsed=[0.1] * count,
        mean_conf=[0.5] * count,
        canvas_index=[0] * count,
        remask_edits=[edit] * count,
    )


@pytest.mark.parametrize("model_id", ["llada", "diffusiongemma"])
def test_the_largest_diffusion_run_passes_its_bounds(
    model_id: str,
) -> None:
    body = _largest(model_id)

    server._check_run_bounds(body)

    longer = body.model_copy(update={"frames": [*body.frames, "a"]})
    with pytest.raises(ValueError, match="frames holds"):
        server._check_run_bounds(longer)


def _priced(record: Any) -> int:
    """Bytes one record takes in a body, its separating comma too."""
    return len(record.model_dump_json(exclude_none=True)) + 1


def test_the_ceiling_holds_the_largest_run_of_any_model() -> None:
    """An estimate from the bounds, for an edited run carrying both
    layers and both candidate captures at their budget. Records are
    priced at full float precision, which is more than the page
    sends, so the estimate errs large."""
    record = server.TokenRecord(
        t=" word", m=False, id=123_456, c=0.1 / 3, e=1 / 3
    )
    alternative = server.TokenAlternative(
        id=123_456, t=" word", p=1 / 3
    )
    per_record = _priced(record) + len(record.t)
    per_entry = _priced(alternative)
    candidates = 2 * CANDIDATE_BUDGET_RECORDS * per_entry
    prompt = 4 * PROMPT_CHARS_MAX
    for model_id, model in REGISTRY.items():
        bounds = run_bounds(model_id)
        records = bounds.frames_max * bounds.frame_positions_max
        if model.capabilities.generation_shape == "append_only":
            records = bounds.positions_max
        entries = bounds.positions_max * ROWS
        layer = records * per_record + entries * per_entry
        estimate = 2 * layer + candidates + prompt

        assert estimate < save_limits.SAVE_BODY_BYTES_MAX, model_id
