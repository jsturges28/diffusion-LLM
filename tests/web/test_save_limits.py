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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from starlette.testclient import TestClient

from src.web import save_limits, server

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
    return [
        name for name in _written(results) if not name.startswith(".")
    ]


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
