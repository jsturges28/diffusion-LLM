"""The download API, over HTTP.

Strategy: FastAPI's TestClient against the real app with the
manager's download half stubbed, mirroring
`test_activation_identity.py`. The manager's own behavior is covered
in `test_download_ownership.py`; what these pin is the wire: the
shapes a browser reads, the status codes it branches on, and the
operation number that keeps two windows apart.

No test touched any download endpoint before `TRUST-04`, so these
are the first, and the cancel endpoint they cover did not exist.

Passing proves a start reports its operation, the poll carries the
number a client fences on, a cancel is accepted for the current
download and refused with 409 for a stale one, and the refusal says
something a user can act on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.web.server import ActivationRefused, app, manager

STATUS_URL = "/api/models/download-status"
CANCEL_URL = "/api/models/download/cancel"


class _StubDownloads:
    """The manager's download half, without a child process.

    Writes the manager's own fields rather than shadowing them, so
    the endpoints under test read exactly what they read in
    production. `monkeypatch` puts the originals back, which matters
    because the app's manager is a module-level singleton.
    """

    def __init__(self) -> None:
        self.operation = 0
        self.cancelled: List[Optional[int]] = []
        self.refuse = False

    def start(self, model_id: str) -> int:
        self.operation += 1
        manager.download_id = self.operation
        manager.download_state = "downloading"
        manager.download_target = model_id
        return self.operation

    async def cancel(
        self, operation: Optional[int] = None
    ) -> None:
        self.cancelled.append(operation)
        if self.refuse:
            raise ActivationRefused(
                "That download has already finished or belongs to"
                " another window."
            )
        manager.download_state = "idle"
        manager.download_target = None


@pytest.fixture()
def downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> _StubDownloads:
    stub = _StubDownloads()
    monkeypatch.setattr(manager, "start_download", stub.start)
    monkeypatch.setattr(manager, "cancel_download", stub.cancel)
    monkeypatch.setattr(manager, "download_state", "idle")
    monkeypatch.setattr(manager, "download_target", None)
    monkeypatch.setattr(manager, "download_id", 0)
    monkeypatch.setattr(manager, "download_progress", None)
    monkeypatch.setattr(manager, "download_error", None)
    return stub


@pytest.fixture()
def client() -> Any:
    with TestClient(app) as test_client:
        yield test_client


def _start(client: Any, model_id: str = "llada") -> Dict[str, Any]:
    response = client.post(
        f"/api/models/{model_id}/download"
    )
    return dict(response.json())


# -- starting --


def test_a_start_reports_the_operation_it_began(
    client: Any, downloads: _StubDownloads
) -> None:
    """The client fences on this, so it has to come back from the
    call that created it rather than from a poll that might already
    describe somebody else's."""
    body = _start(client)

    assert body["ok"] is True
    assert body["operation"] == downloads.operation


def test_an_unknown_model_is_a_404(client: Any) -> None:
    response = client.post("/api/models/nope/download")

    assert response.status_code == 404


# -- polling --


def test_the_poll_carries_the_operation(
    client: Any, downloads: _StubDownloads
) -> None:
    _start(client)

    status = client.get(STATUS_URL).json()

    assert status["operation"] == downloads.operation
    assert status["state"] == "downloading"


def test_the_poll_names_the_model_being_fetched(
    client: Any, downloads: _StubDownloads
) -> None:
    _start(client)

    status = client.get(STATUS_URL).json()

    assert status["target"] == "llada"
    assert status["target_name"]


# -- cancelling --


def test_a_cancel_reaches_the_manager(
    client: Any, downloads: _StubDownloads
) -> None:
    started = _start(client)

    response = client.post(
        CANCEL_URL, json={"operation": started["operation"]}
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert downloads.cancelled == [started["operation"]]


def test_a_cancel_without_a_body_is_allowed(
    client: Any, downloads: _StubDownloads
) -> None:
    """A page that never learned the number can still stop what it
    can see, the way the activation cancel allows."""
    _start(client)

    response = client.post(CANCEL_URL)

    assert response.status_code == 200
    assert downloads.cancelled == [None]


def test_a_stale_cancel_is_refused_with_409(
    client: Any, downloads: _StubDownloads
) -> None:
    _start(client)
    downloads.refuse = True

    response = client.post(CANCEL_URL, json={"operation": 1})

    assert response.status_code == 409
    assert response.json()["ok"] is False


def test_the_refusal_says_something_actionable(
    client: Any, downloads: _StubDownloads
) -> None:
    """A 409 with no sentence leaves the user staring at a bar that
    will not stop."""
    _start(client)
    downloads.refuse = True

    body = client.post(
        CANCEL_URL, json={"operation": 1}
    ).json()

    assert "another window" in body["message"]


def test_an_unexpected_field_is_rejected(client: Any) -> None:
    """`extra="forbid"`, so a typo in the body is a 422 rather than
    a cancel that silently ignores what it was told."""
    response = client.post(
        CANCEL_URL, json={"operatoin": 3}
    )

    assert response.status_code == 422
