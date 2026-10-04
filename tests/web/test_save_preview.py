"""A save answers before its preview is drawn, and a preview of a run
that has moved on is never published.

Strategy: call the save route directly, with a ``BackgroundTasks`` of
the test's own, so the reply and the draw can be watched apart.
Starlette's test client runs a response's background tasks before it
returns, which is why the other save tests are unaffected and why this
file does not go through it. The drawer is a stub that writes the
revision it was asked to draw, so a published preview says which
revision it shows. ``run_store.publish_preview`` is then driven on its
own, including against the publication lock held by another process.

Drawing a long run takes seconds, about eleven at 256 frames, and the
save used to make the page wait through all of it. The draw also ran
outside the publication lock, and ``publish`` empties a run's folder
on every replacement, so a draw of the old revision finishing late
landed in the new revision's folder.

What passing proves: the reply carries the saved run while nothing is
drawn yet, with one draw queued behind it; the preview describes the
run as it was saved; a draw for a revision that was replaced or
deleted is discarded, whichever draw finishes first; a failed draw
costs only the preview; and publishing waits for a save or a delete
in another supervisor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from multiprocessing.synchronize import Event as ProcessEvent
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastapi import BackgroundTasks
from starlette.testclient import TestClient

from src.backends.registry import REGISTRY
from src.web import run_store, save_pipeline, server

from process_race import race_context

LOGGER = "diffusion_supervisor"


@pytest.fixture()
def results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        save_pipeline, "_render_run_gif", _draw_revision
    )
    return tmp_path


def _draw_revision(
    preview: save_pipeline.RunPreview, path: Path
) -> None:
    """Stands in for the real drawer, which takes seconds and writes
    pixels, by writing which revision it was asked to draw."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"revision {preview.revision}", encoding="utf-8")


def _payload(**overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": "llada",
        "prompt": "explain REST",
        "frames": ["frame one", "frame two"],
        "final_text": "hello",
    }
    payload.update(overrides)
    return payload


def _body(**overrides: Any) -> save_pipeline.SaveRunRequest:
    return save_pipeline.SaveRunRequest(**_payload(**overrides))


def _save(
    body: save_pipeline.SaveRunRequest,
) -> Tuple[Dict[str, Any], BackgroundTasks]:
    """The save route's reply, and the tasks it left to run after."""
    tasks = BackgroundTasks()
    response = asyncio.run(server.save_run(body, tasks))
    assert response.status_code == 200, response.body
    reply = json.loads(response.body)
    assert reply["success"] is True
    return reply, tasks


def _edit(
    reply: Dict[str, Any],
) -> Tuple[Dict[str, Any], BackgroundTasks]:
    """Replace the run ``reply`` describes with an edited version."""
    edited, tasks = _save(
        _body(
            run_id=reply["run_id"],
            expected_revision=reply["revision"],
            final_text="hello, edited",
        )
    )
    assert edited["run_id"] == reply["run_id"]
    assert edited["revision"] == reply["revision"] + 1
    return edited, tasks


def _run(tasks: BackgroundTasks) -> None:
    asyncio.run(tasks())


def _preview(root: Path, run_id: str) -> Path:
    return root / run_id / run_store.PREVIEW_NAME


def _staged(root: Path) -> List[Path]:
    """Whatever is left in the staging area."""
    staging = root / run_store.STAGING_DIR_NAME
    if not staging.is_dir():
        return []
    return sorted(staging.iterdir())


# -- the reply --


def test_the_reply_comes_before_the_preview(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drawn: List[int] = []

    def draw(
        preview: save_pipeline.RunPreview, path: Path
    ) -> None:
        drawn.append(preview.revision)
        _draw_revision(preview, path)

    monkeypatch.setattr(save_pipeline, "_render_run_gif", draw)

    reply, tasks = _save(_body())

    assert drawn == []
    assert len(tasks.tasks) == 1
    assert not _preview(results, reply["run_id"]).exists()

    _run(tasks)

    assert drawn == [1]
    preview = _preview(results, reply["run_id"])
    assert preview.read_text(encoding="utf-8") == "revision 1"
    assert _staged(results) == []


def test_the_preview_describes_the_run_as_saved(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Taken at save time, because by the time it is drawn the
    request is gone; the label still comes from the metadata just
    written rather than off the request."""
    seen: List[save_pipeline.RunPreview] = []

    def draw(
        preview: save_pipeline.RunPreview, path: Path
    ) -> None:
        seen.append(preview)
        _draw_revision(preview, path)

    monkeypatch.setattr(save_pipeline, "_render_run_gif", draw)

    reply, tasks = _save(_body())
    _run(tasks)

    assert len(seen) == 1
    preview = seen[0]
    assert preview.root == results
    assert preview.run_id == reply["run_id"]
    assert preview.revision == reply["revision"]
    assert preview.frames == ("frame one", "frame two")
    assert preview.prompt == "explain REST"
    label = REGISTRY["llada"].display_name
    assert preview.model_label == label
    assert preview.model_type == "diffusion"


def test_the_real_drawer_publishes_a_gif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the route with the real drawer, which the test client
    runs before it returns. Every other test here stubs the drawer,
    and a draw that fails is only logged, so without this a preview
    handed to it wrongly would never be noticed."""
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    client = TestClient(server.app)

    response = client.post("/api/save", json=_payload())

    assert response.status_code == 200, response.text
    gif = _preview(tmp_path, response.json()["run_id"])
    assert gif.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
    assert _staged(tmp_path) == []


def test_a_refused_save_queues_no_preview(results: Path) -> None:
    reply, _tasks = _save(_body())
    tasks = BackgroundTasks()
    stale = _body(run_id=reply["run_id"], expected_revision=0)

    response = asyncio.run(server.save_run(stale, tasks))

    assert response.status_code == 409
    assert tasks.tasks == []


# -- a draw for a run that has moved on --


def test_a_draw_for_a_replaced_revision_is_dropped(
    results: Path,
) -> None:
    """Revision 1's draw finishes after revision 2 is saved, before
    revision 2's draw does."""
    first, first_tasks = _save(_body())
    second, second_tasks = _edit(first)
    preview = _preview(results, first["run_id"])

    _run(first_tasks)

    assert not preview.exists()

    _run(second_tasks)

    assert preview.read_text(encoding="utf-8") == "revision 2"
    assert _staged(results) == []


def test_a_late_draw_does_not_replace_the_newer_preview(
    results: Path,
) -> None:
    """Revision 2's preview lands first and revision 1's finishes
    after it, which is the order that left a run showing a picture
    of text it no longer had."""
    first, first_tasks = _save(_body())
    _second, second_tasks = _edit(first)

    _run(second_tasks)
    _run(first_tasks)

    preview = _preview(results, first["run_id"])
    assert preview.read_text(encoding="utf-8") == "revision 2"
    assert _staged(results) == []


def test_a_deleted_runs_draw_writes_nothing(
    results: Path, caplog: pytest.LogCaptureFixture
) -> None:
    reply, tasks = _save(_body())
    run_store.delete(results, reply["run_id"])

    with caplog.at_level(logging.INFO, logger=LOGGER):
        _run(tasks)

    assert not (results / reply["run_id"]).exists()
    assert run_store.list_run_ids(results) == []
    assert _staged(results) == []
    assert "dropped the preview" in caplog.text
    assert "GIF rendering failed" not in caplog.text


def test_a_failed_draw_costs_only_the_preview(
    results: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def draw(
        preview: save_pipeline.RunPreview, path: Path
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("half a picture", encoding="utf-8")
        raise RuntimeError("the renderer broke")

    monkeypatch.setattr(save_pipeline, "_render_run_gif", draw)
    reply, tasks = _save(_body())

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        _run(tasks)

    run_id = reply["run_id"]
    assert f"GIF rendering failed for {run_id}" in caplog.text
    assert run_store.list_run_ids(results) == [run_id]
    assert not _preview(results, run_id).exists()
    assert _staged(results) == []


# -- publishing --


def _saved(root: Path) -> Tuple[str, int]:
    bundle = run_store.RunBundle(
        metadata={"backend": "llada", "prompt": "p"},
        final_text="hello",
        frames=["frame one", "frame two"],
    )
    return run_store.save(root, bundle, model_id="llada")


def _drawn(root: Path, run_id: str, revision: int) -> Path:
    staged = run_store.preview_staging_path(root, run_id, revision)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(f"revision {revision}", encoding="utf-8")
    return staged


def test_each_draw_gets_its_own_staged_file(tmp_path: Path) -> None:
    """Two draws of one revision must not share a file, for the
    reason ``run_store.stage`` gives for bundles."""
    first = run_store.preview_staging_path(tmp_path, "run-a", 1)
    second = run_store.preview_staging_path(tmp_path, "run-a", 1)

    assert first != second
    assert first.parent == tmp_path / run_store.STAGING_DIR_NAME
    assert second.parent == first.parent


def test_a_preview_of_a_deleted_run_is_discarded(
    tmp_path: Path,
) -> None:
    run_id, revision = _saved(tmp_path)
    staged = _drawn(tmp_path, run_id, revision)
    run_store.delete(tmp_path, run_id)

    published = run_store.publish_preview(
        tmp_path, run_id, revision=revision, staged=staged
    )

    assert published is False
    assert not staged.exists()
    assert not (tmp_path / run_id).exists()


def _hold_publication(
    root: Path, holding: ProcessEvent, release: ProcessEvent
) -> None:
    """Another supervisor, part way through a save or a delete."""
    with run_store._PUBLISH_LOCK.held(root):
        holding.set()
        assert release.wait(timeout=30), "never released"


def test_a_preview_waits_for_a_save_in_another_process(
    tmp_path: Path,
) -> None:
    run_id, revision = _saved(tmp_path)
    staged = _drawn(tmp_path, run_id, revision)
    context = race_context()
    holding = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_publication, args=(tmp_path, holding, release)
    )
    outcome: List[bool] = []

    def publish() -> None:
        outcome.append(
            run_store.publish_preview(
                tmp_path, run_id, revision=revision, staged=staged
            )
        )

    publisher = threading.Thread(target=publish, daemon=True)
    holder.start()
    try:
        assert holding.wait(timeout=10), "the lock was never taken"
        publisher.start()
        publisher.join(timeout=0.3)
        waited = publisher.is_alive()
    finally:
        release.set()
        holder.join(timeout=10)
    publisher.join(timeout=10)

    assert waited, "published while another process held the lock"
    assert outcome == [True]
    assert holder.exitcode == 0
    shown = _preview(tmp_path, run_id).read_text(encoding="utf-8")
    assert shown == f"revision {revision}"
