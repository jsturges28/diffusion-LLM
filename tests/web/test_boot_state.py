"""The server answers, in the markup, what the page used to ask for.

Strategy: call the serve-time rewrites directly for the properties
that are about text, and go through `TestClient` for the ones that are
about a route deciding what to send. A fake manager stands in for a
resident worker, because none of this needs a real one.

Two things are being pinned. The Generation nav link ships `hidden`
and used to be revealed by `/api/models`, which cost two `nvidia-smi`
subprocesses to learn one boolean the supervisor already knew, and
moved every link beside it when the answer arrived. And the GPU name,
which cannot change under a running process, is now read once.

What passing proves: a page arrives with its link already in the right
state, the rewrite touches nothing else that happens to be hidden, the
inlined state cannot break out of its script tag, and the probe that
was on every navigation is not on any of them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.web import server

STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "static"
)

LINK_PAGES = ("analytics.html", "settings.html")


class _FakeManager:
    """Just enough manager for the two questions a page asks."""

    def __init__(self, active_id: Optional[str]) -> None:
        self.active_id = active_id
        self.active_device = "cuda"
        self.active_tokenizer: Dict[str, Any] = {"name": active_id}
        self.active_context_length = 65536

    def is_serving(self, model_id: str) -> bool:
        return model_id == self.active_id

    def status(self, model_id: str) -> str:
        if model_id == self.active_id:
            return "active"
        return "idle"

    async def stop(self) -> None:
        """The app's shutdown hook evicts on the way out."""
        self.active_id = None


@pytest.fixture
def resident(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A serving SmolLM3, which is in the registry on any host."""
    monkeypatch.setattr(server, "manager", _FakeManager("smollm3"))
    yield "smollm3"


@pytest.fixture
def idle(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(server, "manager", _FakeManager(None))
    yield None


def _link_tag(html: str) -> str:
    match = re.search(
        r'<a\b[^>]*\bid="link-generation"[^>]*>', html
    )
    assert match is not None, (
        "the Generation link is gone from this page; update this"
        " test rather than deleting it"
    )
    return match.group(0)


def _is_hidden(tag: str) -> bool:
    """The bare attribute, not `aria-hidden`."""
    return re.search(r"\shidden(?=[\s>=])", tag) is not None


# -- the link, revealed where it is decided --


def test_a_resident_model_unhides_the_link() -> None:
    for name in LINK_PAGES:
        html = (STATIC / name).read_text(encoding="utf-8")
        assert _is_hidden(_link_tag(html)), name

        served = server._reveal_generation_link(
            html, resident=True
        )

        assert not _is_hidden(_link_tag(served)), name


def test_no_resident_model_leaves_it_hidden() -> None:
    """The generator is gated on a model, so offering the link with
    nothing loaded would advertise a page that redirects away."""
    for name in LINK_PAGES:
        html = (STATIC / name).read_text(encoding="utf-8")

        served = server._reveal_generation_link(
            html, resident=False
        )

        assert served == html, name


def test_it_unhides_that_link_and_nothing_else() -> None:
    """The negative that matters. `index.html` carries 40-odd hidden
    elements whose visibility is runtime state, and a rewrite loose
    enough to reach one of them would show a control for a run that
    does not exist yet."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    before = html.count(" hidden")

    served = server._reveal_generation_link(html, resident=True)

    assert served.count(" hidden") == before
    assert served == html


def test_a_page_without_the_link_is_untouched() -> None:
    """So the rewrite can run for every page rather than being wired
    per route, which is one fewer thing to remember for a new page."""
    html = (STATIC / "menu.html").read_text(encoding="utf-8")

    assert server._reveal_generation_link(html, resident=True) == (
        html
    )


def test_the_served_page_carries_it_already_revealed(
    resident: str,
) -> None:
    """Through the route, because the rewrite being correct and the
    route calling it are different claims."""
    with TestClient(server.app) as client:
        response = client.get("/analytics.html")

    assert response.status_code == 200
    assert not _is_hidden(_link_tag(response.text))


def test_the_served_page_hides_it_when_nothing_is_loaded(
    idle: None,
) -> None:
    with TestClient(server.app) as client:
        response = client.get("/analytics.html")

    assert response.status_code == 200
    assert _is_hidden(_link_tag(response.text))


def test_the_link_and_the_gate_agree(idle: None) -> None:
    """One predicate feeds both, so a page cannot offer a link to a
    route that turns it away. Checked by behaviour rather than by
    reading the source, since agreeing is the point."""
    with TestClient(server.app) as client:
        page = client.get("/analytics.html")
        gate = client.get("/generate", follow_redirects=False)

    assert _is_hidden(_link_tag(page.text))
    assert gate.status_code == 307


# -- the state inlined beside it --


def test_the_settings_page_names_the_resident_class(
    resident: str,
) -> None:
    """SmolLM3 is autoregressive, and Settings opens its glow preview
    on that class. It used to fetch this, play the diffusion default
    first, and correct itself a moment later."""
    with TestClient(server.app) as client:
        response = client.get("/settings.html")

    state = _boot_state(response.text)
    assert state["active_model_type"] == "autoregressive"


def test_it_names_no_class_when_nothing_is_loaded(
    idle: None,
) -> None:
    with TestClient(server.app) as client:
        response = client.get("/settings.html")

    assert _boot_state(response.text)["active_model_type"] is None


def _boot_state(html: str) -> Dict[str, Any]:
    match = re.search(
        r"window\.__BOOT__=(?P<json>.*?);</script>", html
    )
    assert match is not None, "no boot state was inlined"
    return json.loads(
        match.group("json").replace("\\u003c", "<")
    )


def test_the_generator_carries_what_it_used_to_fetch(
    resident: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves of the chain, in one document. The page fetched
    ui-state, then models in its callback, and could not be drawn
    correctly until both had answered."""
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)

    with TestClient(server.app) as client:
        response = client.get("/generate", follow_redirects=False)

    assert response.status_code == 200
    state = _boot_state(response.text)
    assert isinstance(state["ui_state"], dict)
    assert state["models"]["active"] == "smollm3"
    assert state["models"]["active_device"] == "cuda"

    resident_entry = next(
        m
        for m in state["models"]["models"]
        if m["id"] == "smollm3"
    )
    # The field the parameter column is built from, which is what
    # made the second paint so visible.
    assert resident_entry["param_specs"]


def test_the_generator_payload_costs_no_vram_probe(
    resident: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point. `_models_snapshot` shells out to nvidia-smi
    for free VRAM, and putting that in front of a navigation would
    trade a visible reflow for a wait with the old page still up."""
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    calls: List[str] = []
    monkeypatch.setattr(server, "_gpu_name_cached", None)

    def _record(field: str) -> Optional[str]:
        calls.append(field)
        return "NVIDIA GeForce RTX 4090"

    monkeypatch.setattr(server, "_nvidia_smi_query", _record)

    with TestClient(server.app) as client:
        client.get("/generate", follow_redirects=False)

    assert "memory.free" not in calls
    assert calls == ["name"]


def test_a_model_reads_the_same_either_way(
    resident: str,
) -> None:
    """One builder feeds the inlined payload and the endpoint, so a
    model cannot be described one way in the document and another in
    the response the page refreshes from."""
    inlined = server._models_boot_state()
    endpoint = server._models_snapshot()

    by_id = {m["id"]: m for m in endpoint["models"]}
    for model in inlined["models"]:
        full = by_id[model["id"]]
        for key, value in model.items():
            assert full[key] == value, key


def test_the_inlined_payload_omits_the_probed_fields(
    resident: str,
) -> None:
    """Absent rather than stale or null. A `fits` of False inlined
    from a snapshot that never measured would be a lie the menu
    would happily draw."""
    inlined = server._models_boot_state()

    for model in inlined["models"]:
        assert "vram_headroom_gib" not in model
        assert "fits" not in model
    assert "free_vram_gib" not in inlined


def test_analytics_carries_its_catalog(
    resident: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five endpoints answered this page, three of them before the
    table could draw. The catalog is the one worth inlining: it is
    around 25ms for 240 runs, against a round trip the whole render
    waited on."""
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    run_dir = tmp_path / "2026-09-01_02-29-09_smollm3"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "prompt": "Explain diffusion",
                "model": "SmolLM3-3B",
                "created_at": "2026-09-01T02:29:09",
                "schema_version": 2,
            }
        ),
        encoding="utf-8",
    )

    with TestClient(server.app) as client:
        response = client.get("/analytics.html")

    state = _boot_state(response.text)
    assert [r["run_id"] for r in state["runs"]] == [run_dir.name]
    assert isinstance(state["collections"], list)
    assert state["results_dir"].endswith(tmp_path.name)
    assert isinstance(state["ui_state"], dict)


def test_analytics_does_not_probe_the_gpu_to_open(
    resident: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GPU name is only a fallback for a run that did not record
    its own processor, read in a detail view. Serving the page used
    to spawn three probes: two for the nav link, one for this."""
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    calls: List[str] = []
    monkeypatch.setattr(server, "_gpu_name_cached", None)

    def _record(field: str) -> Optional[str]:
        calls.append(field)
        return "NVIDIA GeForce RTX 4090"

    monkeypatch.setattr(server, "_nvidia_smi_query", _record)

    with TestClient(server.app) as client:
        client.get("/analytics.html")

    assert calls == []


def test_the_state_lands_before_the_scripts_that_read_it() -> None:
    """A blob after `app.js` is a blob `app.js` cannot see."""
    html = "<html><head><title>x</title></head><body>"
    html += '<script src="/app.js"></script></body></html>'

    served = server._inject_boot_state(html, {"a": 1})

    assert served.index("__BOOT__") < served.index("/app.js")


def test_a_value_cannot_close_the_script_tag() -> None:
    """A prompt or a run id reaching the page is text the user chose.
    Escaping `<` keeps `</script>` from ending the block early, and
    the result is still ordinary JSON."""
    served = server._boot_script({"x": "</script><script>bad()"})

    assert "</script><script>bad()" not in served
    assert "\\u003c/script" in served
    payload = served[served.index("=") + 1 : served.rindex(";")]
    assert json.loads(payload)["x"] == "</script><script>bad()"


def test_no_state_means_no_script() -> None:
    """A page that needs nothing should not carry an empty global,
    or every page grows a boot contract it does not have."""
    html = "<html><head></head><body></body></html>"

    assert server._inject_boot_state(html, None) == html
    assert server._inject_boot_state(html, {}) == html


# -- and the probe that used to ride along --


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> Iterator[List[str]]:
    """Record every nvidia-smi field query, with the cache cleared so
    one test's reading cannot answer another's."""
    calls: List[str] = []
    monkeypatch.setattr(server, "_gpu_name_cached", None)

    def _record(field: str) -> Optional[str]:
        calls.append(field)
        return "NVIDIA GeForce RTX 4090"

    monkeypatch.setattr(server, "_nvidia_smi_query", _record)
    yield calls


def test_the_gpu_name_is_read_once(probes: List[str]) -> None:
    """It cannot change under a running process, and it was being
    read on every page load of three pages."""
    first = server._gpu_name()
    for _ in range(10):
        server._gpu_name()

    assert first == "NVIDIA GeForce RTX 4090"
    assert probes == ["name"]


def test_a_failed_read_is_not_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of caching, and the one that bites: a probe can
    lose a race, and remembering that would turn one bad answer into
    a permanent claim of no GPU."""
    calls: List[str] = []
    monkeypatch.setattr(server, "_gpu_name_cached", None)
    answers = [None, "NVIDIA GeForce RTX 4090"]

    def _flaky(field: str) -> Optional[str]:
        calls.append(field)
        return answers[len(calls) - 1]

    monkeypatch.setattr(server, "_nvidia_smi_query", _flaky)

    assert server._gpu_name() is None
    assert server._gpu_name() == "NVIDIA GeForce RTX 4090"
    assert len(calls) == 2


def test_free_vram_is_never_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberately not, and the reason is on the record: a stale
    reading is what let the menu promise headroom that the activation
    then refused."""
    calls: List[str] = []

    def _record(field: str) -> Optional[str]:
        calls.append(field)
        return "16384"

    monkeypatch.setattr(server, "_nvidia_smi_query", _record)

    server._free_vram_gib()
    server._free_vram_gib()

    assert calls == ["memory.free", "memory.free"]


# -- what the pages stopped needing --


def test_neither_page_asks_for_the_model_list_any_more() -> None:
    """The point of the exercise. Both pages loaded `model_client.js`
    for one boolean each; the script tag goes too, or the next reader
    reasonably assumes it is still in use."""
    for name, script in (
        ("analytics", "analytics.js"),
        ("settings", "settings.js"),
    ):
        source = (STATIC / script).read_text(encoding="utf-8")
        markup = (STATIC / f"{name}.html").read_text(
            encoding="utf-8"
        )

        assert "modelClientLoad" not in source, name
        assert "/model_client.js" not in markup, name
