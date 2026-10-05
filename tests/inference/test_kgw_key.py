"""KGW key persistence, override, permissions, and off-path tests.

Strategy: isolate every filesystem case under pytest's temporary
directory, then drive the real append-only validator with its default
watermark switch. Passing proves the durable key is created once with
owner-only permissions, an explicit experiment key bypasses disk, and
an ordinary run cannot accidentally read or create key material.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.detect_kgw import main as detect_main
from src.backends import append_only_backend
from src.backends.smollm3_worker import Smollm3Backend
from src.inference import kgw_key
from src.inference.kgw_key import (
    KGW_KEY_ENV,
    KGW_KEY_FILE,
    KGW_STATE_DIRECTORY,
    WatermarkKey,
    key_id,
    load_key,
    load_or_create_key,
)


def test_key_is_generated_once_with_mode_0600(
    tmp_path: Path,
) -> None:
    first = load_or_create_key(environ={}, state_home=tmp_path)
    second = load_or_create_key(environ={}, state_home=tmp_path)
    path = tmp_path / KGW_STATE_DIRECTORY / KGW_KEY_FILE

    assert first.secret == second.secret
    assert first.key_id == second.key_id
    assert len(first.secret) == 32
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_key_lives_below_xdg_state_home(tmp_path: Path) -> None:
    load_or_create_key(
        environ={"XDG_STATE_HOME": str(tmp_path)}
    )
    path = tmp_path / KGW_STATE_DIRECTORY / KGW_KEY_FILE

    assert path.is_file()
    assert path.stat().st_size == 32
    assert [entry.name for entry in path.parent.iterdir()] == [
        KGW_KEY_FILE
    ]


def test_explicit_hex_key_never_touches_state(
    tmp_path: Path,
) -> None:
    secret = bytes(range(32))
    loaded = load_or_create_key(
        environ={KGW_KEY_ENV: secret.hex()},
        state_home=tmp_path,
    )

    assert loaded.secret == secret
    assert loaded.key_id == key_id(secret)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "value",
    [
        "00",
        "z" * 64,
        "00" * 33,
    ],
)
def test_invalid_override_is_refused(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ValueError):
        load_or_create_key(
            environ={KGW_KEY_ENV: value},
            state_home=tmp_path,
        )


def test_existing_shared_key_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / KGW_STATE_DIRECTORY
    directory.mkdir()
    directory.chmod(0o700)
    path = directory / KGW_KEY_FILE
    path.write_bytes(bytes(range(32)))
    path.chmod(0o644)

    with pytest.raises(PermissionError, match="0600"):
        load_or_create_key(environ={}, state_home=tmp_path)


def test_existing_shared_directory_is_refused(
    tmp_path: Path,
) -> None:
    directory = tmp_path / KGW_STATE_DIRECTORY
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)

    with pytest.raises(PermissionError, match="0700"):
        load_or_create_key(environ={}, state_home=tmp_path)


def test_key_publication_fsyncs_containing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced: list[Path] = []
    monkeypatch.setattr(
        kgw_key,
        "_fsync_directory",
        lambda path: synced.append(path),
    )

    load_or_create_key(environ={}, state_home=tmp_path)

    directory = tmp_path / KGW_STATE_DIRECTORY
    assert tmp_path in synced
    assert directory in synced


def test_load_only_never_creates_a_missing_key(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_key(environ={}, state_home=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_detector_never_creates_a_missing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    ids_path = tmp_path / "ids.json"
    ids_path.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        detect_main(
            [
                str(ids_path),
                "--model-id",
                "smollm3",
                "--tokenizer-fingerprint",
                "ab" * 32,
                "--vocab-size",
                "16",
            ]
        )

    assert not (tmp_path / KGW_STATE_DIRECTORY).exists()


def test_detector_reports_and_checks_existing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    key = load_or_create_key(environ={}, state_home=tmp_path)
    ids_path = tmp_path / "ids.json"
    ids_path.write_text("[1, 2]", encoding="utf-8")
    arguments = [
        str(ids_path),
        "--model-id",
        "smollm3",
        "--tokenizer-fingerprint",
        "ab" * 32,
        "--vocab-size",
        "16",
        "--expected-key-id",
        key.key_id,
    ]

    assert detect_main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["key_id"] == key.key_id
    assert output["p0"] == pytest.approx(0.25)

    wrong = [*arguments[:-1], "0" * 16]
    with pytest.raises(ValueError, match="does not match"):
        detect_main(wrong)


def test_watermark_off_never_loads_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    def forbidden() -> Any:
        calls.append(object())
        raise AssertionError("off path touched the KGW key")

    monkeypatch.setattr(
        append_only_backend, "load_or_create_key", forbidden
    )
    backend = Smollm3Backend()

    params = backend._validate_generate({"prompt": "plain run"})

    assert params["watermark"] is False
    assert params["_watermark"] is None
    assert calls == []


def _loaded_backend() -> Smollm3Backend:
    backend = Smollm3Backend()
    backend.model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=128_256)
    )
    backend.tokenizer = SimpleNamespace(fingerprint="ab" * 32)
    backend.effective_device = "cpu"
    return backend


def test_enabled_watermark_requires_experimental_before_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    def forbidden() -> Any:
        calls.append(object())
        raise AssertionError("key loaded before validation")

    monkeypatch.setattr(
        append_only_backend, "load_or_create_key", forbidden
    )

    with pytest.raises(ValueError, match="Experimental"):
        _loaded_backend()._validate_generate(
            {"prompt": "marked", "watermark": True}
        )

    assert calls == []


def test_enabled_watermark_loads_key_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = WatermarkKey(bytes(range(32)), "0123456789abcdef")
    calls: list[Any] = []

    def load() -> WatermarkKey:
        calls.append(object())
        return key

    monkeypatch.setattr(
        append_only_backend, "load_or_create_key", load
    )
    params = _loaded_backend()._validate_generate(
        {
            "prompt": "marked",
            "watermark": True,
            "experimental": True,
            "watermark_gamma": 0.3,
            "watermark_delta": 1.5,
        }
    )

    watermark = params["_watermark"]
    assert watermark.config.gamma == pytest.approx(0.3)
    assert watermark.config.delta == pytest.approx(1.5)
    assert len(calls) == 1
