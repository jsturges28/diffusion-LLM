"""The supervisor's own log lines reach the terminal.

Strategy: read the configuration both launchers hand uvicorn, and
apply it in a child interpreter, so this process's logging is left as
it was. uvicorn configures its own loggers and nothing else, so before
this every line the supervisor logged at INFO was dropped: the data
root it resolved at startup, each worker it spawned and stopped, each
run saved or deleted. Warnings still appeared, unformatted, through
Python's last-resort handler. The suite never noticed, because
pytest's log capture attaches a handler of its own.

Passing proves the configuration gives the supervisor's loggers a
handler at INFO and leaves uvicorn's own as they were, without
changing uvicorn's module-level dict; that a child which applies it
prints a supervisor INFO line, and one the lease module logs under its
own name; that a child which does not prints neither, which is the
defect; and that the desktop app passes it as the browser launcher
does (``tests/test_main.py``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from uvicorn.config import LOGGING_CONFIG

from src.web.supervisor_logging import (
    SUPERVISOR_LOGGERS,
    supervisor_log_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

SAVED = "saved run to results/run-a"
LEASE = "lease taken for llada"


def _child_stderr(*, configured: bool) -> str:
    """What a child interpreter prints when the supervisor and the
    lease module each log one INFO line."""
    lines = ["import logging", "import logging.config"]
    if configured:
        lines.append(
            "from src.web.supervisor_logging import"
            " supervisor_log_config"
        )
        lines.append(
            "logging.config.dictConfig(supervisor_log_config())"
        )
    lines.append(
        f"logging.getLogger('diffusion_supervisor').info({SAVED!r})"
    )
    lines.append(
        f"logging.getLogger('src.web.model_lease').info({LEASE!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", "\n".join(lines)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return result.stderr


# -- the configuration --


def test_the_supervisors_loggers_get_a_handler() -> None:
    config = supervisor_log_config()

    for name in SUPERVISOR_LOGGERS:
        logger = config["loggers"][name]
        assert logger["handlers"] == ["default"], name
        assert logger["level"] == "INFO", name
        assert logger["propagate"] is False, name
    assert "default" in config["handlers"]


def test_uvicorns_own_loggers_are_left_alone() -> None:
    config = supervisor_log_config()

    for name, settings in LOGGING_CONFIG["loggers"].items():
        assert config["loggers"][name] == settings, name
    assert config["handlers"] == LOGGING_CONFIG["handlers"]
    assert config["formatters"] == LOGGING_CONFIG["formatters"]


def test_the_configuration_is_a_copy() -> None:
    """uvicorn's dict is module state, so changing it in place would
    reach every other use of uvicorn in the process."""
    config = supervisor_log_config()
    config["loggers"]["uvicorn"]["level"] = "DEBUG"

    assert LOGGING_CONFIG["loggers"]["uvicorn"]["level"] == "INFO"
    for name in SUPERVISOR_LOGGERS:
        assert name not in LOGGING_CONFIG["loggers"], name


# -- what reaches the terminal --


def test_a_supervisor_info_line_reaches_the_terminal() -> None:
    stderr = _child_stderr(configured=True)

    assert f"INFO:     {SAVED}" in stderr
    assert f"INFO:     {LEASE}" in stderr


def test_without_it_both_lines_are_lost() -> None:
    """The defect, kept as a pin: the same two lines with nothing
    configured print nothing at all."""
    stderr = _child_stderr(configured=False)

    assert SAVED not in stderr
    assert LEASE not in stderr


# -- the launchers --


def test_the_desktop_app_passes_it_too() -> None:
    """Read rather than run: starting the desktop app opens a
    window, and the browser launcher's check runs ``main()``."""
    source = (REPO_ROOT / "desktop.py").read_text(encoding="utf-8")

    assert "log_config=supervisor_log_config()" in source
