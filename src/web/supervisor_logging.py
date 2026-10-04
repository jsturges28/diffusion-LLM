"""Where the supervisor's own log lines go.

uvicorn configures its own loggers and no others. The supervisor logs
to ``diffusion_supervisor`` (``server.py``, ``model_manager.py``,
``worker_process.py``), and ``model_lease.py`` logs under its module
name, so with nothing else configured neither had a handler. Python's
last-resort handler printed their warnings and errors, unformatted,
and every INFO line was dropped: the data root resolved at startup,
each worker spawned and stopped, each run saved or deleted.

Both launchers hand uvicorn this configuration in place of its
default. It is uvicorn's own with the supervisor's loggers added to
the same stderr handler, so their lines read like uvicorn's. It
imports nothing from the server, because ``main.py`` builds it before
uvicorn imports the server, and the server resolves its data root
when it is imported.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

from uvicorn.config import LOGGING_CONFIG

# The supervisor's own logger, and the package that the lease
# module's module-named logger sits under.
SUPERVISOR_LOGGERS: Tuple[str, ...] = (
    "diffusion_supervisor",
    "src.web",
)


def supervisor_log_config() -> Dict[str, Any]:
    """uvicorn's logging configuration with the supervisor's added.

    A deep copy, so uvicorn's module-level dict is never changed for
    anything else in the process that runs uvicorn.
    """
    config: Dict[str, Any] = copy.deepcopy(LOGGING_CONFIG)
    loggers: Dict[str, Any] = config["loggers"]
    assert "default" in config["handlers"], "uvicorn's stderr handler"
    for name in SUPERVISOR_LOGGERS:
        assert name not in loggers, f"uvicorn configures {name}"
        loggers[name] = {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        }
    return config
