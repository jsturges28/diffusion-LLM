import argparse
import ipaddress
import os
import sys

import uvicorn

# Only the environment variable's name, from a module that pulls in
# nothing. Importing the server here instead would resolve its data
# root before --results-dir had been written into the environment,
# which is the one ordering this file has to get right.
from src.web.data_root import RESULTS_DIR_ENV

# Loopback, not 0.0.0.0. The same unauthenticated origin that renders
# the UI can also activate models, submit saves, and permanently
# delete runs, so the default must not be reachable from the network
# a laptop happens to be on. Remote access stays available behind an
# explicit --host, which warns before serving (see _warn_if_exposed).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Discrete Diffusion LLM web UI.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=(
            f"Bind address (default: {DEFAULT_HOST}). Use"
            " 0.0.0.0 to expose the app to your network, which"
            " is unauthenticated."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind port (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Directory holding saved runs and durable UI state"
            " (default: the repository's results/, wherever you"
            f" start from). Also settable as {RESULTS_DIR_ENV}."
        ),
    )
    return parser.parse_args()


def is_loopback(host: str) -> bool:
    """Whether binding to `host` keeps the app machine-local.

    Answered by ``ipaddress`` rather than by string comparison,
    because loopback is the whole 127.0.0.0/8 block plus ``::1``,
    and a check for the one spelling everybody types would call
    127.0.0.2 remote and 0.0.0.0 unknown.

    ``localhost`` is accepted by name since that is what resolves to
    loopback everywhere this runs, and it is the spelling the desktop
    launcher and the README use. Anything unparseable is reported as
    not loopback: a hostname this cannot vouch for should get the
    warning rather than silence.
    """
    assert isinstance(host, str), "host must be a string"
    candidate = host.strip()
    if candidate == "":
        return False
    if candidate.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _warn_if_exposed(host: str) -> None:
    """Say plainly what an explicit non-loopback bind gives away.

    Printed rather than logged, and to stderr, so it survives the
    uvicorn log configuration that starts a moment later and is
    visible even when the app's own logging is quietened.
    """
    if is_loopback(host):
        return
    print(
        f"[warning] Binding to {host} serves this app to your"
        " network.\n"
        "[warning] There is no authentication: anyone who can"
        " reach this port can\n"
        "[warning] load and unload models, save runs, and"
        " permanently delete them.\n"
        f"[warning] Use --host {DEFAULT_HOST} unless you intend"
        " that.",
        file=sys.stderr,
    )


def main() -> None:
    args = parse_args()
    _warn_if_exposed(args.host)
    # Before uvicorn imports the server, which resolves its data
    # root at import time. The flag wins over an inherited value so
    # one command line beats a stale shell export.
    if args.results_dir is not None:
        os.environ[RESULTS_DIR_ENV] = args.results_dir
    uvicorn.run(
        "src.web.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
