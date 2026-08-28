"""Fetch one repository's weights, as a process that can be killed.

The supervisor spawns this module with its own Python:

    <venv>/bin/python -m src.inference.download_main --repo <id>

A download used to run inside the supervisor, as an asyncio task
delegating to a thread whose helper started a second, daemon thread.
Nothing could reach the fetch: cancelling the task left the thread
running, and closing the app left the daemon downloading until the
process died. Out here it is a child with a pid, so the same
terminate-and-wait ladder that ends a model worker ends a download.

It reports nothing but an exit status, and needs no channel to
report more. Progress was never coming from the downloader:
``hf_download`` measures the cache directory on disk, which the
parent can do just as well while this process does the fetching.

Interrupting it is safe by construction. Partial blobs stay on disk
as ``*.incomplete``, ``is_repo_cached`` already treats a partial
cache as not-cached, and the next attempt resumes rather than
starting over. Nothing here deletes anything: another process may
share the cache, and a valid snapshot in it is not ours to discard.
"""

from __future__ import annotations

import argparse
import os
import sys

# Before the first huggingface_hub import, as in the supervisor and
# the worker entrypoint. Xet bypasses the classic downloader, which
# is what lands bytes in ``blobs`` as ``*.incomplete`` parts, and
# those parts are the only thing the parent's progress sampler can
# see. Without this the bar would sit at zero for the whole fetch.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Exit statuses. The parent turns these back into a message, so they
# are the whole reporting protocol and may only be appended to.
DOWNLOAD_EXIT_OK = 0
DOWNLOAD_EXIT_FAILED = 1
# Neither cached nor reachable, which has a remedy worth naming and
# would otherwise surface as a wall of urllib3 retry text.
DOWNLOAD_EXIT_UNREACHABLE = 3

assert DOWNLOAD_EXIT_OK == 0, "success is zero, as a shell expects"
assert DOWNLOAD_EXIT_FAILED != DOWNLOAD_EXIT_UNREACHABLE, (
    "the parent tells the two failures apart by number alone"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch one model repository's weights."
    )
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    from src.inference.hf_download import _is_unreachable

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(args.repo)
    except BaseException as exc:  # noqa: BLE001 - reported by status.
        # Printed for the log the maintainer reads, not for the
        # parent, which is deliberately reading only the status: a
        # pipe nobody drains is a way to wedge a child.
        print(f"download failed: {exc}", file=sys.stderr)
        if _is_unreachable(exc):
            return DOWNLOAD_EXIT_UNREACHABLE
        return DOWNLOAD_EXIT_FAILED
    return DOWNLOAD_EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
