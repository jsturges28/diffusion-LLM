"""Score tokenizer ids with an existing, identified local KGW key.

This is the detector foundation, not a user-facing authorship tool.
It accepts ids that have already been produced by the tokenizer and
returns the loaded key id, exact null rate, counts, a
normal-approximation z-score,
and either ``scored`` or ``insufficient_evidence``. It never labels
text as AI or human.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.kgw_key import load_key  # noqa: E402
from src.inference.kgw_watermark import (  # noqa: E402
    KgwConfig,
    detect_token_ids,
)


def _arguments(
    arguments: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score tokenizer ids with KGW v1.",
    )
    parser.add_argument(
        "ids",
        help="JSON array path, or - for stdin.",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--tokenizer-fingerprint",
        required=True,
    )
    parser.add_argument(
        "--vocab-size",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--expected-key-id",
        help="Refuse detection unless the loaded key has this id.",
    )
    parser.add_argument(
        "--evidence",
        help=(
            "Optional JSON boolean array. By default the first token "
            "is excluded and every later token is scored."
        ),
    )
    return parser.parse_args(arguments)


def _read_json(path: str) -> object:
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ids(value: object) -> List[int]:
    if not isinstance(value, list):
        raise ValueError("ids must be a JSON array")
    ids: List[int] = []
    for token_id in value:
        if isinstance(token_id, bool):
            raise ValueError("every token id must be an integer")
        if not isinstance(token_id, int):
            raise ValueError("every token id must be an integer")
        ids.append(token_id)
    return ids


def _evidence(value: object, count: int) -> List[bool]:
    if not isinstance(value, list):
        raise ValueError("evidence must be a JSON array")
    if len(value) != count:
        raise ValueError("evidence length differs from ids")
    if not all(isinstance(item, bool) for item in value):
        raise ValueError("every evidence value must be boolean")
    return list(value)


def _default_evidence(count: int) -> List[bool]:
    if count == 0:
        return []
    return [False, *([True] * (count - 1))]


def main(arguments: Optional[Sequence[str]] = None) -> int:
    options = _arguments(arguments)
    token_ids = _ids(_read_json(options.ids))
    evidence = (
        _evidence(_read_json(options.evidence), len(token_ids))
        if options.evidence is not None
        else _default_evidence(len(token_ids))
    )
    key = load_key()
    if (
        options.expected_key_id is not None
        and options.expected_key_id != key.key_id
    ):
        raise ValueError(
            "loaded KGW key id "
            f"{key.key_id} does not match expected "
            f"{options.expected_key_id}"
        )
    config = KgwConfig(
        secret=key.secret,
        key_id=key.key_id,
        model_id=options.model_id,
        tokenizer_fingerprint=options.tokenizer_fingerprint,
        vocab_size=options.vocab_size,
        gamma=options.gamma,
        delta=0.0,
    )
    result = detect_token_ids(
        token_ids,
        evidence,
        config=config,
    )
    payload = {"key_id": key.key_id, **result.as_dict()}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
