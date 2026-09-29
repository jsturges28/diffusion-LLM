"""Llama 3.1's tokenizer, for a model trained on it, without the gate.

The Mamba-3 checkpoint was trained with Llama 3.1's tokenizer, whose
own repository is gated behind a licence acceptance. SmolLM3's
`tokenizer.json`, which the registry already pins, is the same
tokenizer in everything that decides which ids a text becomes: the
text splitting, all 128,000 vocabulary entries, all 280,147 merges,
and begin- and end-of-text at 128000 and 128001. It differs only in
ten reserved special tokens SmolLM3 renamed for its chat format, and
in not prepending begin-of-text itself.

So this builds the tokenizer from the byte-pair model alone. The
file's added tokens are dropped, which means no typed text can ever
become a special token, and begin-of-text is prepended here, as
Llama's own template does. `bpe_fingerprint` hashes exactly the parts
that decide ids, and `load_tokenizer` refuses a file whose
fingerprint is not Llama 3.1's, so a change to SmolLM3's file cannot
quietly feed the model ids it was never trained on.

`Llama31Tokenizer` speaks the part of Hugging Face's tokenizer
interface that the worker shell and the sampler call, and no more:
`encode`, `decode`, calling it on a prompt, `len`, and the fields
`describe_tokenizer` reads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import torch
from torch import Tensor

TOKENIZER_FILE = "tokenizer.json"
BEGIN_OF_TEXT = 128000
END_OF_TEXT = 128001
# `bpe_fingerprint` of Meta's tokenizer.json for Llama-3.1-8B at
# d04e592b, the file whose git blob Meta publishes as f916e710.
# SmolLM3's pinned file gives the same value.
LLAMA31_BPE_FINGERPRINT = (
    "1277bcb60e03df534dfbf92525c92b5979b7613e19c84f821991947e218eb0df"
)

assert BEGIN_OF_TEXT < END_OF_TEXT, "Llama numbers begin before end"
assert len(LLAMA31_BPE_FINGERPRINT) == 64, "a SHA-256 in hex"


class TokenizerMismatchError(ValueError):
    """A tokenizer file that is not the one the model learned from."""


class _Pieces(Protocol):
    @property
    def ids(self) -> List[int]: ...


class BytePairs(Protocol):
    """What this needs of a `tokenizers.Tokenizer`."""

    def encode(
        self, sequence: str, add_special_tokens: bool = True
    ) -> _Pieces: ...

    def decode(self, ids: List[int]) -> str: ...


class Encoding(Dict[str, Tensor]):
    """A prompt's ids in the shape `CompletionTextAdapter` reads: a
    mapping holding `input_ids`, movable to the model's device."""

    def to(self, device: Any) -> "Encoding":
        moved = Encoding()
        for key, value in self.items():
            moved[key] = value.to(device)
        return moved


class Llama31Tokenizer:
    """Llama 3.1's tokenizer as the worker and the sampler call one.

    `vocab_size` is the byte-pair vocabulary, not `len(self)`: that is
    the convention `describe_tokenizer` documents, and the gap between
    the two is where the special ids live.
    """

    is_fast = True  # a Rust `tokenizers` model underneath

    def __init__(
        self, pairs: BytePairs, data: Dict[str, Any], *, source: str
    ) -> None:
        assert source, "say where the file came from"
        self._pairs = pairs
        self._ordinary = len(data["model"]["vocab"])
        self._ids = id_count(data)
        assert 0 < self._ordinary <= self._ids, "specials come last"
        self.name_or_path = source
        self.vocab_size = self._ordinary
        self.bos_token_id = BEGIN_OF_TEXT
        self.eos_token_id = END_OF_TEXT
        self.fingerprint = bpe_fingerprint(data)

    def __len__(self) -> int:
        return self._ids

    def encode(
        self, text: str, add_special_tokens: bool = True
    ) -> List[int]:
        """The byte-pair ids of `text`, after begin-of-text unless
        asked not to add it. The model below is never asked for
        specials of its own: it has none."""
        assert isinstance(text, str), "encode takes a string"
        pieces = self._pairs.encode(text, add_special_tokens=False)
        if add_special_tokens:
            return [BEGIN_OF_TEXT, *pieces.ids]
        return list(pieces.ids)

    def decode(
        self, ids: Sequence[int], skip_special_tokens: bool = False
    ) -> str:
        """Text for `ids`. Special ids have no text in a byte-pair
        model, so they decode to nothing whatever
        `skip_special_tokens` says."""
        ordinary = [int(i) for i in ids if int(i) < self._ordinary]
        return self._pairs.decode(ordinary)

    def __call__(
        self, text: str, return_tensors: str = "pt"
    ) -> Encoding:
        """A prompt as one batch row of ids, begin-of-text first."""
        assert return_tensors == "pt", "only tensors are ever asked"
        encoding = Encoding()
        encoding["input_ids"] = torch.tensor(
            [self.encode(text)], dtype=torch.long
        )
        return encoding


def load_tokenizer(
    path: Path,
    *,
    source: str,
    required: Optional[str] = LLAMA31_BPE_FINGERPRINT,
) -> Llama31Tokenizer:
    """The tokenizer in `path`, a tokenizer.json, checked first.

    Refuses a file whose fingerprint is not `required`. None skips the
    check, which the probe uses because it reports a mismatch as a
    verdict rather than stopping on one.
    """
    from tokenizers import Tokenizer

    data = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = bpe_fingerprint(data)
    if required is not None and fingerprint != required:
        raise TokenizerMismatchError(
            f"{path} is not Llama 3.1's tokenizer: its fingerprint"
            f" is {fingerprint[:16]} where Llama 3.1's is"
            f" {required[:16]}, so the model would read ids it was"
            " never trained on."
        )
    bare = {**data, "added_tokens": [], "post_processor": None}
    pairs = Tokenizer.from_str(json.dumps(bare))
    return Llama31Tokenizer(pairs, data, source=source)


def bpe_fingerprint(data: Dict[str, Any]) -> str:
    """SHA-256 of everything that decides which ids a text becomes:
    the normalizer, the pre-tokenizer, the decoder, the byte-pair
    model with its vocabulary and merges, and the begin and end
    tokens. Other special tokens are left out on purpose; the
    tokenizer never produces them. Merges compare as pairs, because
    tokenizer files store them either as "a b" strings or as lists."""
    model = dict(data["model"])
    model["merges"] = [_merge_pair(pair) for pair in model["merges"]]
    special = {
        token["id"]: token["content"]
        for token in data["added_tokens"]
    }
    essence = {
        "normalizer": data["normalizer"],
        "pre_tokenizer": data["pre_tokenizer"],
        "decoder": data["decoder"],
        "model": model,
        "begin": special.get(BEGIN_OF_TEXT),
        "end": special.get(END_OF_TEXT),
    }
    text = json.dumps(
        essence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def id_count(data: Dict[str, Any]) -> int:
    """How many ids the file defines: the vocabulary and every added
    token after it."""
    added = [token["id"] for token in data["added_tokens"]]
    return 1 + max([*data["model"]["vocab"].values(), *added])


def _merge_pair(merge: Any) -> List[str]:
    if isinstance(merge, str):
        left, right = merge.split(" ", 1)
        return [left, right]
    return list(merge)
