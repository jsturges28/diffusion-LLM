"""Llama 3.1's tokenizer, as the worker and the probe share it.

Strategy: the fingerprint is pure JSON, so hand-made files show what
it reads and what it deliberately ignores. The tokenizer's own logic
wraps a byte-pair model, so a stand-in model shows exactly what it
asks of one and what it adds. A tiny real `tokenizers` model, built
here with whatever version is installed, shows a file loads with its
added tokens dropped. Where SmolLM3's pinned file is cached, the
recorded fingerprint is held to it. Passing proves the tokenizer adds
begin-of-text once and nothing else, that no typed text becomes a
special token, and that a file with other ids is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

import pytest
import torch

from src.backends.registry import SMOLLM3
from src.inference.mamba3_tokenizer import (
    BEGIN_OF_TEXT,
    END_OF_TEXT,
    LLAMA31_BPE_FINGERPRINT,
    TOKENIZER_FILE,
    Llama31Tokenizer,
    TokenizerMismatchError,
    bpe_fingerprint,
    id_count,
    load_tokenizer,
)


def _tokenizer_file(strings: bool = False) -> Dict[str, Any]:
    """The parts of a tokenizer.json the fingerprint reads, small.
    `strings` writes the merges the way Meta's file does."""
    merges: List[Any] = [["a", "b"], ["ab", "c"]]
    if strings:
        merges = ["a b", "ab c"]
    return {
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
            "merges": merges,
        },
        "added_tokens": [
            {"id": BEGIN_OF_TEXT, "content": "<|begin_of_text|>"},
            {"id": END_OF_TEXT, "content": "<|end_of_text|>"},
            {"id": END_OF_TEXT + 1, "content": "<|reserved_0|>"},
        ],
    }


class _Pieces(NamedTuple):
    ids: List[int]


class _BytePairs:
    """Word lengths as ids, recording whether specials were asked for.
    Stands in for `tokenizers.Tokenizer`, whose version in `.venv`
    cannot read SmolLM3's file."""

    def __init__(self) -> None:
        self.asked_for_specials: List[bool] = []

    def encode(
        self, sequence: str, add_special_tokens: bool = True
    ) -> _Pieces:
        self.asked_for_specials.append(add_special_tokens)
        return _Pieces([len(word) for word in sequence.split()])

    def decode(self, ids: List[int]) -> str:
        return ",".join(str(token) for token in ids)


def _tokenizer(pairs: _BytePairs) -> Llama31Tokenizer:
    return Llama31Tokenizer(pairs, _tokenizer_file(), source="test")


# -- the fingerprint --


def test_the_fingerprint_reads_both_merge_formats_alike() -> None:
    """Meta's file writes merges as "a b" strings and SmolLM3's as
    lists; the same rules must give the same fingerprint."""
    strings = bpe_fingerprint(_tokenizer_file(True))
    lists = bpe_fingerprint(_tokenizer_file(False))

    assert strings == lists


def test_the_fingerprint_changes_with_anything_moving_an_id() -> None:
    base = bpe_fingerprint(_tokenizer_file())
    reordered = _tokenizer_file()
    reordered["model"]["merges"].reverse()
    renumbered = _tokenizer_file()
    renumbered["model"]["vocab"]["abc"] = 5
    renamed_end = _tokenizer_file()
    renamed_end["added_tokens"][1]["content"] = "<|im_end|>"

    for changed in (reordered, renumbered, renamed_end):
        assert bpe_fingerprint(changed) != base


def test_the_fingerprint_ignores_tokens_never_emitted() -> None:
    """SmolLM3 renamed ten reserved tokens for its chat format. The
    tokenizer drops every added token, so they must not count."""
    base = bpe_fingerprint(_tokenizer_file())
    renamed = _tokenizer_file()
    renamed["added_tokens"][2]["content"] = "<think>"

    assert bpe_fingerprint(renamed) == base


def test_the_pinned_smollm3_tokenizer_is_llamas() -> None:
    """Holds the recorded fingerprint to the real file, on any
    machine that has run SmolLM3 and so has it cached."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(
        SMOLLM3.checkpoint,
        TOKENIZER_FILE,
        revision=SMOLLM3.revision,
    )
    if not isinstance(path, str):
        pytest.skip("SmolLM3's tokenizer is not cached here")
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    assert bpe_fingerprint(data) == LLAMA31_BPE_FINGERPRINT
    assert id_count(data) == 128256


# -- the tokenizer around a byte-pair model --


def test_encoding_adds_begin_of_text_once_and_no_other() -> None:
    """Begin-of-text comes from here, once, and the byte pairs are
    never asked for specials of their own."""
    pairs = _BytePairs()

    ids = _tokenizer(pairs).encode("ab c")

    assert ids == [BEGIN_OF_TEXT, 2, 1]
    assert pairs.asked_for_specials == [False]


def test_a_fragment_encodes_without_begin_of_text() -> None:
    """How `tokenize_pieces` asks: a typed word spliced into a
    sequence gains no begin-of-text it was never given."""
    pairs = _BytePairs()

    ids = _tokenizer(pairs).encode("ab c", add_special_tokens=False)

    assert ids == [2, 1]
    assert pairs.asked_for_specials == [False]


def test_decoding_drops_every_id_past_the_vocabulary() -> None:
    """Only the specials live there, and they have no text, so the
    end-of-text a run stops on never reaches the answer."""
    tokenizer = _tokenizer(_BytePairs())

    for skip in (False, True):
        text = tokenizer.decode(
            [BEGIN_OF_TEXT, 3, END_OF_TEXT], skip_special_tokens=skip
        )
        assert text == "3"


def test_a_prompt_becomes_one_row_of_ids() -> None:
    """The shape `CompletionTextAdapter` reads, and it moves."""
    tokenizer = _tokenizer(_BytePairs())

    encoded = tokenizer("ab c", return_tensors="pt").to("cpu")

    ids = encoded["input_ids"]
    assert ids.dtype == torch.long
    assert ids.tolist() == [[BEGIN_OF_TEXT, 2, 1]]
    assert encoded.get("attention_mask") is None


def test_the_fields_describe_tokenizer_reads() -> None:
    tokenizer = _tokenizer(_BytePairs())

    assert tokenizer.vocab_size == 5
    assert len(tokenizer) == END_OF_TEXT + 2
    assert tokenizer.bos_token_id == BEGIN_OF_TEXT
    assert tokenizer.eos_token_id == END_OF_TEXT
    assert tokenizer.name_or_path == "test"
    assert tokenizer.fingerprint == bpe_fingerprint(_tokenizer_file())


# -- loading a file --


def test_a_file_with_other_ids_is_refused(tmp_path: Path) -> None:
    """Refused before any tokenizer is built: a model reading ids it
    was never trained on would produce plausible nonsense."""
    changed = _tokenizer_file()
    changed["model"]["vocab"]["abc"] = 5
    path = tmp_path / TOKENIZER_FILE
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(TokenizerMismatchError):
        load_tokenizer(path, source="test")


def test_a_real_file_loads_with_its_added_tokens_dropped(
    tmp_path: Path,
) -> None:
    """With the installed `tokenizers`, whichever it is: a special
    token typed as text stays text, byte for byte."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {symbol: index for index, symbol in enumerate(alphabet)}
    built = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    built.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    built.decoder = decoders.ByteLevel()
    built.add_special_tokens(["<|begin_of_text|>"])
    path = tmp_path / TOKENIZER_FILE
    built.save(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))

    loaded = load_tokenizer(
        path, source="test", required=bpe_fingerprint(data)
    )

    typed = "<|begin_of_text|>hi"
    ids = loaded.encode(typed, add_special_tokens=False)
    assert all(token < len(alphabet) for token in ids)
    assert loaded.decode(ids) == typed
    assert len(loaded) == len(alphabet) + 1
