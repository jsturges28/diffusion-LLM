"""Versioned, deterministic KGW watermark primitives.

This module owns the scheme rather than a model worker. Green lists
are derived from HMAC-SHA256, then selected without replacement by
NumPy's pinned PCG64 generator. That gives an exact-size set without
touching Python, NumPy, or torch's global random generators, and gives
CPU generation and tokenizer-only detection one implementation.

The first output token may be biased from the prompt's last token,
but an output-only detector does not have that predecessor. It is
therefore marked as membership-known but detector-evidence false.
User-forced tokens follow the same evidence rule; generated tokens
after either exclusion are ordinary evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray

KGW_SCHEME = "kgw"
KGW_VERSION = 1
KGW_CACHE_ENTRIES = 32
KGW_DEVICE_CACHE_ENTRIES = 32
KGW_DEVICE_CACHES_MAX = 2
KGW_EVIDENCE_MIN = 50

KGW_SEEDING_CONTRACT = (
    "HMAC-SHA256 over scheme version, model id, tokenizer "
    "fingerprint, vocab width, and previous token"
)
KGW_RNG_CONTRACT = (
    "HMAC-seeded NumPy PCG64 choice without replacement; "
    "no global RNG state"
)
KGW_EXCLUSIONS: Tuple[str, ...] = (
    "first output token lacks its prompt predecessor",
    "user-forced tokens are not model sampling evidence",
)

_HMAC_DOMAIN = b"diffusion-llm/kgw-greenlist\x00"

assert KGW_VERSION >= 1, "a scheme version is positive"
assert KGW_CACHE_ENTRIES > 0, "the LRU holds at least one list"
assert KGW_DEVICE_CACHE_ENTRIES == KGW_CACHE_ENTRIES, (
    "host and device LRUs have the same predecessor budget"
)
assert KGW_DEVICE_CACHES_MAX > 0, "at least one device is supported"
assert KGW_EVIDENCE_MIN >= 2, "detection needs multiple samples"

GreenIds = NDArray[np.int64]


@dataclass(frozen=True)
class KgwConfig:
    """Everything that makes one run's green lists reproducible."""

    secret: bytes = field(repr=False)
    key_id: str
    model_id: str
    tokenizer_fingerprint: str
    vocab_size: int
    gamma: float
    delta: float

    def __post_init__(self) -> None:
        _validate_config_identity(self)
        green_list_size(
            gamma=self.gamma,
            vocab_size=self.vocab_size,
        )
        if not math.isfinite(self.delta):
            raise ValueError("watermark_delta must be finite")
        if self.delta < 0.0:
            raise ValueError("watermark_delta must not be negative")

    @property
    def green_list_size(self) -> int:
        """The exact green-list cardinality for this vocabulary."""
        return green_list_size(
            gamma=self.gamma,
            vocab_size=self.vocab_size,
        )

    @property
    def p0(self) -> float:
        """The exact null membership probability."""
        return null_probability(
            green_list_size=self.green_list_size,
            vocab_size=self.vocab_size,
        )


def _validate_config_identity(config: KgwConfig) -> None:
    """Validate the secret and names that define one KGW domain."""
    if not isinstance(config.secret, bytes):
        raise TypeError("KGW secret must be bytes")
    if len(config.secret) != 32:
        raise ValueError("KGW requires a 256-bit secret")
    if not config.key_id:
        raise ValueError("KGW key id must not be empty")
    if not config.model_id:
        raise ValueError("KGW model id must not be empty")
    if not config.tokenizer_fingerprint:
        raise ValueError("KGW tokenizer fingerprint is required")


@dataclass(frozen=True)
class DetectionResult:
    """A detector score with no authorship verdict attached."""

    status: str
    green_count: int
    scored_count: int
    z_score: float
    p0: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "green_count": self.green_count,
            "scored_count": self.scored_count,
            "z_score": self.z_score,
            "p0": self.p0,
        }


class KgwAccumulator:
    """Exact-null-probability online z-score bookkeeping."""

    def __init__(self, p0: float) -> None:
        if not math.isfinite(p0) or not 0.0 < p0 < 1.0:
            raise ValueError(
                "p0 must be finite and between 0 and 1"
            )
        self.p0 = p0
        self.green_count = 0
        self.scored_count = 0

    def add(self, *, green: bool, evidence: bool) -> DetectionResult:
        if not isinstance(green, bool):
            raise TypeError("green membership must be boolean")
        if not isinstance(evidence, bool):
            raise TypeError("evidence flag must be boolean")
        if evidence:
            self.scored_count += 1
            if green:
                self.green_count += 1
        return self.result()

    def restore(
        self,
        memberships: Sequence[Optional[bool]],
        evidence: Sequence[Optional[bool]],
    ) -> None:
        """Replace counts from an index-aligned retained trace."""
        if len(memberships) != len(evidence):
            raise ValueError("membership and evidence lengths differ")
        self.green_count = 0
        self.scored_count = 0
        for green, scored in zip(
            memberships, evidence, strict=True
        ):
            if green is not None and not isinstance(green, bool):
                raise TypeError("green membership must be boolean")
            if scored is not None and not isinstance(scored, bool):
                raise TypeError("evidence flag must be boolean")
            if scored is True:
                if green is None:
                    raise ValueError("scored token has no membership")
                self.scored_count += 1
                if green:
                    self.green_count += 1
        assert self.green_count <= self.scored_count

    def result(self) -> DetectionResult:
        count = self.scored_count
        score = detection_z_score(
            green_count=self.green_count,
            scored_count=count,
            p0=self.p0,
        )
        return DetectionResult(
            status=detection_status(count),
            green_count=self.green_count,
            scored_count=count,
            z_score=score,
            p0=self.p0,
        )


class KgwWatermark:
    """One run's KGW configuration, cache, and online score."""

    def __init__(
        self,
        config: KgwConfig,
        *,
        cache: Optional["GreenlistCache"] = None,
    ) -> None:
        self.config = config
        self.cache = cache or GreenlistCache(config)
        self.accumulator = KgwAccumulator(config.p0)

    def green_ids(self, previous_token: int) -> GreenIds:
        return self.cache.get(previous_token)

    def is_green(
        self, *, previous_token: int, token_id: int
    ) -> bool:
        _validate_token(token_id, self.config.vocab_size)
        green = self.green_ids(previous_token)
        offset = int(np.searchsorted(green, token_id))
        return offset < green.size and int(green[offset]) == token_id

    def observe(
        self, *, green: bool, evidence: bool
    ) -> DetectionResult:
        return self.accumulator.add(green=green, evidence=evidence)

    def restore(
        self,
        memberships: Sequence[Optional[bool]],
        evidence: Sequence[Optional[bool]],
    ) -> None:
        self.accumulator.restore(memberships, evidence)

    def fork(
        self,
        memberships: Sequence[Optional[bool]],
        evidence: Sequence[Optional[bool]],
    ) -> "KgwWatermark":
        """A branch sharing deterministic lists, not score state."""
        branch = KgwWatermark(self.config, cache=self.cache)
        branch.restore(memberships, evidence)
        return branch

    def provenance(self) -> Dict[str, object]:
        result = self.accumulator.result()
        return {
            "scheme": KGW_SCHEME,
            "version": KGW_VERSION,
            "key_id": self.config.key_id,
            "gamma": self.config.gamma,
            "delta": self.config.delta,
            "vocab_size": self.config.vocab_size,
            "green_list_size": self.config.green_list_size,
            "tokenizer_fingerprint": (
                self.config.tokenizer_fingerprint
            ),
            "seeding_contract": KGW_SEEDING_CONTRACT,
            "rng_contract": KGW_RNG_CONTRACT,
            "exclusions": list(KGW_EXCLUSIONS),
            **result.as_dict(),
        }


class GreenlistCache:
    """Fixed-size host and per-device LRUs of exact green lists."""

    def __init__(self, config: KgwConfig) -> None:
        self.config = config
        self._entries: "OrderedDict[int, GreenIds]" = OrderedDict()
        self._device_entries: (
            "OrderedDict[str, OrderedDict[int, object]]"
        ) = OrderedDict()

    def get(self, previous_token: int) -> GreenIds:
        _validate_token(previous_token, self.config.vocab_size)
        cached = self._entries.pop(previous_token, None)
        if cached is None:
            cached = _green_ids(self.config, previous_token)
        self._entries[previous_token] = cached
        if len(self._entries) > KGW_CACHE_ENTRIES:
            self._entries.popitem(last=False)
        assert len(self._entries) <= KGW_CACHE_ENTRIES
        return cached

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def device_get(
        self, *, device_key: str, previous_token: int
    ) -> Optional[object]:
        """Return and refresh one cached device value."""
        _validate_device_key(device_key)
        _validate_token(previous_token, self.config.vocab_size)
        entries = self._device_entries.pop(device_key, None)
        if entries is None:
            return None
        self._device_entries[device_key] = entries
        cached = entries.pop(previous_token, None)
        if cached is not None:
            entries[previous_token] = cached
        return cached

    def device_put(
        self,
        *,
        device_key: str,
        previous_token: int,
        value: object,
    ) -> None:
        """Cache one device value under both fixed LRU ceilings."""
        _validate_device_key(device_key)
        _validate_token(previous_token, self.config.vocab_size)
        if value is None:
            raise ValueError("a cached device value cannot be None")
        entries = self._device_entries.pop(device_key, None)
        if entries is None:
            entries = OrderedDict()
        self._device_entries[device_key] = entries
        entries.pop(previous_token, None)
        entries[previous_token] = value
        if len(entries) > KGW_DEVICE_CACHE_ENTRIES:
            entries.popitem(last=False)
        if len(self._device_entries) > KGW_DEVICE_CACHES_MAX:
            self._device_entries.popitem(last=False)
        assert len(entries) <= KGW_DEVICE_CACHE_ENTRIES
        assert len(self._device_entries) <= KGW_DEVICE_CACHES_MAX

    def device_entry_count(self, device_key: str) -> int:
        """How many tensors one device currently retains."""
        _validate_device_key(device_key)
        entries = self._device_entries.get(device_key)
        return 0 if entries is None else len(entries)

    @property
    def device_count(self) -> int:
        """How many device-specific LRUs currently exist."""
        return len(self._device_entries)


class KgwOnlineDetector:
    """Tokenizer-id detector whose prefixes match batch detection."""

    def __init__(self, watermark: KgwWatermark) -> None:
        self.watermark = watermark
        self.accumulator = KgwAccumulator(watermark.config.p0)
        self.previous_token: Optional[int] = None

    def add(
        self, token_id: int, *, evidence: bool
    ) -> DetectionResult:
        _validate_token(token_id, self.watermark.config.vocab_size)
        if not isinstance(evidence, bool):
            raise TypeError("evidence flag must be boolean")
        if self.previous_token is None:
            if evidence:
                raise ValueError(
                    "the first output token has no predecessor"
                )
            green = False
        else:
            green = self.watermark.is_green(
                previous_token=self.previous_token,
                token_id=token_id,
            )
        self.previous_token = token_id
        return self.accumulator.add(green=green, evidence=evidence)


def detect_token_ids(
    token_ids: Sequence[int],
    evidence: Sequence[bool],
    *,
    config: KgwConfig,
) -> DetectionResult:
    """Score token ids using only their output-side predecessors."""
    if len(token_ids) != len(evidence):
        raise ValueError("token ids and evidence lengths differ")
    detector = KgwOnlineDetector(KgwWatermark(config))
    result = detector.accumulator.result()
    for token_id, scored in zip(token_ids, evidence, strict=True):
        result = detector.add(token_id, evidence=scored)
    return result


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash the tokenizer rules that decide ids.

    Mamba-3 exposes its audited BPE fingerprint directly. Fast
    Hugging Face tokenizers expose their complete backend JSON, which
    includes normalization, pre-tokenization, vocabulary, merges, and
    post-processing. A source name alone is deliberately not accepted:
    repositories move, while this value is part of the HMAC domain.
    """
    declared = getattr(tokenizer, "fingerprint", None)
    if isinstance(declared, str) and declared:
        return declared
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serialize = getattr(backend, "to_str", None)
    if not callable(serialize):
        raise ValueError(
            "watermarking needs a tokenizer fingerprint"
        )
    encoded = str(serialize()).encode("utf-8")
    if not encoded:
        raise ValueError("tokenizer fingerprint source is empty")
    return hashlib.sha256(encoded).hexdigest()


def green_list_size(*, gamma: float, vocab_size: int) -> int:
    """Exact cardinality selected for one configured vocabulary."""
    if isinstance(vocab_size, bool):
        raise TypeError("vocab_size must be an integer")
    if not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size < 2:
        raise ValueError("vocab_size must contain two tokens")
    if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be finite and between 0 and 1")
    count = int(math.floor(gamma * vocab_size))
    return max(1, min(vocab_size - 1, count))


def null_probability(
    *, green_list_size: int, vocab_size: int
) -> float:
    """Exact null probability implied by the selected cardinality."""
    if not isinstance(green_list_size, int):
        raise TypeError("green_list_size must be an integer")
    if not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if not 1 <= green_list_size < vocab_size:
        raise ValueError("green list must be within the vocabulary")
    probability = green_list_size / vocab_size
    assert 0.0 < probability < 1.0
    return probability


def detection_status(scored_count: int) -> str:
    """Whether a score has enough evidence to report."""
    if not isinstance(scored_count, int) or scored_count < 0:
        raise ValueError("scored_count must be non-negative")
    if scored_count >= KGW_EVIDENCE_MIN:
        return "scored"
    return "insufficient_evidence"


def detection_z_score(
    *, green_count: int, scored_count: int, p0: float
) -> float:
    """Constant-cardinality KGW z-score under the exact null."""
    if not 0 <= green_count <= scored_count:
        raise ValueError("green_count must be within scored_count")
    if not math.isfinite(p0) or not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be finite and between 0 and 1")
    if scored_count == 0:
        return 0.0
    expected = p0 * scored_count
    variance = scored_count * p0 * (1.0 - p0)
    assert variance > 0.0, "valid p0 has positive variance"
    return (green_count - expected) / math.sqrt(variance)


def _green_ids(
    config: KgwConfig, previous_token: int
) -> GreenIds:
    """Select exact ids with a local, pinned PCG64 generator."""
    seed = _seed(config, previous_token)
    generator = np.random.Generator(np.random.PCG64(seed))
    count = config.green_list_size
    chosen = generator.choice(
        config.vocab_size,
        size=count,
        replace=False,
        shuffle=False,
    )
    result = np.sort(chosen.astype(np.int64, copy=False))
    result.flags.writeable = False
    assert result.size == count, "green list lost a candidate"
    return result


def _seed(config: KgwConfig, previous_token: int) -> int:
    """Domain-separated HMAC seed for one predecessor."""
    _validate_token(previous_token, config.vocab_size)
    digest = hmac.new(
        config.secret,
        digestmod=hashlib.sha256,
    )
    digest.update(_HMAC_DOMAIN)
    _hmac_field(digest, "scheme", KGW_SCHEME.encode("ascii"))
    _hmac_field(
        digest,
        "version",
        struct.pack(">I", KGW_VERSION),
    )
    _hmac_field(
        digest, "model", config.model_id.encode("utf-8")
    )
    _hmac_field(
        digest,
        "tokenizer",
        config.tokenizer_fingerprint.encode("utf-8"),
    )
    _hmac_field(
        digest,
        "vocab",
        struct.pack(">Q", config.vocab_size),
    )
    _hmac_field(
        digest,
        "previous",
        struct.pack(">Q", previous_token),
    )
    return int.from_bytes(digest.digest()[:8], "big")


def _hmac_field(
    digest: hmac.HMAC, name: str, value: bytes
) -> None:
    label = name.encode("ascii")
    digest.update(struct.pack(">H", len(label)))
    digest.update(label)
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _validate_token(token_id: int, vocab_size: int) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError("token id must be an integer")
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(
            f"token id {token_id} is outside [0, {vocab_size})"
        )


def _validate_device_key(device_key: str) -> None:
    if not isinstance(device_key, str) or not device_key:
        raise ValueError(
            "device cache key must be a non-empty string"
        )
