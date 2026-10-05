"""Durable secret-key management for the KGW watermark.

The key belongs to the host, not to a run. The first enabled KGW run
creates 32 random bytes below the XDG state directory and every later
run reuses them. Merely importing this module does no I/O, which is
important: generation with watermarking off must never read or create
the key.

Experiments may provide ``DIFFUSION_LLM_KGW_KEY_HEX``. That override
is parsed in memory and does not consult the durable file. Only a
short one-way identifier leaves this module; the secret itself must
never enter provenance, logs, or a saved run.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

KGW_KEY_BYTES = 32
KGW_KEY_ENV = "DIFFUSION_LLM_KGW_KEY_HEX"
KGW_KEY_FILE = "kgw-v1.key"
KGW_STATE_DIRECTORY = "diffusion-llm"
KGW_KEY_MODE = 0o600
KGW_DIRECTORY_MODE = 0o700

assert KGW_KEY_BYTES * 8 == 256, "KGW uses a 256-bit key"
assert KGW_KEY_MODE & 0o077 == 0, "the key must not be shared"


@dataclass(frozen=True)
class WatermarkKey:
    """A secret and the safe identifier derived from it."""

    secret: bytes = field(repr=False)
    key_id: str

    def __post_init__(self) -> None:
        _validate_secret(self.secret)
        assert self.key_id, "a loaded key has an identifier"
        assert len(self.key_id) == 16, "the identifier is 64 bits"


def load_or_create_key(
    *,
    environ: Optional[Mapping[str, str]] = None,
    state_home: Optional[Path] = None,
) -> WatermarkKey:
    """Load the experiment override or the host's durable key.

    This is intentionally an active verb. Callers invoke it only
    after resolving ``watermark=True``; there is no module-level
    lookup and no harmless-looking metadata helper that reads the
    file on the off path.
    """
    environment = os.environ if environ is None else environ
    override = environment.get(KGW_KEY_ENV)
    if override is not None:
        secret = _parse_hex_key(override)
        return WatermarkKey(secret, key_id(secret))

    directory = _key_directory(
        environment=environment,
        state_home=state_home,
    )
    _ensure_secure_directory(directory)
    path = directory / KGW_KEY_FILE
    secret = _read_or_create(path)
    return WatermarkKey(secret, key_id(secret))


def load_key(
    *,
    environ: Optional[Mapping[str, str]] = None,
    state_home: Optional[Path] = None,
) -> WatermarkKey:
    """Load an existing override or durable key without creating."""
    environment = os.environ if environ is None else environ
    override = environment.get(KGW_KEY_ENV)
    if override is not None:
        secret = _parse_hex_key(override)
        return WatermarkKey(secret, key_id(secret))
    directory = _key_directory(
        environment=environment,
        state_home=state_home,
    )
    _validate_directory(directory)
    path = directory / KGW_KEY_FILE
    try:
        secret = _read_key(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"KGW key does not exist: {path}"
        ) from None
    return WatermarkKey(secret, key_id(secret))


def key_id(secret: bytes) -> str:
    """A stable identifier safe to record in provenance."""
    _validate_secret(secret)
    digest = hashlib.sha256(
        b"diffusion-llm/kgw-key-id/v1\x00" + secret
    ).hexdigest()
    identifier = digest[:16]
    assert len(identifier) == 16, "the key id has a fixed width"
    return identifier


def _state_home(environment: Mapping[str, str]) -> Path:
    """Resolve XDG state without touching the filesystem."""
    configured = environment.get("XDG_STATE_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "state"


def _key_directory(
    *,
    environment: Mapping[str, str],
    state_home: Optional[Path],
) -> Path:
    root = (
        _state_home(environment)
        if state_home is None
        else Path(state_home)
    )
    return root / KGW_STATE_DIRECTORY


def _ensure_secure_directory(directory: Path) -> None:
    """Create the private state directory or validate the winner."""
    created = False
    try:
        directory.mkdir(
            mode=KGW_DIRECTORY_MODE,
            parents=True,
            exist_ok=False,
        )
        created = True
    except FileExistsError:
        pass
    if created:
        directory.chmod(KGW_DIRECTORY_MODE)
        _fsync_directory(directory.parent)
    _validate_directory(directory)


def _parse_hex_key(value: str) -> bytes:
    """Parse the explicit 256-bit experiment key."""
    if len(value) != KGW_KEY_BYTES * 2:
        raise ValueError(
            f"{KGW_KEY_ENV} must be exactly 64 hexadecimal characters"
        )
    try:
        secret = bytes.fromhex(value)
    except ValueError:
        raise ValueError(
            f"{KGW_KEY_ENV} must contain only hexadecimal characters"
        ) from None
    _validate_secret(secret)
    return secret


def _read_or_create(path: Path) -> bytes:
    """Publish one complete key atomically, or read the winner."""
    try:
        return _read_key(path)
    except FileNotFoundError:
        pass
    secret = secrets.token_bytes(KGW_KEY_BYTES)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        KGW_KEY_MODE,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            os.fchmod(handle.fileno(), KGW_KEY_MODE)
            handle.write(secret)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _publish_temporary(temporary=temporary, path=path)
    loaded = _read_key(path)
    if loaded != secret:
        return loaded
    return secret


def _publish_temporary(*, temporary: Path, path: Path) -> None:
    """Link one winner, remove the temporary, and persist metadata."""
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _read_key(path: Path) -> bytes:
    """Read an existing regular file without following a symlink."""
    flags = os.O_RDONLY
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags | no_follow)
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        details = os.fstat(handle.fileno())
        _validate_file_details(details, path)
        secret = handle.read(KGW_KEY_BYTES + 1)
    _validate_secret(secret)
    return secret


def _validate_secret(secret: bytes) -> None:
    if not isinstance(secret, bytes):
        raise TypeError("KGW key must be bytes")
    if len(secret) != KGW_KEY_BYTES:
        raise ValueError("KGW key must contain exactly 32 bytes")


def _validate_directory(path: Path) -> None:
    """Refuse a shared, foreign-owned, or redirected key directory."""
    try:
        details = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"KGW state directory does not exist: {path}"
        ) from None
    if not stat.S_ISDIR(details.st_mode):
        raise ValueError(
            f"KGW state path is not a directory: {path}"
        )
    mode = stat.S_IMODE(details.st_mode)
    if mode != KGW_DIRECTORY_MODE:
        raise PermissionError(
            "KGW state directory must have mode 0700, "
            f"found {mode:04o}: {path}"
        )
    _validate_owner(details, path)


def _validate_file_details(
    details: os.stat_result, path: Path
) -> None:
    """Refuse key files that are shared, foreign, or non-regular."""
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"KGW key is not a regular file: {path}")
    mode = stat.S_IMODE(details.st_mode)
    if mode != KGW_KEY_MODE:
        raise PermissionError(
            f"KGW key must have mode 0600, found {mode:04o}: {path}"
        )
    _validate_owner(details, path)


def _validate_owner(details: os.stat_result, path: Path) -> None:
    """Require the current account to own durable secret state."""
    effective_user = getattr(os, "geteuid", None)
    if effective_user is None:
        return
    if details.st_uid != effective_user():
        raise PermissionError(
            f"KGW state must be owned by the current account: {path}"
        )


def _fsync_directory(path: Path) -> None:
    """Persist directory entries after creation, link, and unlink."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
