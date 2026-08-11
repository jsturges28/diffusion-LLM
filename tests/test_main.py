"""Tests that the browser launcher stays machine-local by default.

Strategy: parse arguments with a stubbed ``sys.argv`` and exercise
the loopback test and its warning directly, so nothing binds a real
socket. What a passing test proves is that the default command in the
README serves only this machine, that every spelling of loopback is
recognized as such, and that an explicit remote bind says out loud
what it is exposing.

The bind address is the whole safety property here: the same
unauthenticated origin that draws the UI can also activate models and
permanently delete saved runs, so "local tool" has to mean local by
default rather than by convention.
"""

from __future__ import annotations

import socket
from typing import List

import pytest

import main


def _parse(
    monkeypatch: pytest.MonkeyPatch, argv: List[str]
) -> object:
    monkeypatch.setattr("sys.argv", ["main.py", *argv])
    return main.parse_args()


def test_the_default_bind_is_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression itself: this used to default to 0.0.0.0."""
    args = _parse(monkeypatch, [])

    assert args.host == "127.0.0.1"
    assert main.is_loopback(args.host)


def test_the_default_port_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paired with the test above so a future edit to the parser
    cannot quietly move the port while fixing the host."""
    args = _parse(monkeypatch, [])

    assert args.port == 8000


def test_an_explicit_host_still_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(monkeypatch, ["--host", "0.0.0.0"])

    assert args.host == "0.0.0.0"


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "127.0.0.2",
        "127.255.255.254",
        "localhost",
        "LOCALHOST",
        "::1",
        "  127.0.0.1  ",
    ],
)
def test_loopback_spellings_are_recognized(host: str) -> None:
    """The whole 127.0.0.0/8 block and ::1, not one string.

    A check for the single spelling everybody types would warn on
    127.0.0.2, which is just as local, and teach the user to ignore
    the warning that matters.
    """
    assert main.is_loopback(host)


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "192.168.1.5",
        "10.0.0.7",
        "::",
        "",
        "   ",
        "example.local",
    ],
)
def test_non_loopback_hosts_are_not_mistaken_for_local(
    host: str,
) -> None:
    """Negative space, including the two that are not addresses.

    An empty or unresolvable host is reported as remote on purpose:
    a name this cannot vouch for should get the warning rather than
    silence.
    """
    assert not main.is_loopback(host)


def test_an_exposed_bind_warns_about_what_it_exposes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main._warn_if_exposed("0.0.0.0")

    warning = capsys.readouterr().err
    assert "0.0.0.0" in warning
    assert "no authentication" in warning
    assert "delete" in warning


def test_a_loopback_bind_says_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The pair to the test above: the default must be quiet, or
    the warning becomes noise the user learns to skip."""
    main._warn_if_exposed("127.0.0.1")

    assert capsys.readouterr().err == ""


# -- the property the argument default exists to produce --
#
# The tests above prove the launcher hands uvicorn a loopback
# address. These two prove what that buys, by binding real sockets
# and reaching for them across this machine's own network address.
# They come as a pair on purpose: an "unreachable" result alone
# would also be produced by a firewall, so the 0.0.0.0 case has to
# demonstrate that the same reach succeeds when the bind allows it.
# Both skip on a host with no non-loopback address, where the
# question does not arise.


def _network_address() -> str:
    """This machine's own non-loopback IPv4 address, or "".

    Found by connecting a UDP socket to an address reserved for
    documentation, which sends nothing but makes the kernel pick
    the source address it would route from.

    Empty means the question cannot be asked here: either the host
    has no non-loopback address, or the environment forbids sockets
    outright, which the agent sandbox does. Both skip below rather
    than pretend to have checked.
    """
    try:
        probe = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM
        )
    except OSError:
        return ""
    try:
        probe.connect(("192.0.2.1", 9))
        address = str(probe.getsockname()[0])
    except OSError:
        return ""
    finally:
        probe.close()
    if main.is_loopback(address):
        return ""
    return address


def _reachable_at(bind_host: str, target: str) -> bool:
    """Whether a listener on `bind_host` answers on `target`."""
    listener = socket.socket()
    try:
        listener.bind((bind_host, 0))
        listener.listen(1)
        port = int(listener.getsockname()[1])
        client = socket.socket()
        client.settimeout(2.0)
        try:
            client.connect((target, port))
            return True
        except OSError:
            return False
        finally:
            client.close()
    finally:
        listener.close()


_NO_NETWORK = (
    "no usable non-loopback IPv4 address on this host"
)


def test_the_default_listener_is_unreachable_from_the_network(
) -> None:
    address = _network_address()
    if address == "":
        pytest.skip(_NO_NETWORK)

    assert not _reachable_at(main.DEFAULT_HOST, address)


def test_an_exposed_listener_is_reachable_from_the_network(
) -> None:
    """Not a feature request, a control. Without it the test above
    would pass on a host that simply refuses every connection."""
    address = _network_address()
    if address == "":
        pytest.skip(_NO_NETWORK)

    assert _reachable_at("0.0.0.0", address)
