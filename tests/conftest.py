"""
Shared test fixtures.

The ``no_network`` fixture lives here rather than in a single test module so
every suite can assert AR-01 (default path makes zero network calls) rather
than only the hygiene suite. Any test touching a default code path should
request it.
"""
import socket

import pytest


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test if anything attempts a network connection (AR-01).

    Use on every default-path test. Quickstart and local flows must pass under
    this fixture with zero exemptions.
    """

    def _blocked(*args, **kwargs):  # pragma: no cover - triggered only on violation
        raise AssertionError("AR-01 violated: default path attempted a network call")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    yield
