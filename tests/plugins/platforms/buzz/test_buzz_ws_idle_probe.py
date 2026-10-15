"""Regression tests for the Buzz WebSocket idle bound vs a quiet relay.

The read loop's idle bound counts data frames only — protocol-level ping/pong never reaches the async
iterator — so a relay with nothing to say tripped the bound every 300s and the adapter reconnected,
re-authenticated (NIP-42) and re-subscribed, forever (~288 times/day per instance on the relays this was
measured on). The fix probes the relay with one ``limit: 0`` REQ first, and reconnects only when that probe
goes unanswered, so the CLOSE_WAIT safety net stays intact.

Everything here drives the adapter with a fake websocket whose inbound side is a queue: the relay "answers"
exactly when the adapter sends the probe, so the tests wait on conditions instead of the clock. No relay, no
network, no sleeping.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from plugins.platforms.buzz import adapter


class FakeWebSocket:
    """Records outbound frames; a probe REQ is answered with an EOSE unless ``answer_probe`` is False."""

    def __init__(self, *, answer_probe: bool = True) -> None:
        self.sent: list[list] = []
        self._answer_probe = answer_probe
        self._inbound: asyncio.Queue = asyncio.Queue()

    async def send(self, payload: str) -> None:
        frame = json.loads(payload)
        self.sent.append(frame)
        if frame[0] == "REQ" and self._answer_probe:
            # The relay's answer to a `limit: 0` REQ is an immediate EOSE.
            self._inbound.put_nowait(["EOSE", frame[1]])

    def __aiter__(self):
        return self._generate()

    async def _generate(self):
        while True:
            yield json.dumps(await self._inbound.get())

    def requests(self) -> list[list]:
        return [frame for frame in self.sent if frame[0] == "REQ"]

    def closes(self) -> list[list]:
        return [frame for frame in self.sent if frame[0] == "CLOSE"]


class DeadWebSocket:
    """A socket that cannot even accept the probe."""

    def __init__(self) -> None:
        self.sent: list[list] = []

    async def send(self, payload: str) -> None:
        raise ConnectionError("socket gone")


def make_adapter():
    """An adapter with only the state the idle-probe paths touch (no config needed)."""
    obj = object.__new__(adapter.BuzzAdapter)
    obj._self_pubkey = "ab" * 32
    obj.idle_probe_keeps = 0
    obj._channel_state = {}
    obj._restricted_channels = set()
    obj._membership_since = 0
    return obj


def parked_read(ws) -> asyncio.Task:
    """The read task the idle bound waits on: parked on the inbound queue."""
    return asyncio.ensure_future(ws.__aiter__().__anext__())


def forever() -> asyncio.Task:
    """A read task that never completes (a socket that sends nothing at all)."""
    return asyncio.ensure_future(asyncio.Event().wait())


class TestIdleProbe:
    def test_answered_probe_keeps_the_connection(self, monkeypatch):
        monkeypatch.setattr(adapter, "_WS_IDLE_PROBE_GRACE", 1.0)

        async def scenario():
            obj = make_adapter()
            ws = FakeWebSocket()
            task = parked_read(ws)
            kept = await obj._probe_idle_relay(ws, task)
            return obj, ws, kept

        obj, ws, kept = asyncio.run(scenario())

        assert kept is True
        assert obj.idle_probe_keeps == 1
        assert ws.requests()[0][1] == adapter._WS_IDLE_PROBE_SUB_ID
        assert ws.requests()[0][2]["limit"] == 0
        assert ws.closes() == [["CLOSE", adapter._WS_IDLE_PROBE_SUB_ID]]

    def test_unanswered_probe_still_reports_silence(self, monkeypatch):
        monkeypatch.setattr(adapter, "_WS_IDLE_PROBE_GRACE", 0.2)

        async def scenario():
            obj = make_adapter()
            ws = FakeWebSocket(answer_probe=False)
            task = parked_read(ws)
            return obj, ws, await obj._probe_idle_relay(ws, task)

        obj, ws, kept = asyncio.run(scenario())

        assert kept is False, "an unanswered probe must fall back to the reconnect path"
        assert obj.idle_probe_keeps == 0
        # The probe subscription is returned even when nothing came back, or the relay accumulates one per probe.
        assert ws.closes() == [["CLOSE", adapter._WS_IDLE_PROBE_SUB_ID]]

    def test_probe_send_failure_reports_silence(self):
        async def scenario():
            obj = make_adapter()
            return await obj._probe_idle_relay(DeadWebSocket(), forever())

        assert asyncio.run(scenario()) is False


class TestReadLoopIdleBound:
    def test_quiet_relay_does_not_reconnect(self, monkeypatch):
        monkeypatch.setattr(adapter, "_WS_IDLE_PROBE_GRACE", 1.0)

        async def scenario():
            obj = make_adapter()
            obj.ws_idle_timeout = 0.05  # the 300s bound, compressed
            ws = FakeWebSocket()
            try:
                await asyncio.wait_for(obj._ws_read_loop(ws, {}), timeout=0.8)
            except TimeoutError:
                return "still connected", ws
            except ConnectionError as exc:  # pragma: no cover - the regression this file exists for
                return f"reconnected: {exc}", ws
            return "exited", ws

        outcome, ws = asyncio.run(scenario())

        assert outcome == "still connected"
        assert len(ws.requests()) >= 2, "a long quiet window must keep probing"
        assert len(ws.closes()) == len(ws.requests())

    def test_silent_socket_still_reconnects(self, monkeypatch):
        monkeypatch.setattr(adapter, "_WS_IDLE_PROBE_GRACE", 0.1)

        async def scenario():
            obj = make_adapter()
            obj.ws_idle_timeout = 0.05
            ws = FakeWebSocket(answer_probe=False)  # silent socket + no probe answer = the dead-socket case
            try:
                await asyncio.wait_for(obj._ws_read_loop(ws, {}), timeout=1.0)
            except ConnectionError:
                return "reconnected"
            except TimeoutError:  # pragma: no cover
                return "hung"
            return "exited"

        assert asyncio.run(scenario()) == "reconnected"


class TestProbeClosedAck:
    def test_probe_ack_is_not_a_channel_rejection(self):
        async def scenario():
            obj = make_adapter()
            await obj._handle_ws_message(
                DeadWebSocket(), {}, ["CLOSED", adapter._WS_IDLE_PROBE_SUB_ID, "closed by client"]
            )

        asyncio.run(scenario())  # must not raise: the ack is our own CLOSE, not a relay rejection

    def test_real_channel_closed_still_disconnects(self):
        async def scenario():
            obj = make_adapter()
            await obj._handle_ws_message(DeadWebSocket(), {"hermes-buzz-0": "chan"}, ["CLOSED", "hermes-buzz-0", "oops"])

        with pytest.raises(ConnectionError):
            asyncio.run(scenario())


class TestIdleTimeoutConfiguration:
    def test_default_is_the_module_constant(self, monkeypatch):
        monkeypatch.delenv("BUZZ_WS_IDLE_TIMEOUT", raising=False)
        assert adapter.BuzzAdapter(SimpleNamespace(extra={})).ws_idle_timeout == adapter._WS_READ_IDLE_TIMEOUT

    def test_extra_value_is_used(self, monkeypatch):
        monkeypatch.delenv("BUZZ_WS_IDLE_TIMEOUT", raising=False)
        assert adapter.BuzzAdapter(SimpleNamespace(extra={"ws_idle_timeout": 45})).ws_idle_timeout == 45.0

    def test_env_wins_over_extra(self, monkeypatch):
        monkeypatch.setenv("BUZZ_WS_IDLE_TIMEOUT", "77")
        assert adapter.BuzzAdapter(SimpleNamespace(extra={"ws_idle_timeout": 45})).ws_idle_timeout == 77.0

    def test_below_floor_is_clamped(self, monkeypatch):
        monkeypatch.delenv("BUZZ_WS_IDLE_TIMEOUT", raising=False)
        assert (
            adapter.BuzzAdapter(SimpleNamespace(extra={"ws_idle_timeout": 5})).ws_idle_timeout
            == adapter._MIN_WS_IDLE_TIMEOUT
        )

    def test_invalid_value_falls_back(self, monkeypatch):
        monkeypatch.delenv("BUZZ_WS_IDLE_TIMEOUT", raising=False)
        assert (
            adapter.BuzzAdapter(SimpleNamespace(extra={"ws_idle_timeout": "bad"})).ws_idle_timeout
            == adapter._WS_READ_IDLE_TIMEOUT
        )

    def test_read_loop_uses_the_configured_value(self, monkeypatch):
        """The bound must come from the instance: a config that only shows up in logs is decoy."""
        monkeypatch.setattr(adapter, "_WS_IDLE_PROBE_GRACE", 0.1)

        async def scenario():
            obj = make_adapter()
            obj.ws_idle_timeout = 0.05
            ws = FakeWebSocket()
            try:
                await asyncio.wait_for(obj._ws_read_loop(ws, {}), timeout=0.5)
            except (TimeoutError, ConnectionError):
                pass
            return ws

        assert asyncio.run(scenario()).requests(), "a 0.05s bound must have probed"
