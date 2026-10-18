"""Live bug 2026-10-07: one Gemini Live drop (1008 at 13:20) sent the reconnect to the offline
fallback and 7 stayed offline for 9 hours; the offline brain was off, so every
turn — even pure noise — answered "My local brain isn't running". These checks pin the fix:
a dropped LIVE session reconnects online; a failed connect falls back offline but keeps retrying
online; noise-only turns get no reply; the offline brain defaults to Qwen3-4B :8097."""

import asyncio
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from live7.offline import OfflineSession, offline_session_factory  # noqa: E402
from live7.server import start_server  # noqa: E402
from live7.tools import ToolBridge  # noqa: E402
from sentinel_orchestrator import ApprovalGate, AuditLog, LocalSandbox, PolicyEngine, SecretsBroker, ToolRegistry  # noqa: E402
from test_m7_offline import fakes, feed, silence, tone  # noqa: E402


def bridge():
    async def no(i, d):
        return False
    return ToolBridge(ToolRegistry(), PolicyEngine(), LocalSandbox(SecretsBroker(), simulated_latency_s=0),
                      AuditLog(), ApprovalGate(no, timeout_s=1))


class Live:
    """Fake online session; dies on receive() when die=True."""
    def __init__(self, die):
        self.die = die
    async def send_client_content(self, **kw): pass
    async def send_realtime_input(self, **kw): pass
    async def send_tool_response(self, **kw): pass
    async def receive(self):
        if self.die:
            await asyncio.sleep(0.05)
            raise ConnectionError("1008 The operation was aborted.")
        while True:
            await asyncio.sleep(3600)
            yield None


async def modes(server, seconds):
    from websockets.asyncio.client import connect
    base = f"http://127.0.0.1:{server.port}"
    got = []
    async with connect(f"ws://127.0.0.1:{server.port}/live", origin=base, max_size=None) as ws:
        end = asyncio.get_running_loop().time() + seconds
        while asyncio.get_running_loop().time() < end:
            try:
                m = await asyncio.wait_for(ws.recv(), 0.2)
            except asyncio.TimeoutError:
                continue
            if isinstance(m, str):
                e = json.loads(m)
                if e.get("type") == "mode":
                    got.append(e["mode"])
    return got


def test_dropped_live_session_reconnects_online_not_offline(monkeypatch):
    monkeypatch.setenv("SEVEN_BACKOFF_SCALE", "0.01")
    monkeypatch.delenv("SEVEN_MODE", raising=False)
    calls = []

    def online(**kw):
        calls.append(1)
        @contextlib.asynccontextmanager
        async def cm():
            yield Live(die=len(calls) == 1)
        return cm()

    async def go():
        server = await start_server(online, bridge(), port=0, offline_factory=offline_session_factory(*fakes()[:3]))
        try:
            return await modes(server, 1.5)
        finally:
            await server.close()
    got = asyncio.run(go())
    assert got[:2] == ["online", "online"] and "offline" not in got, got


def test_offline_fallback_retries_online(monkeypatch):
    monkeypatch.setenv("SEVEN_BACKOFF_SCALE", "0.01")
    monkeypatch.setenv("SEVEN_ONLINE_RETRY_S", "0.4")
    monkeypatch.delenv("SEVEN_MODE", raising=False)
    calls = []

    def online(**kw):
        calls.append(1)
        @contextlib.asynccontextmanager
        async def cm():
            if len(calls) == 1:
                raise ConnectionError("no internet")
            yield Live(die=False)
        return cm()

    async def go():
        server = await start_server(online, bridge(), port=0, offline_factory=offline_session_factory(*fakes()[:3]))
        try:
            return await modes(server, 2.0)
        finally:
            await server.close()
    got = asyncio.run(go())
    assert got[:2] == ["offline", "online"], got


def test_forced_offline_never_retries_online(monkeypatch):
    monkeypatch.setenv("SEVEN_MODE", "offline")
    monkeypatch.setenv("SEVEN_ONLINE_RETRY_S", "0.2")
    calls = []

    def online(**kw):
        calls.append(1)
        raise AssertionError("must not dial out")

    async def go():
        server = await start_server(online, bridge(), port=0, offline_factory=offline_session_factory(*fakes()[:3]))
        try:
            return await modes(server, 1.0)
        finally:
            await server.close()
    assert asyncio.run(go()) == ["offline"] and calls == []


def test_noise_only_turn_gets_no_reply():
    stt, brain, tts, seen = fakes()
    asked = []

    async def brain2(text, history):
        asked.append(text)
        return "should not be said"

    async def go():
        s = OfflineSession(lambda pcm, lang: "", brain2, tts, silence_ms=300)
        assert s.idle
        await feed(s, tone(400) + silence(500))
        msgs = [m async for m in s.receive()]
        return s, msgs
    s, msgs = asyncio.run(go())
    assert asked == [] and s.history == [] and s.idle
    assert [m.server_content.turn_complete for m in msgs] == [True]


def test_offline_brain_defaults_to_qwen():
    import live7.__main__ as m
    assert m.OFFLINE_BRAINS[0] == ("http://127.0.0.1:8097", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
