"""Regression (2026-10-06, user: "the voice is doubled"): two 7 windows were open, each with
its own Gemini session answering the same mic. Only the newest browser may stay connected."""

import asyncio
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from live7.server import start_server  # noqa: E402
from live7.tools import ToolBridge  # noqa: E402
from sentinel_orchestrator import (ApprovalGate, AuditLog, LocalSandbox, PolicyEngine,  # noqa: E402
                                   SecretsBroker, ToolRegistry)


def test_newer_window_replaces_older_one():
    from websockets.asyncio.client import connect

    async def no(i, d):
        return False

    bridge = ToolBridge(ToolRegistry(), PolicyEngine(), LocalSandbox(SecretsBroker(), simulated_latency_s=0),
                        AuditLog(), ApprovalGate(no, timeout_s=1))
    live, closed = [], []

    class S:
        async def send_client_content(self, **kw): pass
        async def send_realtime_input(self, **kw): pass
        async def send_tool_response(self, **kw): pass
        async def receive(self):
            while True:
                await asyncio.sleep(3600)
                yield None

    def factory(**kw):
        @contextlib.asynccontextmanager
        async def cm():
            s = S()
            live.append(s)
            try:
                yield s
            finally:
                closed.append(s)
        return cm()

    async def go():
        server = await start_server(factory, bridge, port=0)
        base = f"http://127.0.0.1:{server.port}"
        try:
            async with connect(f"ws://127.0.0.1:{server.port}/live", origin=base) as first:
                await asyncio.sleep(0.3)
                async with connect(f"ws://127.0.0.1:{server.port}/live", origin=base) as second:
                    got = []
                    with contextlib.suppress(Exception):
                        while True:
                            m = await asyncio.wait_for(first.recv(), 2)
                            if isinstance(m, str):
                                got.append(json.loads(m))
                    await asyncio.sleep(0.3)
                    second_open = second.state.name == "OPEN"
                    snapshot = (list(live), list(closed))     # before the test tears everything down
                    return got, first.close_code, second_open, snapshot
        finally:
            await server.close()

    got, code, second_open, (live, closed) = asyncio.run(go())
    assert any(e.get("type") == "replaced" for e in got)
    assert code == 4000
    assert second_open
    assert len(live) == 2 and live[0] in closed and live[1] not in closed   # only ONE Gemini session left
