"""Regression (2026-10-06 review): the M12 rule allows 10 MB drops, but the websocket used the 1 MiB
default, so any file over ~750 KB closed the connection (1009). Also: unsupported files are not kept."""

import asyncio
import base64
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from live7.files import ingest_file  # noqa: E402
from live7.server import start_server  # noqa: E402
from live7.tools import ToolBridge  # noqa: E402
from sentinel_orchestrator import (ApprovalGate, AuditLog, LocalSandbox, PolicyEngine,  # noqa: E402
                                   SecretsBroker, ToolRegistry)


def test_9mb_text_file_over_real_websocket(tmp_path, monkeypatch):
    from websockets.asyncio.client import connect

    monkeypatch.setenv("SEVEN_UPLOADS", str(tmp_path))

    async def no(i, d):
        return False

    bridge = ToolBridge(ToolRegistry(), PolicyEngine(), LocalSandbox(SecretsBroker(), simulated_latency_s=0),
                        AuditLog(), ApprovalGate(no, timeout_s=1))
    turns = []

    class S:
        async def send_client_content(self, turns=None, **kw): turns_.append(turns)
        async def send_realtime_input(self, **kw): pass
        async def send_tool_response(self, **kw): pass
        async def receive(self):
            while True:
                await asyncio.sleep(3600)
                yield None

    turns_ = turns

    def factory(**kw):
        @contextlib.asynccontextmanager
        async def cm():
            yield S()
        return cm()

    payload = ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ hello\n" * 400_000).encode()[: 9 * 1024 * 1024]
    payload = payload.decode("utf-8", "ignore").encode()

    async def go():
        server = await start_server(factory, bridge, port=0)
        base = f"http://127.0.0.1:{server.port}"
        try:
            async with connect(f"ws://127.0.0.1:{server.port}/live", origin=base, max_size=None) as ws:
                await asyncio.sleep(0.3)
                await ws.send(json.dumps({"type": "file", "name": "big.txt", "mime": "text/plain",
                                          "data_b64": base64.b64encode(payload).decode()}))
                got = []
                with contextlib.suppress(asyncio.TimeoutError):
                    while not any(e.get("type") == "file_received" for e in got):
                        m = await asyncio.wait_for(ws.recv(), 10)
                        if isinstance(m, str):
                            got.append(json.loads(m))
                return got, ws.close_code
        finally:
            await server.close()

    got, code = asyncio.run(go())
    assert code is None, f"connection closed with {code}"
    assert any(e.get("type") == "file_received" and e.get("kind") == "text" for e in got)
    assert turns and "[User shared file big.txt]" in turns[-1][0].parts[0].text


def test_unsupported_file_is_not_kept(tmp_path):
    r = ingest_file("prog.exe", "application/octet-stream", b"MZ", tmp_path)
    assert r["kind"] == "unsupported" and list(tmp_path.iterdir()) == []
