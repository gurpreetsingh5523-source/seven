"""Regression (2026-10-06): the page loaded but its script died with
"Identifier 'STATES' has already been declared" (live_core.js leaked top-level
names into the page's script scope). Static HTML checks could not see it.
This test runs the real page in headless Chrome: no console errors, and the
page's own JavaScript must open the /live WebSocket."""

import asyncio
import contextlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CHROME = next((p for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                           shutil.which("google-chrome") or "", shutil.which("chromium") or "")
               if p and Path(p).exists()), None)
pytestmark = pytest.mark.skipif(CHROME is None, reason="Chrome not installed")


def test_page_script_runs_and_connects():
    from live7.server import start_server
    from live7.tools import ToolBridge
    from sentinel_orchestrator import (ApprovalGate, AuditLog, LocalSandbox, PolicyEngine,
                                       SecretsBroker, ToolRegistry)

    async def no(i, d):
        return False

    bridge = ToolBridge(ToolRegistry(), PolicyEngine(), LocalSandbox(SecretsBroker(), simulated_latency_s=0),
                        AuditLog(), ApprovalGate(no, timeout_s=1))
    opened = []

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
            opened.append(kw)
            yield S()
        return cm()

    async def go():
        server = await start_server(factory, bridge, port=0)
        try:
            with tempfile.TemporaryDirectory() as prof:
                proc = await asyncio.create_subprocess_exec(
                    CHROME, "--headless=new", "--disable-gpu", "--enable-logging=stderr", "--v=0",
                    f"--user-data-dir={prof}", "--virtual-time-budget=5000", "--dump-dom",
                    f"http://127.0.0.1:{server.port}/",
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
                # An open WebSocket keeps headless Chrome alive: wait for the page to
                # connect (or 20 s), then stop Chrome ourselves and read its console log.
                for _ in range(80):
                    if opened:
                        break
                    await asyncio.sleep(0.25)
                await asyncio.sleep(1.0)
                with contextlib.suppress(ProcessLookupError):   # Chrome may already have exited
                    proc.kill()
                _, err = await proc.communicate()
                return err.decode(errors="ignore")
        finally:
            await server.close()

    log = asyncio.run(go())
    console = [l for l in log.splitlines() if "CONSOLE" in l]
    errors = [l for l in console if "Uncaught" in l or "Error" in l]
    assert errors == [], errors
    assert opened, "the page's JavaScript never opened the /live WebSocket"
