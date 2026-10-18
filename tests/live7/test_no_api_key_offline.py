"""Without GOOGLE_API_KEY, 7 Live must still start and serve offline mode instead of
exiting at launch (local-only users have no Gemini key)."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import live7.__main__ as main_mod  # noqa: E402
from live7.offline import offline_session_factory  # noqa: E402
from live7.server import start_server  # noqa: E402
from test_m7_offline import fakes  # noqa: E402
from test_online_recovery import bridge, modes  # noqa: E402


def test_load_api_key_returns_empty_instead_of_exiting(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(main_mod, "HERMES_ENV", tmp_path / "missing.env")
    assert main_mod.load_api_key() == ""


def test_keyless_online_factory_falls_back_to_offline(monkeypatch):
    monkeypatch.setenv("SEVEN_BACKOFF_SCALE", "0.01")
    monkeypatch.delenv("SEVEN_MODE", raising=False)

    def keyless(voice=None, resume_handle=None):
        raise RuntimeError("GOOGLE_API_KEY is not set")

    async def go():
        server = await start_server(keyless, bridge(), port=0,
                                    offline_factory=offline_session_factory(*fakes()[:3]))
        try:
            return await modes(server, 1.5)
        finally:
            server.close()

    got = asyncio.run(go())
    assert "offline" in got
    assert "online" not in got
