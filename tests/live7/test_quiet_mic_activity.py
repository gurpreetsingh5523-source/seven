"""Regression (2026-10-06, live E2E): while the user was still talking (75 s monologue) 7 spoke a
Hermes announcement after 23 s, because input transcriptions arrive late and the relay judged the
user "quiet". Mic energy must count as activity."""

import asyncio
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from live7.relay import LiveRelay  # noqa: E402


class S:
    def __init__(self):
        self.client_content = []

    async def send_realtime_input(self, **kw):
        pass

    async def send_client_content(self, turns=None, turn_complete=None):
        self.client_content.append(turns)


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def tone(amp):
    return b"".join(struct.pack("<h", int(amp * math.sin(i / 3))) for i in range(1600))


async def _noop(e):
    pass


def test_speech_energy_keeps_7_quiet_and_silence_does_not():
    clock = Clock()
    s = S()
    relay = LiveRelay(s, None, _noop, clock=clock, quiet_s=2.5)
    job = {"id": "j", "task": "t", "status": "done", "reply": "answer 13"}

    async def go():
        await relay.handle_audio(tone(4000))          # user talking
        await relay.announce_hermes(job)
        assert s.client_content == []                 # queued, not spoken over the user
        for _ in range(10):                           # keeps talking for 10 s
            clock.t += 1.0
            await relay.handle_audio(tone(4000))
            await relay.tick()
        assert s.client_content == []
        for _ in range(3):                            # now only room noise
            clock.t += 1.0
            await relay.handle_audio(tone(100))
            await relay.tick()
        assert len(s.client_content) == 1             # spoken in the quiet moment

    asyncio.run(go())
