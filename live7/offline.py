"""Offline mode (SPEC E8): no internet — local STT -> local brain -> local TTS.

An OfflineSession looks EXACTLY like a google.genai Live session to
LiveRelay (send_realtime_input / send_client_content / send_tool_response /
receive()), so the relay and server need no offline-specific code:

  browser PCM16 16 kHz -> energy VAD -> stt(pcm, lang) [in a thread]
    -> brain(text, history) -> tts(reply) -> PCM16 24 kHz -> browser

and every turn yields LiveServerMessage objects in the Live SDK's order:
input_transcription, output_transcription, audio model_turn (if the TTS
returned bytes), turn_complete. Like the real SDK, receive()'s iterator
ends after each turn_complete, so LiveRelay.pump() re-enters per turn.

Honesty (SPEC §6): Gurmukhi replies are text-only (tts returns b"") — no
local Gurmukhi TTS exists on this Mac, so we never fake speech.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import struct
from typing import Any, Awaitable, Callable, Dict, List, Optional

from google.genai import types

SAMPLE_RATE = 16000            # mic frames up
TTS_RATE = 24000               # audio down (matches Gemini Live's PCM rate)
FRAME_BYTES = 3200             # 100 ms of PCM16 mono 16 kHz
MAX_HISTORY = 20               # last N user/model entries kept

STTCallable = Callable[[bytes, str], str]                       # sync
BrainCallable = Callable[[str, List[Dict[str, str]]], Awaitable[str]]
TTSCallable = Callable[[str], Awaitable[bytes]]


def _rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of a PCM16 LE buffer (0 if empty)."""
    n = len(pcm) // 2
    if n <= 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    return math.sqrt(sum(s * s for s in samples) / n)


class OfflineSession:
    """A Gemini-Live-shaped session backed entirely by local components."""

    def __init__(
        self,
        stt: STTCallable,
        brain: BrainCallable,
        tts: TTSCallable,
        silence_ms: int = 700,
        threshold: float = 500,
    ) -> None:
        self._stt = stt
        self._brain = brain
        self._tts = tts
        self.silence_ms = silence_ms
        self.threshold = threshold
        self.lang = "en"
        self.history: List[Dict[str, str]] = []
        self._buf = bytearray()      # bytes of the current utterance
        self._spoken = False         # any frame at/above threshold yet
        self._quiet_ms = 0           # consecutive sub-threshold audio
        self._pending: "asyncio.Queue[tuple]" = asyncio.Queue()
        self._busy = False           # a turn (STT -> brain -> TTS) is running

    @property
    def idle(self) -> bool:
        """True when nothing is heard, queued or being answered — the server
        only switches back to Gemini Live at such a moment."""
        return not self._busy and not self._spoken and self._pending.empty()

    # -- configuration -----------------------------------------------------

    def set_lang(self, lang: str) -> None:
        """Switch the STT language: \"pa\" (Gurmukhi) or \"en\"."""
        self.lang = "pa" if str(lang).lower().startswith("pa") else "en"

    # -- the Live-session interface ---------------------------------------

    async def send_realtime_input(
        self,
        audio: Optional[types.Blob] = None,
        audio_stream_end: bool = False,
        **_: Any,
    ) -> None:
        """Buffer 16 kHz PCM16 frames; detect end-of-speech by energy.

        A turn ends after `silence_ms` of quiet (frame RMS < threshold) that
        FOLLOWS speech (RMS >= threshold), or immediately on audio_stream_end.
        Pure silence never makes a turn.
        """
        if audio_stream_end:
            if self._buf and self._spoken:
                pcm = bytes(self._buf)
                self._buf.clear()
                self._spoken = False
                self._quiet_ms = 0
                self._pending.put_nowait(("voice", pcm))
            return
        if audio is None or not getattr(audio, "data", None):
            return
        pcm = bytes(audio.data)
        self._buf.extend(pcm)
        ms = len(pcm) * 1000.0 / (SAMPLE_RATE * 2)
        if _rms(pcm) >= self.threshold:
            self._spoken = True
            self._quiet_ms = 0
        else:
            self._quiet_ms += ms
            if self._spoken and self._quiet_ms >= self.silence_ms:
                utter = bytes(self._buf)
                self._buf.clear()
                self._spoken = False
                self._quiet_ms = 0
                self._pending.put_nowait(("voice", utter))

    async def send_client_content(
        self,
        turns: Any = None,
        turn_complete: bool = True,
        **_: Any,
    ) -> None:
        """A typed/text turn (or a Hermes announcement): skips STT."""
        if not turn_complete:
            return
        text = " ".join(
            str(getattr(part, "text", "") or "")
            for turn in (turns or [])
            for part in (getattr(turn, "parts", None) or [])
        ).strip()
        if text:
            self._pending.put_nowait(("text", text))

    async def send_tool_response(self, function_responses: Any = None, **_: Any) -> None:
        """Accepted and ignored: the offline brain never emits tool calls."""

    def receive(self):
        """Yield the next turn's LiveServerMessages; the iterator ends at
        turn_complete, exactly like google-genai's session.receive()."""
        return self._receive_gen()

    # -- internals ---------------------------------------------------------

    async def _receive_gen(self):
        kind, payload = await self._pending.get()
        self._busy = True
        try:
            async for msg in self._turn(kind, payload):
                yield msg
        finally:
            self._busy = False

    async def _turn(self, kind, payload):
        if kind == "voice":
            # STT is sync and CPU-heavy: keep the loop responsive.
            heard = await asyncio.to_thread(self._stt, payload, self.lang)
            if not str(heard or "").strip():
                # Noise only (the STT's VAD removed everything): no reply.
                yield _content(turn_complete=True)
                return
            yield _content(input_transcription=types.Transcription(text=heard))
        else:
            heard = payload
        reply = await self._brain(heard, [dict(h) for h in self.history])
        reply = str(reply or "")
        yield _content(output_transcription=types.Transcription(text=reply))
        audio = await self._tts(reply)
        if audio:
            blob = types.Blob(data=bytes(audio),
                              mime_type=f"audio/pcm;rate={TTS_RATE}")
            # LiveServerMessage.data is a read-only property computed from
            # model_turn's inline_data, so the audio reaches both the
            # resp.data shortcut and the server_content path.
            yield _content(model_turn=types.Content(
                role="model", parts=[types.Part(inline_data=blob)]))
        yield _content(turn_complete=True)
        self.history.append({"role": "user", "text": heard})
        self.history.append({"role": "model", "text": reply})
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

    async def close(self) -> None:
        """Nothing to close; present for symmetry with the real session."""


def _content(**kwargs: Any) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(**kwargs)
    )


def offline_session_factory(
    stt: STTCallable,
    brain: BrainCallable,
    tts: TTSCallable,
    silence_ms: int = 700,
    threshold: float = 500,
) -> Callable[..., Any]:
    """Return a drop-in session_factory that yields a fresh OfflineSession.

    Ignores voice/resume_handle kwargs (offline mode has neither), so the
    server's reconnect loop can call it exactly like the online factory.
    """

    def factory(**_ignored: Any):
        @contextlib.asynccontextmanager
        async def cm():
            yield OfflineSession(
                stt, brain, tts, silence_ms=silence_ms, threshold=threshold
            )

        return cm()

    return factory
