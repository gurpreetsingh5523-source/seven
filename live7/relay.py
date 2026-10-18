"""Live relay: one Gemini Live session per browser connection.

Browser -> model:  text messages become user turns via send_client_content;
                   mic PCM16 mono 16 kHz frames go up via
                   send_realtime_input(audio=Blob(...)) — NEVER the deprecated
                   `media=` keyword (Gemini 3.1 closes the socket with 1007);
                   {"type": "audio_end"} -> send_realtime_input(audio_stream_end=True).
Model  -> browser: server_content becomes JSON events (transcript_in /
                   transcript_out / interrupted / stop_playback / state);
                   24 kHz PCM16 audio goes to send_audio (binary frame), with
                   exactly one {"type":"state","state":"speaking"} per turn;
                   tool calls go through ToolBridge -> send_tool_response.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from google.genai import types

SendEvent = Callable[[Dict[str, Any]], Awaitable[None]]
SendAudio = Callable[[bytes], Awaitable[None]]

MIC_MIME_TYPE = "audio/pcm;rate=16000"

# Vision privacy rules (M10): at most 2 frames/s per connection, never over
# 1.5 MB, JPEG or PNG only. Frames are never stored or forwarded elsewhere.
VIDEO_MAX_BYTES = 1_500_000
VIDEO_MAX_PER_SECOND = 2
VIDEO_MIME_TYPES = frozenset({"image/jpeg", "image/png"})

# Set while pump() is running: a tool task inside pump must not block the
# receive loop when an approvals channel exists (voice "haan" arrives via
# later transcripts). Direct handle_response() callers always await the task.
_in_pump: contextvars.ContextVar[bool] = contextvars.ContextVar("_in_pump", default=False)


SPEECH_RMS = 500.0   # PCM16 RMS; browser-AGC'd speech is ~1500-8000, room noise < 300


def _rms16(pcm: bytes) -> float:
    """RMS of little-endian PCM16 without numpy."""
    import array
    import math
    a = array.array("h")
    a.frombytes(bytes(pcm)[: len(pcm) - (len(pcm) % 2)])
    if not a:
        return 0.0
    return math.sqrt(sum(x * x for x in a) / len(a))


class LiveRelay:
    def __init__(
        self,
        session: Any,
        bridge: Any,
        send_event: SendEvent,
        send_audio: Optional[SendAudio] = None,
        approvals: Any = None,
        clock: Optional[Callable[[], float]] = None,
        quiet_s: float = 2.5,
        chatlog: Any = None,
    ) -> None:
        self._session = session
        self._bridge = bridge
        self._send = send_event
        self._send_audio = send_audio
        self._approvals = approvals
        self._chatlog = chatlog  # M20: local text log of the conversation (or None)
        self._tool_tasks: List[asyncio.Task] = []  # in-flight tool dispatches
        self._speaking = False  # one "speaking" state per model turn
        self._video_times: List[float] = []  # monotonic stamps, <=2/s window
        # Resilience (SPEC E10): latest resumable session handle, and the
        # server's go_away warning that it is about to drop the connection.
        self.resume_handle: Optional[str] = None
        self.go_away = False
        # M12 quiet-moment announcements: "now" moves forward on every sign
        # of life; finished Hermes jobs wait until nothing happened for
        # quiet_s before they are spoken (never interrupt the user).
        self._clock: Callable[[], float] = clock or time.monotonic
        self._quiet_s = float(quiet_s)
        self._last_activity = float("-inf")
        self._queue: List[Dict[str, Any]] = []  # pending announcements, FIFO

    async def handle_client(self, msg: Dict[str, Any]) -> None:
        """One message from the browser. Anything unknown -> error event."""
        if isinstance(msg, dict):
            if msg.get("type") == "file":
                # M12: the user dropped/picked a file in the page.
                await self.handle_file(msg)
                return
            if msg.get("type") == "text":
                self.note_activity()
                text = str(msg.get("text", ""))
                if self._chatlog is not None:
                    self._chatlog.write("typed", text=text)
                turn = types.Content(
                    role="user", parts=[types.Part(text=text)]
                )
                await self._session.send_client_content(turns=[turn], turn_complete=True)
                return
            if msg.get("type") == "audio_end":
                # Browser says the user stopped talking; close the mic stream
                # so the model finalises its turn.
                await self._session.send_realtime_input(audio_stream_end=True)
                return
            if msg.get("type") == "approval":
                # Click path (SPEC E7): only the page's Yes/No buttons or the
                # user's own speech may settle a pending approval — the model
                # never can.
                if self._approvals is not None:
                    self._approvals.resolve(
                        str(msg.get("action_id", "")), bool(msg.get("approved"))
                    )
                return
        bad = msg.get("type") if isinstance(msg, dict) else type(msg).__name__
        await self._send({"type": "error", "error": f"unsupported client message: {bad!r}"})

    async def handle_audio(self, pcm: bytes) -> None:
        """One mic frame (PCM16 mono 16 kHz) from the browser -> the model.

        Uses send_realtime_input(audio=Blob). The deprecated `media=` /
        `media_chunks` path is rejected by Gemini 3.1 with close code 1007.
        """
        # The user speaking is the strongest "busy" signal. Input transcriptions arrive late
        # (often only at end of turn), so a long monologue looked "quiet" and 7 interrupted it
        # with a Hermes announcement (seen live 2026-10-06). Measure mic energy directly.
        if _rms16(pcm) >= SPEECH_RMS:
            self.note_activity()
        await self._session.send_realtime_input(
            audio=types.Blob(data=bytes(pcm), mime_type=MIC_MIME_TYPE)
        )

    async def handle_video(self, jpeg: bytes, mime: str = "image/jpeg") -> bool:
        """One screen/camera frame from the browser -> the model as video.

        Privacy rules (M10): returns False and sends NOTHING when the frame
        is over 1.5 MB, the mime is not JPEG/PNG, or the connection is over
        the 2 frames/second limit (measured with time.monotonic()). Frames
        are never stored; they go only into the live session.
        """
        data = bytes(jpeg)
        if len(data) > VIDEO_MAX_BYTES or mime not in VIDEO_MIME_TYPES:
            return False
        now = time.monotonic()
        window = self._video_times
        window[:] = [t for t in window if now - t < 1.0]
        if len(window) >= VIDEO_MAX_PER_SECOND:
            return False
        window.append(now)
        await self._session.send_realtime_input(
            video=types.Blob(data=data, mime_type=mime)
        )
        return True

    async def handle_response(self, resp: types.LiveServerMessage) -> None:
        """One message from the Live model."""
        update = resp.session_resumption_update
        if update is not None and update.resumable and update.new_handle:
            # Only resumable handles may seed a reconnect (SPEC E10).
            self.resume_handle = update.new_handle
        if resp.go_away is not None:
            # Gemini is about to drop us; the server reconnects proactively
            # once the current turn completes.
            self.go_away = True
        content = resp.server_content
        if content is not None:
            if getattr(content, "interrupted", False):
                await self._send({"type": "interrupted"})
                # Tell the page to flush its playback queue immediately (E3).
                await self._send({"type": "stop_playback"})
                self._speaking = False
            if content.input_transcription and content.input_transcription.text:
                text = content.input_transcription.text
                self.note_activity()
                if self._chatlog is not None:
                    self._chatlog.write("user", text=text)
                # Voice approval path FIRST: the user's own speech is the only
                # spoken channel that may settle a pending approval (the model's
                # output transcription is never offered to it).
                if self._approvals is not None:
                    self._approvals.on_user_transcript(text)
                await self._send({"type": "transcript_in", "text": text})
            if content.output_transcription and content.output_transcription.text:
                self.note_activity()
                if self._chatlog is not None:
                    self._chatlog.write("seven", text=content.output_transcription.text)
                await self._send(
                    {"type": "transcript_out", "text": content.output_transcription.text}
                )
            turn = getattr(content, "model_turn", None)
            if turn is not None:
                for part in turn.parts or []:
                    blob = getattr(part, "inline_data", None)
                    if blob is not None and blob.data:
                        self.note_activity()
                        if self._send_audio is not None:
                            if not self._speaking:
                                await self._send({"type": "state", "state": "speaking"})
                                self._speaking = True
                            await self._send_audio(bytes(blob.data))
                    text = getattr(part, "text", None)
                    if text:
                        self.note_activity()
                        if self._chatlog is not None:
                            self._chatlog.write("seven", text=text)
                        # TEXT-modality model turns also reach the browser.
                        await self._send({"type": "transcript_out", "text": text})
            if getattr(content, "turn_complete", False):
                self._speaking = False
                await self._send({"type": "state", "state": "listening"})
            if (getattr(content, "turn_complete", False)
                    or getattr(content, "interrupted", False)):
                # The user's utterance is over: one spoken "haan" may settle
                # at most ONE pending approval (Gemini splits one word into
                # several input-transcription fragments).
                if self._approvals is not None:
                    end = getattr(self._approvals, "end_utterance", None)
                    if end is not None:
                        end()

        tool_call = resp.tool_call
        if tool_call and tool_call.function_calls:
            await self._send({"type": "state", "state": "executing"})
            # Dispatch in a task: a MEDIUM/HIGH call waits on a user approval
            # and the receive loop must keep processing transcripts meanwhile.
            # handle_response still awaits the task, so callers that run it
            # sequentially keep their ordering; pump() awaits it only for
            # approval-free relays (see _pump_wait).
            task = asyncio.create_task(self._dispatch_tool_call(tool_call.function_calls))
            self._tool_tasks.append(task)
            task.add_done_callback(self._tool_task_done)
            if self._pump_wait():
                await task

    def _pump_wait(self) -> bool:
        """pump() waits for a tool task only when it cannot resolve by voice.

        With no approvals channel, user speech cannot unblock a gated call, so
        sequential handling is safe (and keeps M1-style callers simple). With
        approvals wired, pump() returns immediately and keeps receiving.
        Direct handle_response() callers always await the task regardless.
        """
        return self._approvals is None or not _in_pump.get()

    def _tool_task_done(self, task: asyncio.Task) -> None:
        with contextlib.suppress(ValueError):
            self._tool_tasks.remove(task)
        if not task.cancelled():
            task.exception()  # consume (and log) instead of "never retrieved"

    async def _dispatch_tool_call(self, function_calls: List[Any]) -> None:
        responses = []
        for fc in function_calls:
            self.note_activity()
            await self._send({"type": "tool_started", "name": fc.name})
            result = await self._bridge.dispatch(fc.name, dict(fc.args or {}))
            if self._chatlog is not None:
                args = dict(fc.args or {})
                if isinstance(result, dict):
                    ok = bool(result.get("ok", True))
                    error = str(result.get("error") or "")
                else:
                    ok, error = True, ""
                self._chatlog.write("tool", name=fc.name, args=args, ok=ok, error=error)
            await self._send({"type": "tool_result", "name": fc.name, "result": result})
            responses.append(
                types.FunctionResponse(id=fc.id, name=fc.name, response=result)
            )
        await self._session.send_tool_response(function_responses=responses)

    async def handle_launch(self) -> None:
        """M15L: the app just opened on ?launch=1 -> greet the user by voice.

        ONE user turn; 7 says hello in one short sentence in the user's
        language and OFFERS the daily briefing (never reads it unasked).
        """
        text = (
            "[Launch] The user just opened 7 Live. Greet them in ONE short "
            "sentence in their language, then offer the daily briefing "
            "(ask if they want it) — do NOT read the briefing unless they "
            "say yes."
        )
        turn = types.Content(role="user", parts=[types.Part(text=text)])
        await self._session.send_client_content(turns=[turn], turn_complete=True)

    async def say(self, text: str) -> None:
        """Push a short user turn into the live session so 7 speaks it."""
        turn = types.Content(role="user", parts=[types.Part(text=text)])
        await self._session.send_client_content(turns=[turn], turn_complete=True)

    async def ask_approval(self, question: str) -> None:
        """Ask the user an approval question WITHOUT sounding like a request.

        Sending the bare question as a user turn ("I want to run app_open
        (name chrome). Should I go ahead?") looks to the model like the user
        asking for Chrome — it calls the tool again and the approval loop
        storms (seen live 2026-10-08). So the turn is an INSTRUCTION: 7 must
        only speak the question aloud and must not call any tool.
        """
        text = (
            f"[Approval] Ask the user this question out loud, in one short "
            f"sentence in the user's language: \"{question}\" — Do not call "
            f"any tool right now; only ask, then wait for their spoken answer."
        )
        turn = types.Content(role="user", parts=[types.Part(text=text)])
        await self._session.send_client_content(turns=[turn], turn_complete=True)

    async def pause(self, minutes: int) -> None:
        """M13 assistant_pause: tell the page to mute the mic for <minutes>."""
        await self._send({"type": "paused", "minutes": int(minutes)})

    async def send_event(self, event: Dict[str, Any]) -> None:
        """Send one JSON event to the browser (approval cards, etc.)."""
        await self._send(event)

    # ------------------------------------------------------------------ #
    # M12: quiet-moment announcements, work timeline, dropped files
    # ------------------------------------------------------------------ #

    def note_activity(self) -> None:
        """Mark "now" as busy (user speech, model audio/text, tool calls)."""
        self._last_activity = self._clock()

    def is_idle(self) -> bool:
        """True when nothing happened for >= quiet_s and no tool is running."""
        if any(not t.done() for t in self._tool_tasks):
            return False
        return (self._clock() - self._last_activity) >= self._quiet_s

    async def tick(self) -> None:
        """Quiet moment? Speak ONE queued announcement (FIFO), never more."""
        if self._queue and self.is_idle():
            await self._speak_job(self._queue.pop(0))

    async def job_event(self, job: Dict[str, Any]) -> None:
        """One timeline event for the page's jobs panel."""
        started = job.get("started_at")
        finished = job.get("finished_at")
        if finished is not None and started:
            seconds: Any = round(float(finished) - float(started), 1)
        elif started:
            seconds = round(time.time() - float(started), 1)
        else:
            seconds = 0.0
        await self._send({
            "type": "job",
            "job_id": job.get("job_id") or job.get("id"),
            "task": job.get("task", ""),
            "status": job.get("status", ""),
            "seconds": seconds,
        })

    async def handle_file(self, msg: Dict[str, Any]) -> None:
        """One file the user dropped or picked in the page (M12).

        Images go straight to the model as video=Blob frames — an explicit
        user action, allowed even when screen sharing is off. Text files
        become one user turn "[User shared file <name>]\\n<first 8000 chars>".
        Everything is saved under the uploads dir first (~/.seven/uploads,
        or SEVEN_UPLOADS); unsupported types only get an error event.
        """
        from live7.files import ingest_file

        name = str(msg.get("name") or "upload")
        try:
            data = base64.b64decode(str(msg.get("data_b64") or ""))
        except (ValueError, TypeError):
            await self._send({"type": "error", "error": f"bad file data for {name!r}"})
            return
        uploads_dir = Path(os.environ["SEVEN_UPLOADS"]) if os.environ.get(
            "SEVEN_UPLOADS"
        ) else None
        result = ingest_file(name, str(msg.get("mime") or ""), data, uploads_dir)
        if not result.get("ok"):
            await self._send({
                "type": "error",
                "error": result.get("error", f"could not ingest {name!r}"),
            })
            return
        kind = result.get("kind")
        if kind == "image":
            # Explicit user action: allowed even when screen sharing is off.
            mime = str(msg.get("mime") or "image/png")
            await self._session.send_realtime_input(
                video=types.Blob(data=data, mime_type=mime)
            )
        elif kind == "text":
            text = str(result.get("text") or "")[:8000]
            turn = types.Content(
                role="user",
                parts=[types.Part(text=f"[User shared file {name}]\n{text}")],
            )
            await self._session.send_client_content(turns=[turn], turn_complete=True)
        else:
            await self._send({
                "type": "error",
                "error": f"unsupported file type: {name}",
            })
            return
        await self._send({
            "type": "file_received",
            "name": name,
            "kind": kind,
            "chars": len(data),
        })

    async def announce_hermes(self, job: Dict[str, Any]) -> None:
        """Speak a finished Hermes job in a quiet moment (SPEC E6, M12).

        The browser's timeline is updated AT ONCE (hermes_done events), but
        the spoken turn waits until the user is idle (is_idle / tick) so a
        finished job never interrupts. The spoken summary is at most 300
        characters of the reply plus a hint that hermes_result has the full
        answer.
        """
        reply = str(job.get("reply") or "").strip()
        if not reply:
            reply = str(job.get("error") or f"status: {job.get('status')}")
        summary = reply[:300]
        job_id = job.get("job_id") or job.get("id")
        await self._send({
            "type": "hermes_done",
            "job_id": job_id,
            "status": job.get("status"),
            "summary": summary,
        })
        if self.is_idle():
            await self._speak_job({"job_id": job_id, "status": job.get("status"),
                                   "summary": summary})
        else:
            self._queue.append({"job_id": job_id, "status": job.get("status"),
                                "summary": summary})

    async def _speak_job(self, item: Dict[str, Any]) -> None:
        """Push the queued announcement into the live session as one user turn."""
        text = (
            f"[Hermes finished] (status: {item.get('status')}, "
            f"job: {item.get('job_id')}) {item.get('summary', '')} "
            f"— say this briefly; the full answer is available via hermes_result."
        )
        turn = types.Content(role="user", parts=[types.Part(text=text)])
        await self._session.send_client_content(turns=[turn], turn_complete=True)

    async def pump(self) -> None:
        """Handle every message the session yields, turn after turn.

        google-genai's `session.receive()` iterator ENDS at each turn_complete,
        so it must be re-entered for the next turn; otherwise 7 answers once
        and then goes silent. A session that yields nothing at all is treated
        as closed (avoids a busy loop on a dead session)."""
        token = _in_pump.set(True)
        try:
            while True:
                got_any = False
                async for resp in self._session.receive():
                    got_any = True
                    await self.handle_response(resp)
                if not got_any:
                    return
        finally:
            _in_pump.reset(token)
            # Session ended: don't leave a pending approval (or its tool
            # response) hanging past the connection.
            pending = [t for t in self._tool_tasks if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
