"""7 Live server: loopback-only HTTP + WebSocket front end (SPEC §3, §7.5).

- GET /            -> the static/live.html page (via websockets' process_request)
- WebSocket /live  -> one Gemini Live session per browser connection:
    text frames   = JSON client messages  -> relay.handle_client
    binary frames = mic PCM16 16 kHz      -> relay.handle_audio,
                  except a binary frame whose first byte is 0x56 ('V'):
                  the rest is a JPEG/PNG video frame (M10) -> relay.handle_video,
                  but ONLY while this connection's vision state is on
                  ({"type":"vision","source":"screen"|"camera"} — echoed back);
                  frames received while the source is "off" are dropped.
    events out    = JSON text frames; model PCM16 24 kHz -> binary frames
- Only this server's own Origin (http://127.0.0.1:<port> /
  http://localhost:<port>) is allowed to open the socket (blocks other sites).
- Everything binds to loopback only; assert_loopback() refuses otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

from google.genai import types
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response
from websockets.protocol import State as WsState

from live7.relay import LiveRelay

log = logging.getLogger("live7")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7879  # 7878 is the old Agent 7 UI

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
PAGE_PATH = STATIC_DIR / "live.html"
CORE_PATH = STATIC_DIR / "live_core.js"
WS_PATH = "/live"

DEFAULT_MODEL = "models/gemini-3.1-flash-live-preview"

# Prebuilt Gemini Live voices (a picker subset; extend freely).
# https://ai.google.dev/gemini-api/docs/live-api#voices
VOICES = [
    "Charon",   # deep, calm
    "Puck",     # friendly
    "Kore",     # warm, neutral
    "Fenrir",   # authoritative
    "Aoede",    # bright, clear
    "Orus",     # low, smooth
    "Iapetus",  # resonant
    "Enceladus",# airy
]

_DEFAULT_VOICE = "Charon"
_VOICE_SET = frozenset(VOICES)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def assert_loopback(host: str) -> None:
    """Raise ValueError unless `host` is a loopback address."""
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"refusing to bind {host!r}: 7 Live listens on loopback only "
            f"(allowed: {sorted(_LOOPBACK_HOSTS)})"
        )


def live_model() -> str:
    """Model for the Live session; override with SEVEN_LIVE_MODEL."""
    return os.environ.get("SEVEN_LIVE_MODEL", DEFAULT_MODEL)


def _resolve_voice(voice: Optional[str]) -> str:
    """Pick the TTS voice: explicit arg > SEVEN_VOICE env > default Charon."""
    chosen = voice or os.environ.get("SEVEN_VOICE") or _DEFAULT_VOICE
    if chosen not in _VOICE_SET:
        raise ValueError(
            f"unknown voice {chosen!r}: pick one of {sorted(_VOICE_SET)}"
        )
    return chosen


def _transcription_langs() -> Optional[List[str]]:
    """Input-transcription language hints (M20): the user speaks Punjabi and
    English only, so hinting ["pa-IN", "en-IN"] stops Gemini guessing Hindi/
    Urdu. SEVEN_LANGS="en-US,pa-IN" overrides; SEVEN_LANGS="auto" -> no hint.
    """
    raw = (os.environ.get("SEVEN_LANGS") or "").strip()
    if raw.lower() == "auto":
        return None
    if not raw:
        return ["pa-IN", "en-IN"]
    return [code.strip() for code in raw.split(",") if code.strip()] or None


def build_live_config(
    declarations: List[dict],
    system_instruction: str,
    voice: Optional[str] = None,
    resume_handle: Optional[str] = None,
) -> types.LiveConnectConfig:
    """AUDIO-modality Live config: captions, voice, tools, resilience (SPEC E10).

    session_resumption carries the last resumption handle (None on the first
    session — Gemini then issues one). context_window_compression uses a
    sliding window so long conversations never hit the context limit, and the
    VAD silence_duration_ms (SEVEN_SILENCE_MS, default 500) shortens the
    end-of-speech wait for faster replies (SPEC E2).
    """
    voice_name = _resolve_voice(voice)
    silence_ms = int(os.environ.get("SEVEN_SILENCE_MS") or 500)
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=_transcription_langs()
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        tools=[{"function_declarations": declarations}],
        system_instruction=system_instruction,
        session_resumption=types.SessionResumptionConfig(handle=resume_handle),
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow()
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                silence_duration_ms=silence_ms
            )
        ),
    )


def _backoff_delay(attempt: int) -> float:
    """Reconnect delay for attempt N (1-based): 0.5, 1, 2, 4, 8 s (cap 8 s).

    SEVEN_BACKOFF_SCALE (float, default 1) multiplies every delay; tests set
    it to 0.01 so the retry loop runs in milliseconds.
    """
    try:
        scale = float(os.environ.get("SEVEN_BACKOFF_SCALE") or 1)
    except ValueError:
        scale = 1.0
    return min(0.5 * (2 ** (attempt - 1)), 8.0) * scale


def _online_retry_s() -> float:
    """While on the offline fallback, try Gemini Live again every N seconds
    (SEVEN_ONLINE_RETRY_S, default 60). One dropped connect must not leave 7
    offline for the rest of the day."""
    try:
        return max(float(os.environ.get("SEVEN_ONLINE_RETRY_S") or 60), 0.05)
    except ValueError:
        return 60.0


class LiveServer:
    """Thin wrapper so tests get `server.port` and an async `server.close()`."""

    def __init__(self, ws_server: Any, host: str, port: int) -> None:
        self._ws = ws_server
        self.host = host
        self.port = port

    async def close(self) -> None:
        self._ws.close()
        await self._ws.wait_closed()

    async def __aenter__(self) -> "LiveServer":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


def _reject(connection: ServerConnection, status: int, text: str) -> Response:
    return connection.respond(status, text)


async def start_server(
    session_factory: Callable[[], Any],
    bridge: Any,
    host: str = DEFAULT_HOST,
    port: int = 0,
    on_relay: Optional[Callable[[Optional[LiveRelay]], None]] = None,
    approvals: Any = None,
    offline_factory: Optional[Callable[[], Any]] = None,
) -> LiveServer:
    """Bind the 7 Live server on loopback and return it.

    session_factory: async context manager factory yielding one Live session
    per browser connection (a fake in tests, google.genai .aio.live.connect
    in production).
    on_relay: optional hook called with each new LiveRelay as it connects
    and with None when the browser disconnects — lets the Hermes bridge
    announce finished jobs into whichever session is currently live.
    approvals: optional VoiceApprovals shared by all connections; voice
    answers and page clicks settle pending MEDIUM/HIGH actions (SPEC E7).
    offline_factory: optional local-only fallback (SPEC E8). When the online
    session_factory fails to CONNECT (raises entering its context) the
    offline session is used immediately and the browser gets
    {"type":"mode","mode":"offline"}; a successful online connection gets
    {"type":"mode","mode":"online"}. SEVEN_MODE=offline forces offline from
    the start (no online connect is attempted). If the fallback is never
    given, reconnect/backoff behaves exactly as before.
    """
    assert_loopback(host)

    page_html = PAGE_PATH.read_bytes()
    core_js = CORE_PATH.read_bytes()
    boot_html = (STATIC_DIR / "boot.html").read_bytes()

    active_browser: List[Any] = [None]

    # M20: keep a local text record of every conversation so problems can be
    # checked afterwards. Local files only (0700 dir, 0600 files); nothing
    # leaves the Mac. SEVEN_CHATLOG=0 disables it.
    chatlog = None
    if (os.environ.get("SEVEN_CHATLOG") or "").strip() != "0":
        try:
            from live7.chatlog import ChatLog
            chatlog = ChatLog(Path.home() / ".seven" / "chats")
        except Exception:
            log.exception("chat log unavailable; continuing without it")
            chatlog = None

    async def process_request(
        connection: ServerConnection, request: Request
    ) -> Optional[Response]:
        req_path = request.path.split("?")[0]
        if req_path == WS_PATH:
            # WebSocket handshake: allow only this server's own origins so no
            # other website can open the socket from a browser tab.
            local_port = connection.local_address[1]
            allowed = {
                f"http://127.0.0.1:{local_port}",
                f"http://localhost:{local_port}",
            }
            origin = request.headers.get("Origin")
            if origin not in allowed:
                return _reject(connection, 403, "forbidden origin")
            return None  # proceed to the handler
        if req_path in ("/", "/index.html"):
            return Response(
                200,
                "OK",
                Headers(
                    {
                        "Content-Type": "text/html; charset=utf-8",
                        "Content-Length": str(len(page_html)),
                        "Cache-Control": "no-store",
                    }
                ),
                page_html,
            )
        if req_path == "/boot.html":
            return Response(
                200,
                "OK",
                Headers(
                    {
                        "Content-Type": "text/html; charset=utf-8",
                        "Content-Length": str(len(boot_html)),
                        "Cache-Control": "no-store",
                    }
                ),
                boot_html,
            )
        if req_path == "/live_core.js":
            return Response(
                200,
                "OK",
                Headers(
                    {
                        "Content-Type": "application/javascript; charset=utf-8",
                        "Content-Length": str(len(core_js)),
                        "Cache-Control": "no-store",
                    }
                ),
                core_js,
            )
        return _reject(connection, 404, "not found")

    async def handler(connection: ServerConnection) -> None:
        # One 7 at a time: two open windows meant two Gemini sessions answering the
        # same mic at once ("the voice is doubled", 2026-10-06). A new window takes
        # over and the previous one is told why and closed.
        previous = active_browser[0]
        active_browser[0] = connection
        if previous is not None:
            with contextlib.suppress(Exception):
                await previous.send(json.dumps({"type": "replaced",
                                                "reason": "7 was opened in another window"}))
                await previous.close(code=4000, reason="replaced by a newer window")
        try:
            await connection.send(json.dumps({"type": "state", "state": "listening"}))
            voice: Optional[str] = None
            resume_handle: Optional[str] = None
            first_session = True
            attempts = 0                     # consecutive failed (re)connects
            go_away_reconnect = False        # proactive reconnect after this turn
            # Vision privacy (M10): video frames are forwarded ONLY while the
            # user has screen/camera sharing on for THIS connection.
            vision_source = "off"
            # SPEC E8: the user asked for offline explicitly -> never dial out.
            offline_forced = (os.environ.get("SEVEN_MODE") or "").strip().lower() == "offline"
            # Offline is a fallback for when Gemini Live cannot be REACHED. A live
            # online session that drops (e.g. 1008) reconnects online, and while on
            # the fallback we keep retrying online every _online_retry_s().
            online_failed = False
            while True:                      # re-entered on set_voice / session loss
                reconnect = False
                retry_online = False
                factory_kwargs: dict = {}
                if not first_session and voice is not None:
                    factory_kwargs["voice"] = voice
                if resume_handle is not None:
                    factory_kwargs["resume_handle"] = resume_handle
                first_session = False
                session_ok = False
                session_error: Optional[BaseException] = None
                # Offline when the user forced it, or when one is wired up and
                # the online connect itself failed on this connection.
                use_offline = offline_factory is not None and (
                    offline_forced or online_failed
                )
                if use_offline:
                    assert offline_factory is not None
                    active_factory = offline_factory
                else:
                    active_factory = session_factory
                mode = "offline" if use_offline else "online"
                try:
                    async with active_factory(**factory_kwargs) as session:
                        session_ok = True
                        attempts = 0         # a live session resets the counter
                        await connection.send(
                            json.dumps({"type": "mode", "mode": mode}))

                        async def send_event(event: dict) -> None:
                            await connection.send(json.dumps(event, ensure_ascii=False))

                        async def send_audio(pcm: bytes) -> None:
                            await connection.send(pcm)

                        relay = LiveRelay(session, bridge, send_event, send_audio=send_audio,
                                          approvals=approvals, chatlog=chatlog)
                        if on_relay is not None:
                            on_relay(relay)
                        pump = asyncio.create_task(relay.pump())

                        async def ticker():
                            # M12: ~2x a second, speak ONE queued Hermes
                            # announcement if the user has gone quiet.
                            while True:
                                await asyncio.sleep(0.5)
                                try:
                                    await relay.tick()
                                except Exception:
                                    log.exception("relay tick failed")

                        async def browser_messages():
                            try:
                                async for raw in connection:
                                    yield raw
                            except Exception:
                                # Websocket dropped: stop iterating.
                                return

                        messages = browser_messages()
                        tick_task = asyncio.create_task(ticker())
                        retry_at = (asyncio.get_running_loop().time() + _online_retry_s()
                                    if mode == "offline" and not offline_forced else None)
                        msg_task = None
                        try:
                            while True:
                                if msg_task is None:
                                    msg_task = asyncio.ensure_future(messages.__anext__())
                                timeout = None
                                if retry_at is not None:
                                    timeout = max(retry_at - asyncio.get_running_loop().time(), 0)
                                done, _ = await asyncio.wait(
                                    {pump, msg_task}, timeout=timeout,
                                    return_when=asyncio.FIRST_COMPLETED)
                                if not done:
                                    # Offline fallback: time to try Gemini Live again,
                                    # but never in the middle of an offline reply.
                                    if getattr(session, "idle", True):
                                        msg_task.cancel()
                                        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                                            await msg_task
                                        retry_online = True
                                        break
                                    retry_at = asyncio.get_running_loop().time() + 1.0
                                    continue
                                if pump in done:
                                    # Live session ended (or died): cancel the
                                    # pending read so __anext__ is never awaited
                                    # concurrently on re-entry.
                                    msg_task.cancel()
                                    with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                                        await msg_task
                                    session_error = pump.exception()
                                    break
                                try:
                                    raw = msg_task.result()
                                except StopAsyncIteration:
                                    break  # browser went away
                                msg_task = None
                                if relay.go_away:
                                    # Gemini warned it is about to drop us; the
                                    # current turn just completed -> reconnect now.
                                    go_away_reconnect = True
                                    break
                                if isinstance(raw, (bytes, bytearray)):
                                    if raw[:1] == b"V":
                                        # M10 vision: 'V' + JPEG/PNG. Privacy:
                                        # forwarded ONLY while sharing is on;
                                        # in offline mode frames simply drop.
                                        if mode != "offline" and vision_source != "off":
                                            await relay.handle_video(bytes(raw[1:]))
                                    else:
                                        await relay.handle_audio(bytes(raw))
                                else:
                                    try:
                                        msg = json.loads(raw)
                                    except json.JSONDecodeError:
                                        await send_event(
                                            {"type": "error", "error": "invalid JSON frame"}
                                        )
                                        continue
                                    if isinstance(msg, dict) and msg.get("type") == "set_voice":
                                        new_voice = msg.get("voice")
                                        if new_voice in _VOICE_SET:
                                            voice = new_voice
                                            await send_event({"type": "voice", "voice": voice})
                                            reconnect = True   # drop this session, re-enter with new voice
                                            break
                                        await send_event(
                                            {"type": "error",
                                             "error": f"unknown voice {new_voice!r}"}
                                        )
                                        continue
                                    if isinstance(msg, dict) and msg.get("type") == "vision":
                                        # M10: remember this connection's sharing
                                        # state and confirm it back to the page.
                                        source = msg.get("source")
                                        if source in ("screen", "camera", "off"):
                                            vision_source = source
                                        await send_event(
                                            {"type": "vision", "source": vision_source})
                                        continue
                                    if isinstance(msg, dict) and msg.get("type") == "launch":
                                        # M15L: the page opened with ?launch=1;
                                        # 7 greets the user by voice.
                                        await relay.handle_launch()
                                        continue
                                    await relay.handle_client(msg)
                        finally:
                            pump.cancel()
                            tick_task.cancel()
                            if relay.resume_handle is not None:
                                resume_handle = relay.resume_handle
                            with contextlib.suppress(asyncio.CancelledError):
                                await pump
                            with contextlib.suppress(asyncio.CancelledError):
                                await tick_task
                        if session_error is not None:
                            raise session_error
                except Exception:
                    # The Live session dropped while the browser is still
                    # connected (SPEC E10): reconnect with backoff.
                    log.exception("live session lost; reconnecting")
                if mode == "online":
                    # Only a failed CONNECT sends us to the offline fallback.
                    online_failed = not session_ok
                if retry_online:
                    online_failed = False
                    continue                   # try Gemini Live again now
                if reconnect:
                    continue                   # set_voice: immediate, no backoff
                if connection.state is not WsState.OPEN:  # browser went away
                    break
                if go_away_reconnect:
                    go_away_reconnect = False
                    continue                   # proactive reconnect, no backoff
                attempts += 1
                if attempts > 6:
                    with contextlib.suppress(Exception):
                        await connection.send(json.dumps(
                            {"type": "error", "error": "live session unavailable"}))
                    break
                await connection.send(json.dumps({"type": "state", "state": "reconnecting"}))
                # First drop of a live session reconnects at once; subsequent
                # failures back off 0.5, 1, 2, 4, 8 s (capped, env-scaled).
                if not (session_ok and attempts == 1):
                    await asyncio.sleep(_backoff_delay(attempts))
                if connection.state is not WsState.OPEN:
                    break
                await connection.send(json.dumps({"type": "state", "state": "listening"}))
        except Exception:  # never let a handler traceback leak to the client
            log.exception("live session failed")   # ...but always keep it in our log
            with contextlib.suppress(Exception):
                await connection.send(
                    json.dumps({"type": "error", "error": "session failed"})
                )
        finally:
            if on_relay is not None:
                on_relay(None)

    # A dropped 10 MB file arrives as ~13.4 MB of base64 JSON; the websockets default (1 MiB)
    # closed the connection with 1009 for anything over ~750 KB.
    ws_server = await serve(handler, host, port, process_request=process_request,
                            max_size=16 * 1024 * 1024)
    bound_port = list(ws_server.sockets)[0].getsockname()[1]
    return LiveServer(ws_server, host, bound_port)
