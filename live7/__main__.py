"""`python -m live7` — run the real 7 Live server against Gemini Live.

API key: env GOOGLE_API_KEY, else the GOOGLE_API_KEY line in ~/.hermes/.env.
The key is never printed, logged, or sent to the browser (SPEC §7.6).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sentinel_orchestrator import (
    ApprovalGate,
    AuditLog,
    LocalSandbox,
    PolicyEngine,
    SecretsBroker,
)

from live7.approvals import VoiceApprovals
from live7.hermes_bridge import HermesBridge
from live7.offline import offline_session_factory
from live7.registry import build_live_registry
from live7.server import build_live_config, live_model, start_server
from live7.tools import ToolBridge

log = logging.getLogger("live7.main")

STATE_DIR = Path.home() / ".seven"
HERMES_JOBS_DIR = STATE_DIR / "hermes_jobs"
HERMES_ENV = Path.home() / ".hermes" / ".env"
PORT = 7879  # 7878 is the old Agent 7 UI

# Offline brain: any OpenAI-compatible server on loopback (default: mlx_lm.server
# with Qwen3-4B on :8097). SEVEN_OFFLINE_BRAIN_URL / SEVEN_OFFLINE_BRAIN_MODEL
# put another brain in front.
OFFLINE_BRAIN_URL = "http://127.0.0.1:8097"
OFFLINE_BRAIN_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
OFFLINE_BRAINS = [(OFFLINE_BRAIN_URL, OFFLINE_BRAIN_MODEL)]

# SPEC §6, offline variant — the local brain answers from chat text only.
OFFLINE_SYSTEM = """\
You are 7, a private voice-first assistant that runs entirely on the user's \
own Mac and is currently OFFLINE — no internet, no cloud model. Keep every \
reply short: one or two sentences, spoken out loud. Reply in the language \
the user uses: proper Gurmukhi if they speak or write Punjabi, otherwise \
English. Do not claim you checked email, calendar, files, or the system — \
offline you cannot call tools. If a turn starts with "[Hermes finished]" \
read the result out loud without being asked.\
"""

# SPEC §6.
SYSTEM_INSTRUCTION = """\
You are 7, a private voice-first assistant that runs entirely on the user's \
own Mac. You speak out loud, so keep every reply short — one or two sentences \
unless the user asks for more. The user only speaks English and Punjabi; use \
one language per reply — English when they spoke English, Punjabi in proper \
Gurmukhi when they spoke Punjabi (romanised Punjabi counts as Punjabi, and a \
transcript that looks like Hindi or Urdu is Punjabi misheard); never reply \
in Hindi, Urdu or any other language, and never mix two languages in one \
reply. Call a tool instead of guessing: never claim you \
checked email, calendar, files, or the system unless a tool actually returned \
that data. When the user mentions Hermes — speech may render it as "her mees", \
"hermis", "harmes", "Khamis", or "Hamies" — that is the user's own agent; \
delegate real work to it with ask_hermes right away, keep talking while it \
works, and when a turn arrives that starts with "[Hermes finished]" read the \
result out loud without being asked; hermes_result shows the latest jobs if \
the user asks how it went. When the user shares their screen or camera, frames \
arrive as video: describe briefly what is visible and help with whatever is on \
it; when sharing is off, say you cannot see the screen or camera and never \
claim otherwise. Your other \
abilities, briefly: weather_report for the weather in a city; web_search and \
smart_search to search the web and web_fetch to read a page; globe_distance \
for the distance between two cities; system_monitor for CPU, RAM and disk; \
app_open, site_open, desktop_open_app, file_organizer and browser_task all \
change the outside world — call them directly when the user asks. 7's own \
approval flow asks the user and waits for a clear yes by itself, so never \
ask for confirmation yourself first and never bypass it. undo reverses the \
last reversible \
action. memory_write stores a note and recall finds what you remember; \
team_status and team_ask talk to the agent team, code_review reviews code, \
and page_draft/page_list/page_read plus page_approve/page_reject manage the \
user's written pages. You can control this Mac with mac_volume, mac_brightness, mac_info, \
clipboard_read, clipboard_write, window_control, music_control, \
reminder_add, reminders_list, reminder_delete, reminder_complete, \
files_find, file_op, file_trash, type_text, \
key_shortcut, app_quit and assistant_pause; file actions stay inside the \
user's home, and the approval tools (type_text, key_shortcut, file_op, \
app_quit, file_trash) need a clear yes through the approval flow before \
anything runs. You can make and read documents with doc_create, doc_read, doc_edit and \
doc_to_pdf (Word, Excel, slides, PDF, saved in Documents/7; a finished document \
opens in front of the user automatically — say so, and pass "show": false only if \
they ask not to see it), manage the \
calendar with calendar_list and calendar_add, and give the morning summary \
with daily_briefing — a briefing only ever reports what those tools actually \
returned, and names anything unavailable instead of inventing it. For news, \
show headlines right on your screen with news_show (feeds "world" or "india") \
and live TV news in your window with news_live (Al Jazeera English, DW News, \
Sky News, France 24 English), and hide them again with news_close. \
For news, do not open a browser (no web_search or site_open) unless the user \
asks for a specific web page; deep news research goes to ask_hermes. If an \
action needs confirmation, the approval flow asks the user by itself — \
never ask for confirmation yourself. If an action is not approved, do not retry it — \
ask the user what they want instead. If no tool can do it (what the user asked), \
say so plainly in one sentence and never use a different tool instead — for \
example never use file_trash for reminders or calendar events. File tools take \
folder names like "Downloads" \
or "Desktop" or full paths under the user's home folder, which is {Path.home()}.\
""".replace("{Path.home()}", str(Path.home()))


def load_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()
    if HERMES_ENV.exists():
        for line in HERMES_ENV.read_text().splitlines():
            line = line.strip()
            if line.startswith("GOOGLE_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    # No key: 7 still starts. The online connect fails at once and the server
    # falls back to offline mode (local STT, local brain, macOS voice).
    return ""


def make_offline_factory():
    """Real offline mode (SPEC E8): everything local, nothing leaves the Mac.

    stt   stt7.LocalSTT.transcribe_pcm (faster-whisper en / wav2vec2 pa)
    brain agent7.LocalOpenAICompatProvider against SEVEN_OFFLINE_BRAIN_URL
          (default Qwen3-4B on 127.0.0.1:8097)
    tts   macOS `say` -> ffmpeg -> PCM16 mono 24 kHz for English; b"" for
          Gurmukhi (text-only reply — honest: we have no local pa TTS).
    """
    import shutil

    from agent7 import LocalOpenAICompatProvider
    from stt7 import LocalSTT

    local_stt = LocalSTT()
    brains = list(OFFLINE_BRAINS)
    if os.environ.get("SEVEN_OFFLINE_BRAIN_URL"):
        brains.insert(0, (os.environ["SEVEN_OFFLINE_BRAIN_URL"],
                          os.environ.get("SEVEN_OFFLINE_BRAIN_MODEL", OFFLINE_BRAIN_MODEL)))
    providers = [(url, LocalOpenAICompatProvider(host=url, model=model)) for url, model in brains]

    def stt(pcm: bytes, lang: str) -> str:
        try:
            return local_stt.transcribe_pcm(pcm, lang)
        except Exception:
            log.exception("offline STT failed")
            return "(ਕੁਝ ਸੁਣਾਈ ਨਹੀਂ ਦਿੱਤਾ — I didn't catch that)"

    async def brain(text: str, history: list) -> str:
        prompt = "\n".join(f"{h['role']}: {h['text']}" for h in history[-10:])
        prompt = (prompt + "\n" if prompt else "") + f"user: {text}"
        for url, provider in providers:
            try:
                resp = await provider.complete(OFFLINE_SYSTEM, prompt, max_tokens=300)
                return resp.text.strip()
            except Exception as exc:
                log.warning("offline brain unreachable at %s: %s", url, exc)
        return "My local brain isn't running — I can only listen right now."

    async def tts(text: str) -> bytes:
        has_gurmukhi = any("਀" <= ch <= "੿" for ch in text)
        say, ffmpeg = shutil.which("say"), shutil.which("ffmpeg")
        if has_gurmukhi or not say or not ffmpeg:
            return b""          # Gurmukhi: text-only reply (honest, SPEC E8)
        def _synth() -> bytes:
            # `say -o -` is not supported ("Opening output file failed"), so render to a
            # temp WAV and convert it to 24 kHz PCM16 with ffmpeg.
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                wav = os.path.join(tmp, "say.wav")
                r = subprocess.run([say, "-o", wav, "--data-format=LEI16@22050", text],
                                   capture_output=True, timeout=60)
                if r.returncode != 0 or not os.path.exists(wav):
                    return b""
                proc = subprocess.run(
                    [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", wav,
                     "-ar", "24000", "-ac", "1", "-f", "s16le", "pipe:1"],
                    capture_output=True, timeout=30,
                )
            return proc.stdout if proc.returncode == 0 else b""

        try:
            return await asyncio.to_thread(_synth)
        except Exception:
            log.exception("offline TTS failed")
            return b""

    return offline_session_factory(stt, brain, tts)


async def main() -> None:
    api_key = load_api_key()

    # One Hermes bridge and one approval gate for the whole process. Both
    # announce into whichever live relay / browser is currently connected;
    # with no browser connected the Hermes job file is kept (hermes_result
    # can read it) and approvals simply time out = deny (SPEC §7.2).
    active_relay: list = [None]

    async def approval_event(event: dict) -> None:
        relay = active_relay[0]
        if relay is not None:
            try:
                await relay.send_event(event)
            except Exception:
                log.exception("could not forward approval event %s",
                              event.get("type"))

    async def approval_announce(question: str) -> None:
        relay = active_relay[0]
        if relay is not None:
            try:
                # ask_approval, never say(): the bare question as a user turn
                # made the model call the tool again (approval storm).
                await relay.ask_approval(question)
            except Exception:
                log.exception("could not speak approval question")

    approvals = VoiceApprovals(approval_event, announce=approval_announce,
                               timeout_s=45.0)

    async def on_hermes_done(job: dict) -> None:
        relay = active_relay[0]
        if relay is not None:
            try:
                # M12: timeline first, then the (queued) quiet-moment speech.
                await relay.job_event(job)
                await relay.announce_hermes(job)
            except Exception:
                log.exception("could not announce hermes job %s",
                              job.get("job_id"))

    async def on_hermes_start(job: dict) -> None:
        relay = active_relay[0]
        if relay is not None:
            try:
                await relay.job_event(job)
            except Exception:
                log.exception("could not emit job-start for %s",
                              job.get("job_id"))

    async def on_pause(minutes: int) -> None:
        relay = active_relay[0]
        if relay is not None:
            try:
                await relay.pause(minutes)
            except Exception:
                log.exception("could not send pause for %s minutes", minutes)

    async def on_event(event: dict) -> None:
        # M16: news_show / news_live / news_close -> the page's Work panel.
        relay = active_relay[0]
        if relay is not None:
            try:
                await relay.send_event(event)
            except Exception:
                log.exception("could not forward page event %s",
                              event.get("type"))

    hermes_bridge = HermesBridge(HERMES_JOBS_DIR, on_done=on_hermes_done,
                                 on_start=on_hermes_start)
    registry, _service = build_live_registry(STATE_DIR, hermes_bridge=hermes_bridge,
                                             on_pause=on_pause, on_event=on_event)
    audit = AuditLog(STATE_DIR / "live_audit.jsonl")

    bridge = ToolBridge(
        registry,
        PolicyEngine(),
        LocalSandbox(SecretsBroker()),
        audit,
        ApprovalGate(approvals.callback, timeout_s=60.0),
    )

    if api_key:
        from google import genai

        client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

        def session_factory(voice=None, resume_handle=None):
            # No-arg call for the first session (and test doubles); a voice is
            # passed on reconnect after a set_voice message from the page, and
            # the last resumption handle once the session has issued one (E10).
            config = build_live_config(bridge.declarations(), SYSTEM_INSTRUCTION,
                                       voice=voice, resume_handle=resume_handle)
            return client.aio.live.connect(model=live_model(), config=config)
    else:
        log.warning("GOOGLE_API_KEY not set — 7 Live runs offline only "
                    "(set it in the environment or ~/.hermes/.env for Gemini Live voice)")

        def session_factory(voice=None, resume_handle=None):
            # Fails the online connect at once; the server then uses offline mode.
            raise RuntimeError("GOOGLE_API_KEY is not set")

    def set_active_relay(relay) -> None:
        active_relay[0] = relay

    server = await start_server(session_factory, bridge, port=PORT,
                                approvals=approvals,
                                on_relay=set_active_relay,
                                offline_factory=make_offline_factory())
    url = f"http://127.0.0.1:{server.port}/"
    print(f"7 Live is listening — open {url} (loopback only)", flush=True)
    try:
        await asyncio.Future()  # serve forever
    finally:
        await hermes_bridge.wait_idle()
        await server.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
