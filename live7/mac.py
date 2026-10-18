"""M13: Mac control pack for 7 Live.

16 tools so the user can say "volume 30", "copy this", "quit Chrome",
"remind me at 9", "play some music", "pause for 10 minutes" — and 7 does it.

Safe by design:
  * reversible things (MEDIUM, no external effect) run at once and push an
    inverse action onto the undo stack, so the existing "undo" tool reverses
    them; things that touch other apps' input, files, or kill apps sit at
    MEDIUM+EXTERNAL_EFFECT / HIGH, which the policy engine gates behind a
    spoken/clicked YES.
  * run_cmd uses asyncio.create_subprocess_exec — NEVER a shell — so
    argv items are passed verbatim to the program.
  * USER TEXT IS NEVER FORMATTED INTO AN APPLESCRIPT SOURCE STRING. Anything
    the user said (text to type, app names, reminder titles, search words)
    goes to osascript as an argument after an "on run argv" script:
        ["osascript", "-e", "on run argv", "-e", "... item 1 of argv ...",
         "-e", "end run", <user text>]
    Only fixed allow-list words and integers the tool computed itself
    (volume levels, key codes) may appear inside "-e" scripts.
  * Every tool validates first; on bad input it returns
    {"ok": False, "error": ...} WITHOUT running anything, and no tool ever
    raises — every fn catches all exceptions and returns a dict with "ok".
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sentinel_orchestrator import Capability, RiskLevel, ToolRegistry, ToolSpec

Runner = Callable[..., Awaitable[Tuple[int, str, str]]]

CLIP_READ_MAX = 4000
CLIP_WRITE_MAX = 100_000
TYPE_TEXT_MAX = 2000
DUE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})$")
APP_RE = re.compile(r"^[A-Za-z0-9 ._()&+-]{1,64}$")
PROTECTED_APPS = {"finder", "dock", "systemuiserver", "loginwindow", "windowserver"}
WINDOW_ACTIONS = ("minimize", "maximize", "left_half", "right_half", "center")
MUSIC_ACTIONS = ("play", "pause", "playpause", "next", "previous", "now_playing")
MUSIC_APPS = ("Music", "Spotify")
FILE_OPS = ("move", "copy", "rename", "mkdir")

# M19c: desktop_open_app on macOS — really open an installed app with
# `open -a <name>` (never a shell), resolved case-insensitively.
OPEN_APP_DIRS = (
    Path("/Applications"),
    Path("/System/Applications"),
    Path("/System/Applications/Utilities"),
    Path.home() / "Applications",
)


def _resolve_mac_app(name: str) -> Optional[str]:
    """Spoken app name -> the real .app name, or None if it cannot resolve.

    First the Darwin column of apps7.APP_ALIASES ("calculator" -> "Calculator"),
    then an existing "<name>.app" (case-insensitive on the name) in the
    standard app folders. Web-shortcut aliases ("@YOUTUBE") never resolve.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    text = name.strip()
    from apps7 import APP_ALIASES  # local import: apps7 sits at repo root
    entry = APP_ALIASES.get(text.lower())
    if entry is not None:
        darwin = entry.get("Darwin", "")
        if darwin and not darwin.startswith("@"):
            return darwin
    wanted = (text[:-4] if text.lower().endswith(".app") else text).lower()
    for folder in OPEN_APP_DIRS:
        try:
            for child in folder.iterdir():
                if child.suffix == ".app" and child.stem.lower() == wanted:
                    return child.stem
        except OSError:
            continue
    return None


def validate_open_app(payload: Dict[str, Any]) -> None:
    """Raise ValueError when desktop_open_app can never work with this payload."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    name = payload.get("app")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("desktop_open_app needs an app name")
    text = name.strip()
    if text.startswith("-"):
        raise ValueError(f"app name cannot start with '-': {name!r}")
    for bad in "/;&|`$":
        if bad in text:
            raise ValueError(f"illegal character {bad!r} in app name: {name!r}")
    resolved = _resolve_mac_app(text)
    if resolved is None:
        raise ValueError(
            f"no app named {text!r} — try the exact Mac app name "
            f"(e.g. 'Calculator', 'Safari') or a known alias")


async def open_app(payload: Dict[str, Any], runner: Optional[Runner] = None) -> Dict[str, Any]:
    """Really open a Mac app: ["open", "-a", <resolved name>], never a shell."""
    if runner is None:
        runner = run_cmd
    validate_open_app(payload)
    resolved = _resolve_mac_app(str(payload["app"]).strip())
    assert resolved is not None  # validate_open_app already checked
    rc, _, err = await runner(["open", "-a", resolved])
    if rc != 0:
        return _err(err.strip() or f"could not open {resolved}")
    return {"ok": True, "opened": resolved}

_MODS = {
    "cmd": "command down", "command": "command down",
    "shift": "shift down",
    "option": "option down", "alt": "option down",
    "ctrl": "control down", "control": "control down",
}
_NAMED_KEYS = {
    "return": 36, "tab": 48, "space": 49, "escape": 53, "delete": 51,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97,
    "f7": 98, "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}
# quit / force-quit / lock / log-out have their own approved paths — refused here.
# Each tuple is (sorted modifier words..., key); matched against a sorted combo.
REFUSED_SHORTCUTS = frozenset({
    ("command", "q"),
    ("command", "option", "escape"),
    ("command", "control", "q"),
    ("command", "shift", "q"),
})


async def run_cmd(argv: List[str], input: Optional[bytes] = None,
                  timeout_s: float = 15) -> Tuple[int, str, str]:
    """The real runner: asyncio.create_subprocess_exec (NEVER a shell)."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if input is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(input), timeout=timeout_s)
    except asyncio.TimeoutError:
        # e.g. osascript stuck on a "Where is <app>?" dialog — never leave it behind
        proc.kill()
        await proc.wait()
        return 124, "", f"timed out after {timeout_s:g} s: {argv[0]}"
    return (proc.returncode or 0,
            out_b.decode("utf-8", "replace"),
            err_b.decode("utf-8", "replace"))


# --------------------------------------------------------------------------- #
# small helpers (all pure — validation happens before anything runs)
# --------------------------------------------------------------------------- #

def _err(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg}


def _int(value: Any, lo: int, hi: int, default: Optional[int] = None) -> Optional[int]:
    """Best-effort int with clamping; None -> default (or None if no default)."""
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError, OverflowError):
        return default


def _osascript(lines: List[str], args: List[str]) -> List[str]:
    """osascript argv: fixed script lines, user text ONLY as trailing arguments."""
    argv = ["osascript"]
    for line in lines:
        argv += ["-e", line]
    argv += args
    return argv


# Moving or trashing these (or home itself) would wreck the account — refused as a SOURCE.
_PROTECTED_TOP = frozenset({"Desktop", "Documents", "Downloads", "Pictures", "Music", "Movies",
                            "Public", "Applications", "Sites"})


def _app_installed(app: str) -> bool:
    return any((d / f"{app}.app").exists() for d in
               (Path("/Applications"), Path("/System/Applications"), Path.home() / "Applications"))


def _check_src(raw: Any, home: Path) -> Tuple[Optional[Path], Optional[str]]:
    """_check_path, plus: never home itself or a standard top-level folder."""
    path, err = _check_path(raw, home)
    if err:
        return None, err
    rel = path.relative_to(Path(home).resolve())
    if not rel.parts or (len(rel.parts) == 1 and rel.parts[0] in _PROTECTED_TOP):
        return None, f"refusing to move or trash {raw} (home or a standard folder)"
    return path, None


def _check_path(raw: Any, home: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve one path against home; refuse anything outside home, with a
    hidden component, or under home/Library."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "path must be a non-empty string"
    text = raw.strip()
    if text.startswith("~"):
        text = str(home) + text[1:]
    try:
        resolved = Path(text).resolve()
        home_r = Path(home).resolve()
        resolved.relative_to(home_r)
    except (OSError, RuntimeError, ValueError):
        return None, f"path escapes home: {raw}"
    try:
        rel = resolved.relative_to(home_r)
    except ValueError:
        return None, f"path escapes home: {raw}"
    for part in rel.parts:
        if part.startswith("."):
            return None, f"hidden component in path: {raw}"
    if rel.parts and rel.parts[0] == "Library":
        return None, f"path under ~/Library is off limits: {raw}"
    return resolved, None


# --------------------------------------------------------------------------- #
# tool registration
# --------------------------------------------------------------------------- #

def register_mac_tools(registry: ToolRegistry, undo=None, runner: Runner = run_cmd,
                       home: Optional[Path] = None, on_pause=None) -> None:
    home = Path(home) if home is not None else Path.home()

    def push_undo(label: str, undo_fn) -> None:
        if undo is not None:
            undo.push(label, undo_fn)

    async def set_volume(level: int) -> None:
        await runner(_osascript([f"set volume output volume {int(level)}"], []))

    # ---------------------------------------------------------------- LOW #

    async def mac_info(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                payload = {}
            what = str(payload.get("what") or "all").lower()
            if what not in ("volume", "apps", "frontmost", "battery", "all"):
                return _err(f"unknown 'what': {what}")
            out: Dict[str, Any] = {"ok": True}
            if what in ("volume", "all"):
                rc, o, e = await runner(_osascript(
                    ["output volume of (get volume settings)"], []))
                if rc != 0:
                    return _err(e.strip() or "could not read volume")
                try:
                    out["volume"] = int(float(o.strip()))
                except ValueError:
                    out["volume"] = None
            if what in ("apps", "all"):
                rc, o, e = await runner(_osascript([
                    'tell application "System Events" to get name of every '
                    "process whose background only is false"], []))
                if rc != 0:
                    return _err(e.strip() or "could not list apps")
                out["apps"] = [a.strip() for a in o.strip().split(",") if a.strip()]
            if what in ("frontmost", "all"):
                rc, o, e = await runner(_osascript([
                    'tell application "System Events" to get name of first '
                    "process whose frontmost is true"], []))
                if rc != 0:
                    return _err(e.strip() or "could not read frontmost app")
                out["frontmost"] = o.strip()
            if what in ("battery", "all"):
                rc, o, e = await runner(["pmset", "-g", "batt"])
                if rc != 0:
                    return _err(e.strip() or "pmset failed")
                m = re.search(r"(\d+)%", o)
                out["battery"] = int(m.group(1)) if m else None
            return out
        except Exception as exc:
            return _err(str(exc))

    async def assistant_pause(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                payload = {}
            minutes = _int(payload.get("minutes"), 1, 120, default=10)
            if minutes is None:
                return _err("minutes must be a number")
            if on_pause is not None:
                await on_pause(minutes)
            return {"ok": True, "minutes": minutes,
                    "paused_until": time.time() + minutes * 60}
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------- LOW, SENSITIVE_ACCESS #

    async def clipboard_read(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            rc, o, e = await runner(["pbpaste"])
            if rc != 0:
                return _err(e.strip() or "pbpaste failed")
            return {"ok": True, "text": o[:CLIP_READ_MAX],
                    "truncated": len(o) > CLIP_READ_MAX}
        except Exception as exc:
            return _err(str(exc))

    def _rem_list_lines(limit: int) -> List[str]:
        """Script lines printing one line per incomplete reminder: id<TAB>name<TAB>due."""
        return [
            'tell application "Reminders"',
            "set out to {}",
            "set n to 0",
            "repeat with r in (reminders whose completed is false)",
            f"if n < {int(limit)} then",
            'set d to ""',
            "try",
            "set d to (due date of r) as string",
            "end try",
            'set end of out to (id of r) & "\\t" & (name of r) & "\\t" & d',
            "set n to n + 1",
            "end if",
            "end repeat",
            'set AppleScript\'s text item delimiters to "\\n"',
            "return out as string",
            "end tell",
        ]

    async def _open_reminders(limit: int = 50) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
        """List incomplete reminders as {id, name, due} dicts (never raises)."""
        rc, o, e = await runner(_osascript(_rem_list_lines(limit), []))
        if rc != 0:
            return None, (e.strip() or "could not list reminders")
        reminders: List[Dict[str, str]] = []
        for line in o.strip().splitlines():
            if not line.strip():
                continue
            rid, _, rest = line.partition("\t")
            name, _, due = rest.partition("\t")
            reminders.append({"id": rid.strip(), "name": name.strip(), "due": due.strip()})
        return reminders, None

    def _norm_title(text: str) -> str:
        """Case-insensitive, whitespace-collapsed reminder title matching."""
        return " ".join(str(text).split()).lower()

    def _match_reminders(reminders: List[Dict[str, str]], title: str) -> List[Dict[str, str]]:
        """Exact (normalised) name matches win; otherwise names containing the title."""
        wanted = _norm_title(title)
        exact = [r for r in reminders if _norm_title(r["name"]) == wanted]
        if exact:
            return exact
        return [r for r in reminders if wanted and wanted in _norm_title(r["name"])]

    async def reminders_list(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                payload = {}
            limit = _int(payload.get("limit"), 1, 50, default=20) or 20
            reminders, err = await _open_reminders(limit)
            if err:
                return _err(err)
            return {"ok": True, "reminders": reminders}
        except Exception as exc:
            return _err(str(exc))

    def _validate_reminder_title(payload: Dict[str, Any]) -> None:
        """ToolSpec.validate: no approval card for a call that can never match."""
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        title = payload.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")

    async def reminder_delete(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            title = payload.get("title")
            if not isinstance(title, str) or not title.strip():
                return _err("title must be a non-empty string")
            title = title.strip()
            all_matches = bool(payload.get("all", False))
            reminders, err = await _open_reminders()
            if err:
                return _err(err)
            assert reminders is not None
            matches = _match_reminders(reminders, title)
            names = [r["name"] for r in reminders][:10]
            if not matches:
                return {"ok": False, "error": f"no reminder called {title!r}",
                        "names": names}
            if len(matches) > 1 and not all_matches:
                return {"ok": False, "matches": len(matches),
                        "names": [r["name"] for r in matches][:10],
                        "error": f"{len(matches)} reminders match {title!r} — say "
                                 "'all' to delete every one, or give a more exact title"}
            lines = ["on run argv",
                     'tell application "Reminders"',
                     "set n to (count of argv)",
                     "repeat with i from 1 to n",
                     "try",
                     "delete (reminder id (item i of argv))",
                     "end try",
                     "end repeat",
                     "end tell",
                     "end run"]
            rc, _, e = await runner(_osascript(lines, [r["id"] for r in matches]))
            if rc != 0:
                return _err(e.strip() or "could not delete reminder")
            deleted = [{"name": r["name"], "due": r["due"]} for r in matches]

            async def _undo(deleted=deleted) -> Dict[str, Any]:
                # Re-create every deleted reminder; user text (name, due date
                # string) goes only as osascript argv, never in the source.
                for item in deleted:
                    await runner(_osascript(
                        ["on run argv",
                         'tell application "Reminders"',
                         "set r to make new reminder with properties {name:item 1 of argv}",
                         'if item 2 of argv is not "" then',
                         "try",
                         "set due date of r to date (item 2 of argv)",
                         "end try",
                         "end if",
                         "return id of r",
                         "end tell",
                         "end run"],
                        [item["name"], item.get("due") or ""]))
                return {"restored_reminders": [d["name"] for d in deleted]}
            push_undo(f"reminder_delete: restore {len(deleted)} reminder(s)", _undo)
            return {"ok": True, "deleted": deleted}
        except Exception as exc:
            return _err(str(exc))

    async def reminder_complete(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            title = payload.get("title")
            if not isinstance(title, str) or not title.strip():
                return _err("title must be a non-empty string")
            title = title.strip()
            all_matches = bool(payload.get("all", False))
            reminders, err = await _open_reminders()
            if err:
                return _err(err)
            assert reminders is not None
            matches = _match_reminders(reminders, title)
            names = [r["name"] for r in reminders][:10]
            if not matches:
                return {"ok": False, "error": f"no reminder called {title!r}",
                        "names": names}
            if len(matches) > 1 and not all_matches:
                return {"ok": False, "matches": len(matches),
                        "names": [r["name"] for r in matches][:10],
                        "error": f"{len(matches)} reminders match {title!r} — say "
                                 "'all' to complete every one, or give a more exact title"}

            def _set_completed(value: bool, ids: List[str]) -> List[str]:
                word = "true" if value else "false"
                return ["on run argv",
                        'tell application "Reminders"',
                        "set n to (count of argv)",
                        "repeat with i from 1 to n",
                        "try",
                        f"set completed of (reminder id (item i of argv)) to {word}",
                        "end try",
                        "end repeat",
                        "end tell",
                        "end run"]

            ids = [r["id"] for r in matches]
            rc, _, e = await runner(_osascript(_set_completed(True, ids), ids))
            if rc != 0:
                return _err(e.strip() or "could not complete reminder")
            done_names = [r["name"] for r in matches]

            async def _undo(ids=ids, done_names=done_names) -> Dict[str, Any]:
                await runner(_osascript(_set_completed(False, ids), ids))
                return {"reopened_reminders": done_names}
            push_undo(f"reminder_complete: reopen {len(done_names)} reminder(s)", _undo)
            return {"ok": True, "completed": done_names}
        except Exception as exc:
            return _err(str(exc))

    async def files_find(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            query = payload.get("query")
            if not isinstance(query, str) or not query.strip():
                return _err("query must be a non-empty string")
            limit = _int(payload.get("limit"), 1, 50, default=20) or 20
            rc, o, e = await runner(
                ["mdfind", "-onlyin", str(home), "-name", query.strip()])
            if rc != 0:
                return _err(e.strip() or "mdfind failed")
            results = []
            for line in o.splitlines():
                p = line.strip()
                if not p:
                    continue
                try:
                    rel = Path(p).resolve().relative_to(home.resolve())
                except (OSError, ValueError):
                    continue
                parts = rel.parts
                if any(part.startswith(".") for part in parts):
                    continue
                if parts and parts[0] == "Library":
                    continue
                results.append(str(Path(p)))
                if len(results) >= limit:
                    break
            return {"ok": True, "results": results}
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------- MEDIUM, NONE #

    async def mac_volume(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            mute = bool(payload.get("mute", False))
            rc, o, e = await runner(_osascript(
                ["output volume of (get volume settings)"], []))
            if rc != 0:
                return _err(e.strip() or "could not read current volume")
            try:
                old = int(float(o.strip()))
            except ValueError:
                old = 50
            if mute:
                await runner(_osascript(["set volume with output muted"], []))
                push_undo("mac_volume: unmute",
                          lambda: runner(_osascript(
                              ["set volume without output muted"], [])))
                return {"ok": True, "muted": True}
            level = _int(payload.get("level"), 0, 100, default=None)
            if level is None and payload.get("mute") is False:
                await runner(_osascript(["set volume without output muted"], []))
                return {"ok": True, "muted": False}
            if level is None:
                return _err("level must be a number 0..100")
            rc, _, e = await runner(_osascript(
                [f"set volume output volume {level}"], []))
            if rc != 0:
                return _err(e.strip() or "could not set volume")
            async def _undo(old=old) -> Dict[str, Any]:
                await set_volume(old)
                return {"restored_volume": old}
            push_undo(f"mac_volume: back to {old}", _undo)
            return {"ok": True, "level": level, "was": old}
        except Exception as exc:
            return _err(str(exc))

    async def _brightness_press(code: int, steps: int) -> None:
        lines = ['tell application "System Events"']
        lines += [f"key code {code}"] * steps
        lines.append("end tell")
        await runner(_osascript(lines, []))

    async def mac_brightness(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            direction = str(payload.get("direction") or "").lower()
            if direction not in ("up", "down"):
                return _err("direction must be 'up' or 'down'")
            steps = _int(payload.get("steps"), 1, 16, default=2) or 2
            code = 144 if direction == "up" else 145
            await _brightness_press(code, steps)
            other = 145 if code == 144 else 144

            async def _undo(other=other, steps=steps) -> Dict[str, Any]:
                await _brightness_press(other, steps)
                return {"brightness_restored": True}
            push_undo(f"mac_brightness: {'down' if code == 144 else 'up'} x{steps}", _undo)
            return {"ok": True, "direction": direction, "steps": steps}
        except Exception as exc:
            return _err(str(exc))

    async def clipboard_write(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            text = payload.get("text")
            if not isinstance(text, str) or not text:
                return _err("text must be a non-empty string")
            if len(text) > CLIP_WRITE_MAX:
                return _err(f"text too long (max {CLIP_WRITE_MAX} chars)")
            rc, old, e = await runner(["pbpaste"])
            if rc != 0:
                old = ""
            rc, _, e = await runner(["pbcopy"], input=text.encode("utf-8"))
            if rc != 0:
                return _err(e.strip() or "pbcopy failed")

            async def _undo(old=old) -> Dict[str, Any]:
                await runner(["pbcopy"], input=old.encode("utf-8"))
                return {"clipboard_restored": True}
            push_undo("clipboard_write: restore previous clipboard", _undo)
            return {"ok": True, "chars": len(text)}
        except Exception as exc:
            return _err(str(exc))

    async def window_control(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            action = str(payload.get("action") or "").lower()
            if action not in WINDOW_ACTIONS:
                return _err(f"unknown action: {action}")
            lines = ['tell application "System Events"',
                     "set p to first process whose frontmost is true",
                     "set w to front window of p"]
            if action == "minimize":
                lines.append("set value of attribute \"AXMinimized\" of w to true")
            else:
                lines.append('tell application "Finder" to set b to bounds of '
                             "window of desktop")
                lines.append("set screenW to item 3 of b")
                lines.append("set screenH to item 4 of b")
                if action == "maximize":
                    lines.append("set position of w to {0, 25}")
                    lines.append("set size of w to {screenW, screenH - 25}")
                elif action == "left_half":
                    lines.append("set position of w to {0, 25}")
                    lines.append("set size of w to {screenW div 2, screenH - 25}")
                elif action == "right_half":
                    lines.append("set position of w to {screenW div 2, 25}")
                    lines.append("set size of w to {screenW div 2, screenH - 25}")
                else:  # center
                    lines.append("set s to size of w")
                    lines.append("set position of w to "
                                 "{(screenW - (item 1 of s)) div 2, "
                                 "(screenH - (item 2 of s)) div 2}")
            lines.append("end tell")
            rc, _, e = await runner(_osascript(lines, []))
            if rc != 0:
                return _err(e.strip() or f"window_control {action} failed")
            return {"ok": True, "action": action}
        except Exception as exc:
            return _err(str(exc))

    async def music_control(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            action = str(payload.get("action") or "").lower()
            app = str(payload.get("app") or "Music")
            query = payload.get("query")
            if action not in MUSIC_ACTIONS:
                return _err(f"unknown action: {action}")
            if app not in MUSIC_APPS:
                return _err(f"app must be one of {', '.join(MUSIC_APPS)}")
            if query is not None and not isinstance(query, str):
                return _err("query must be a string")
            if runner is run_cmd and not _app_installed(app):
                # osascript would hang on a "Where is <app>?" dialog
                return _err(f"{app} is not installed on this Mac")
            quoted = f'"{app}"'   # allow-listed word; AppleScript needs a quoted app name
            lines: List[str] = []
            args: List[str] = []
            if action == "play" and app == "Music" and query and query.strip():
                lines = ["on run argv",
                         f"tell application {quoted}",
                         "play (first track whose name contains (item 1 of argv))",
                         "end tell",
                         "end run"]
                args = [query.strip()]
            elif action == "play":
                lines = [f"tell application {quoted} to play"]
            elif action == "pause":
                lines = [f"tell application {quoted} to pause"]
            elif action == "playpause":
                lines = [f"tell application {quoted} to playpause"]
            elif action == "next":
                lines = [f"tell application {quoted} to next track"]
            elif action == "previous":
                lines = [f"tell application {quoted} to previous track"]
            else:  # now_playing
                lines = [f"tell application {quoted}",
                         "set t to name of current track",
                         "set a to artist of current track",
                         'return t & "\\t" & a',
                         "end tell"]
            rc, o, e = await runner(_osascript(lines, args))
            if rc != 0:
                return _err(e.strip() or f"music_control {action} failed")
            if action == "now_playing":
                track, _, artist = o.strip().partition("\t")
                return {"ok": True, "track": track.strip(), "artist": artist.strip()}
            return {"ok": True, "action": action, "app": app}
        except Exception as exc:
            return _err(str(exc))

    async def reminder_add(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            title = payload.get("title")
            if not isinstance(title, str) or not title.strip():
                return _err("title must be a non-empty string")
            due = payload.get("due")
            lst = payload.get("list")
            if lst is not None and not isinstance(lst, str):
                return _err("list must be a string")
            lst = (lst or "").strip()
            parts = ["", "", "", "", "", ""]
            if due not in (None, ""):
                if not isinstance(due, str):
                    return _err("due must be a string")
                m = DUE_RE.match(due.strip())
                if not m:
                    return _err("due must match YYYY-MM-DD HH:MM")
                parts = [str(int(g)) for g in m.groups()]
            lines = ["on run argv",
                     'tell application "Reminders"',
                     "set r to make new reminder with properties {name:item 1 of argv}",
                     'if item 2 of argv is not "" then',
                     "set y to (item 2 of argv) as integer",
                     "set mo to (item 3 of argv) as integer",
                     "set d to (item 4 of argv) as integer",
                     "set h to (item 5 of argv) as integer",
                     "set mi to (item 6 of argv) as integer",
                     "set dueDate to current date",
                     "set day of dueDate to 1",
                     "set year of dueDate to y",
                     "set month of dueDate to mo",
                     "set day of dueDate to d",
                     "set hours of dueDate to h",
                     "set minutes of dueDate to mi",
                     "set seconds of dueDate to 0",
                     "set due date of r to dueDate",
                     "end if",
                     'if item 7 of argv is not "" then',
                     "try",
                     "move r to list (item 7 of argv)",
                     "end try",
                     "end if",
                     "return id of r",
                     "end tell",
                     "end run"]
            rc, o, e = await runner(_osascript(
                lines, [title.strip()] + parts + [lst]))
            if rc != 0:
                return _err(e.strip() or "could not create reminder")
            rid = o.strip()

            async def _undo(rid=rid) -> Dict[str, Any]:
                await runner(_osascript(
                    ["on run argv",
                     'tell application "Reminders" to delete (reminder id (item 1 of argv))',
                     "end run"], [rid]))
                return {"deleted_reminder": rid}
            push_undo(f"reminder_add: delete {title.strip()[:30]}", _undo)
            return {"ok": True, "id": rid}
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------- MEDIUM, EXTERNAL_EFFECT #

    async def type_text(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            text = payload.get("text")
            if not isinstance(text, str) or not text:
                return _err("text must be a non-empty string")
            if len(text) > TYPE_TEXT_MAX:
                return _err(f"text too long (max {TYPE_TEXT_MAX} chars)")
            lines = ["on run argv",
                     'tell application "System Events" to keystroke (item 1 of argv)',
                     "end run"]
            rc, _, e = await runner(_osascript(lines, [text]))
            if rc != 0:
                return _err(e.strip() or "type_text failed")
            return {"ok": True, "chars": len(text)}
        except Exception as exc:
            return _err(str(exc))

    async def key_shortcut(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            keys = payload.get("keys")
            if not isinstance(keys, str) or not keys.strip():
                return _err("keys must be a non-empty string like 'cmd+shift+4'")
            tokens = [t.strip().lower() for t in keys.split("+") if t.strip()]
            mods: List[str] = []
            key: Optional[str] = None
            for t in tokens:
                if t in _MODS:
                    if key is not None:
                        return _err(f"modifier after key: {t}")
                    mods.append(_MODS[t])
                elif key is None and ((len(t) == 1 and t not in '"\\') or t in _NAMED_KEYS):
                    key = t
                else:
                    return _err(f"unknown key: {t}")
            if key is None:
                return _err("no key given")
            if tuple(sorted(m.replace(" down", "") for m in mods) + [key]) in REFUSED_SHORTCUTS:
                return _err("that shortcut is refused — quit/lock actions have "
                            "their own approved paths")
            mods_clause = ""
            if mods:
                mods_clause = " using {" + ", ".join(mods) + "}"
            if len(key) == 1:
                press = f'keystroke "{key}"{mods_clause}'
            else:
                press = f"key code {_NAMED_KEYS[key]}{mods_clause}"
            lines = [f'tell application "System Events" to {press}']
            rc, _, e = await runner(_osascript(lines, []))
            if rc != 0:
                return _err(e.strip() or "key_shortcut failed")
            return {"ok": True, "keys": keys}
        except Exception as exc:
            return _err(str(exc))

    async def file_op(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            op = str(payload.get("op") or "").lower()
            if op not in FILE_OPS:
                return _err(f"unknown op: {op}")
            src, err = _check_path(payload.get("src"), home)
            if err:
                return _err(err)
            assert src is not None
            if op == "mkdir":
                src.mkdir(parents=True, exist_ok=False)

                async def _undo_mkdir(src=src) -> Dict[str, Any]:
                    src.rmdir()  # only removes an EMPTY dir
                    return {"removed_dir": str(src)}
                push_undo(f"file_op mkdir: remove {src.name}", _undo_mkdir)
                return {"ok": True, "op": op, "path": str(src)}
            if op != "mkdir":
                src, err = _check_src(payload.get("src"), home)
                if err:
                    return _err(err)
            dst, err = _check_path(payload.get("dst"), home)
            if err:
                return _err(err)
            assert dst is not None
            if op == "rename":
                dst = src.parent / dst.name
            if not src.exists():
                return _err(f"no such file: {src}")
            if dst.exists():
                return _err(f"destination already exists: {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if op in ("move", "rename"):
                shutil.move(str(src), str(dst))

                async def _undo_move(src=src, dst=dst) -> Dict[str, Any]:
                    shutil.move(str(dst), str(src))
                    return {"restored": str(src)}
                push_undo(f"file_op {op}: {dst.name} back to {src.name}", _undo_move)
            else:  # copy
                if src.is_dir():
                    shutil.copytree(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))

                async def _undo_copy(dst=dst) -> Dict[str, Any]:
                    if dst.is_dir():
                        shutil.rmtree(str(dst))
                    elif dst.exists():
                        dst.unlink()
                    return {"removed_copy": str(dst)}
                push_undo(f"file_op copy: remove {dst.name}", _undo_copy)
            return {"ok": True, "op": op, "src": str(src), "dst": str(dst)}
        except Exception as exc:
            return _err(str(exc))

    # ---------------------------------------------- HIGH, EXTERNAL_EFFECT #

    async def app_quit(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            app = payload.get("app")
            if not isinstance(app, str) or not APP_RE.match(app):
                return _err("app name must match [A-Za-z0-9 ._()&+-]{1,64}")
            if app.strip().lower() in PROTECTED_APPS:
                return _err(f"refusing to quit system process: {app}")
            force = bool(payload.get("force", False))
            if force:
                rc, _, e = await runner(["pkill", "-9", "-x", app])
            else:
                rc, _, e = await runner(_osascript(
                    ["on run argv",
                     "tell application (item 1 of argv) to quit",
                     "end run"], [app]))
            if rc != 0:
                return _err(e.strip() or f"could not quit {app}")
            return {"ok": True, "app": app, "force": force}
        except Exception as exc:
            return _err(str(exc))

    async def file_trash(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            src, err = _check_src(payload.get("path"), home)
            if err:
                return _err(err)
            assert src is not None
            if not src.exists():
                return _err(f"no such file: {src}")
            trash = home / ".Trash"
            trash.mkdir(exist_ok=True)
            dst = trash / src.name
            n = 1
            while dst.exists():
                dst = trash / f"{src.stem} {n}{src.suffix}"
                n += 1
            shutil.move(str(src), str(dst))

            async def _undo(src=src, dst=dst) -> Dict[str, Any]:
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst), str(src))
                return {"restored": str(src)}
            push_undo(f"file_trash: restore {src.name}", _undo)
            return {"ok": True, "trashed": str(dst)}
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------ register #
    L, M, H = RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH
    NONE = Capability.NONE
    SENS = Capability.SENSITIVE_ACCESS
    EXT = Capability.EXTERNAL_EFFECT
    for spec in (
        ToolSpec("mac_info", mac_info, L, NONE),
        ToolSpec("assistant_pause", assistant_pause, L, NONE),
        ToolSpec("clipboard_read", clipboard_read, L, SENS),
        ToolSpec("reminders_list", reminders_list, L, SENS),
        ToolSpec("files_find", files_find, L, SENS),
        ToolSpec("mac_volume", mac_volume, M, NONE),
        ToolSpec("mac_brightness", mac_brightness, M, NONE),
        ToolSpec("clipboard_write", clipboard_write, M, NONE),
        ToolSpec("window_control", window_control, M, NONE),
        ToolSpec("music_control", music_control, M, NONE),
        ToolSpec("reminder_add", reminder_add, M, NONE),
        ToolSpec("reminder_delete", reminder_delete, M, EXT,
                 validate=_validate_reminder_title),
        ToolSpec("reminder_complete", reminder_complete, M, NONE,
                 validate=_validate_reminder_title),
        ToolSpec("type_text", type_text, M, EXT),
        ToolSpec("key_shortcut", key_shortcut, M, EXT),
        ToolSpec("file_op", file_op, M, EXT),
        ToolSpec("app_quit", app_quit, H, EXT),
        ToolSpec("file_trash", file_trash, H, EXT),
    ):
        registry.register(spec)
