"""Claude's M13 review (2026-10-06): bugs the exam missed, plus a compile check of every AppleScript
the tools generate (osacompile only parses — nothing runs, no app is driven)."""

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from live7.mac import register_mac_tools  # noqa: E402
from sentinel_orchestrator import ToolRegistry  # noqa: E402
from undo7 import UndoStack  # noqa: E402


class Rec:
    def __init__(self):
        self.calls = []

    async def __call__(self, argv, input=None, timeout_s=15):
        self.calls.append(list(argv))
        if "output volume of" in " ".join(argv):
            return 0, "40", ""
        return 0, "x-apple-reminder://ID", ""


def build(home):
    reg, undo, r = ToolRegistry(), UndoStack(), Rec()
    register_mac_tools(reg, undo=undo, runner=r, home=home)
    return reg, undo, r


def call(reg, name, payload):
    return asyncio.run(reg.get(name).fn(payload))


def test_never_trash_or_move_home_or_standard_folders(tmp_path):
    reg, _, _ = build(tmp_path)
    (tmp_path / "Documents").mkdir()
    for p in ("~", str(tmp_path), "~/", "~/Documents", "~/Desktop/.."):
        assert call(reg, "file_trash", {"path": p})["ok"] is False, p
        assert call(reg, "file_op", {"op": "move", "src": p, "dst": "~/elsewhere"})["ok"] is False, p
    assert (tmp_path / "Documents").is_dir()
    (tmp_path / "Documents" / "keep.txt").write_text("k")
    assert call(reg, "file_trash", {"path": "~/Documents/keep.txt"})["ok"]   # files inside are fine


def test_music_app_name_is_quoted_for_applescript(tmp_path):
    reg, _, r = build(tmp_path)
    call(reg, "music_control", {"action": "next", "app": "Spotify"})
    assert 'tell application "Spotify" to next track' in r.calls[-1]


def test_unmute(tmp_path):
    reg, _, r = build(tmp_path)
    assert call(reg, "mac_volume", {"mute": False})["ok"]
    assert "set volume without output muted" in r.calls[-1]


def test_quote_key_refused(tmp_path):
    reg, _, r = build(tmp_path)
    for k in ('cmd+"', "cmd+\\"):
        n = len(r.calls)
        assert call(reg, "key_shortcut", {"keys": k})["ok"] is False and len(r.calls) == n


def scripts_for_every_action(home):
    reg, undo, r = build(home)
    (home / "Desktop").mkdir(exist_ok=True)
    calls = [("mac_info", {"what": "all"}), ("reminders_list", {}), ("mac_volume", {"level": 30}),
             ("mac_volume", {"mute": True}), ("mac_brightness", {"direction": "up", "steps": 2}),
             ("type_text", {"text": "hi"}), ("key_shortcut", {"keys": "cmd+shift+4"}),
             ("key_shortcut", {"keys": "ctrl+f5"}), ("app_quit", {"app": "Safari"}),
             ("reminder_add", {"title": "t", "due": "2026-10-07 09:00", "list": "Work"})]
    calls += [("window_control", {"action": a}) for a in ("minimize", "maximize", "left_half", "right_half", "center")]
    apps = ["Music"] + (["Spotify"] if Path("/Applications/Spotify.app").exists() else [])
    calls += [("music_control", {"action": a, "app": app}) for app in apps
              for a in ("play", "pause", "playpause", "next", "previous", "now_playing")]
    calls += [("music_control", {"action": "play", "app": "Music", "query": "simran"})]
    for name, payload in calls:
        assert call(reg, name, payload)["ok"], (name, payload)
    while undo.pending():
        asyncio.run(undo.undo_last())
    return [argv for argv in r.calls if argv[0] == "osascript"]


@pytest.mark.skipif(not shutil.which("osacompile"), reason="macOS only")
def test_every_generated_applescript_compiles(tmp_path):
    for i, argv in enumerate(scripts_for_every_action(tmp_path)):
        src = [a for j, a in enumerate(argv) if j > 0 and argv[j - 1] == "-e"]
        cmd = ["osacompile", "-o", str(tmp_path / f"s{i}.scpt")]
        for line in src:
            cmd += ["-e", line]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert res.returncode == 0, (src, res.stderr)


def test_real_runner_kills_on_timeout():
    from live7.mac import run_cmd
    import time
    t = time.time()
    rc, _, err = asyncio.run(run_cmd(["/bin/sleep", "30"], timeout_s=0.5))
    assert rc == 124 and "timed out" in err and time.time() - t < 5
    assert subprocess.run(["pgrep", "-f", "/bin/sleep 30"], capture_output=True).returncode != 0


def test_real_runner_refuses_missing_music_app():
    from live7.mac import _app_installed, run_cmd
    reg = ToolRegistry()
    register_mac_tools(reg, runner=run_cmd)
    if not _app_installed("Spotify"):
        out = asyncio.run(reg.get("music_control").fn({"action": "next", "app": "Spotify"}))
        assert out["ok"] is False and "not installed" in out["error"]
