"""7 Live as its own desktop app (2026-10-07). Seen on the real desktop: the launch opened
7 in a Chrome window that stayed grey ~2 s after the animation, and macOS warned that
"python3.12 was prevented from modifying apps". Like a real desktop assistant, 7 must open
in its OWN window right as the entry animation hands over, start listening and greet you,
and start with the Mac (animation once per boot)."""

import plistlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from live7 import launcher  # noqa: E402


def test_bundle_declares_mic_camera_and_icon(tmp_path):
    app = launcher.make_app(dest=tmp_path / "7 Live.app", python="/py", repo=tmp_path)
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    # Without these macOS kills an app the first time it touches the microphone.
    assert info["NSMicrophoneUsageDescription"] and info["NSCameraUsageDescription"]
    assert info["CFBundleIconFile"] == "7Live"
    icns = app / "Contents" / "Resources" / "7Live.icns"
    assert icns.exists() and icns.stat().st_size > 10_000


def test_media_granted_only_to_7s_own_page():
    assert launcher.media_grant_allowed("127.0.0.1", 7879)
    assert not launcher.media_grant_allowed("127.0.0.1", 8000)
    assert not launcher.media_grant_allowed("evil.example", 7879)


def test_boot_animation_once_per_boot_when_starting_with_the_mac():
    s = dict(launcher.DEFAULT_SETTINGS)
    assert launcher.should_play_boot(False, s, 100)              # opening the app: always
    assert launcher.should_play_boot(True, s, 100)               # first start this boot
    s["last_boot_stamp"] = 100
    assert not launcher.should_play_boot(True, s, 100)           # already played this boot
    assert launcher.should_play_boot(True, s, 200)               # new boot
    s["startup_animation"] = False
    assert not launcher.should_play_boot(False, s, 200)


def test_settings_roundtrip(tmp_path):
    p = tmp_path / "settings.json"
    assert launcher.load_settings(p)["startup_animation"] is True
    launcher.save_settings({"launch_minimized": True}, p)
    got = launcher.load_settings(p)
    assert got["launch_minimized"] is True and got["startup_animation"] is True


def test_login_item_install_and_remove(tmp_path):
    dest = launcher.install_login_item(Path("/Apps/7 Live.app"), agents_dir=tmp_path)
    info = plistlib.loads(dest.read_bytes())
    assert info["Label"] == "com.seven.live" and info["RunAtLoad"] is True
    assert info["ProgramArguments"][-1] == "--startup" and "/Apps/7 Live.app" in info["ProgramArguments"]
    assert launcher.uninstall_login_item(agents_dir=tmp_path) and not dest.exists()


def test_main_prefers_native_window_over_chrome(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(launcher, "SETTINGS_PATH", tmp_path / "s.json")
    monkeypatch.setattr(launcher, "ensure_server", lambda **kw: True)
    monkeypatch.setattr(launcher, "boot_stamp", lambda: 0)
    monkeypatch.setattr(launcher, "run_app", lambda base, **kw: calls.append(("native", kw)) or True)
    monkeypatch.setattr(launcher, "show_overlay", lambda *a, **k: calls.append("chrome-overlay"))
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: calls.append("popen"))
    assert launcher.main([]) == 0
    assert calls == [("native", {"play_boot": True, "minimized": False})], calls


def test_page_has_native_start_hook():
    html = (ROOT / "static" / "live.html").read_text(encoding="utf-8")
    assert "window.sevenNativeStart" in html and 'get("native")' in html
