"""7 Live launcher (M15L): the launch moment, from the desktop to a voice.

`python -m live7.launcher`:
  1. ensure_server() — start the 7 Live server if nothing answers on the port;
  2. show_overlay()  — a transparent, click-through layer straight on the
     desktop (no window chrome) playing static/boot.html: the ੭ ignites,
     SEVEN reveals, LIVE ripples, then it fades;
  3. run_app() — 7's own native window (WKWebView, Dock icon "7 Live") is built
     and loaded BEHIND the animation; when the animation hands over, the window
     fades in, takes focus and 7 greets you by voice. No browser.

`python -m live7.launcher --make-app` builds ~/Applications/7 Live.app (with icon)
and installs the login item (7 starts with the Mac; `--uninstall-login` removes it).
`--startup` (used by the login item): the animation plays once per boot.
Settings in ~/.seven/settings.json: startup_animation (true), launch_minimized (false).

Everything degrades: no WebKit -> Chrome app mode (window_command); no display ->
skip the overlay; no Chrome -> `open`.
"""

from __future__ import annotations

import json
import os
import plistlib
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

PORT = 7879
HOME = Path.home()
SEVEN_DIR = HOME / ".seven"
LOG_PATH = SEVEN_DIR / "live7.log"
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CHROME_PROFILE = SEVEN_DIR / "chrome-profile"
REPO = Path(__file__).resolve().parent.parent
APP_DEST = HOME / "Applications" / "7 Live.app"
SETTINGS_PATH = SEVEN_DIR / "settings.json"
LAUNCH_AGENTS = HOME / "Library" / "LaunchAgents"
LOGIN_LABEL = "com.seven.live"
LAUNCH_LOG = SEVEN_DIR / "launcher.log"
class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


_KEEP: list = []       # PyObjC delegates must outlive the call that made them
DEFAULT_SETTINGS = {"startup_animation": True, "launch_minimized": False, "last_boot_stamp": 0}


def _log(msg: str) -> None:
    try:
        SEVEN_DIR.mkdir(parents=True, exist_ok=True)
        with open(LAUNCH_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    out = dict(DEFAULT_SETTINGS)
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            out.update(data)
    except Exception:
        pass
    return out


def save_settings(settings: dict, path: Path = SETTINGS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(settings, indent=2), encoding="utf-8")


def boot_stamp() -> int:
    """When this Mac booted (kern.boottime seconds); 0 if unknown."""
    try:
        out = subprocess.run(["sysctl", "-n", "kern.boottime"], capture_output=True, text=True, timeout=3).stdout
        return int(out.split("sec =")[1].split(",")[0])
    except Exception:
        return 0


def should_play_boot(startup: bool, settings: dict, stamp: int) -> bool:
    """Opening the app always plays the animation (unless switched off); starting
    with the Mac plays it once per boot."""
    if not settings.get("startup_animation", True):
        return False
    if startup:
        return not stamp or int(settings.get("last_boot_stamp") or 0) != stamp
    return True


def media_grant_allowed(host: str, port: int, expected_port: int = PORT) -> bool:
    """The native window grants the mic/camera only to 7's own loopback page."""
    return str(host) in ("127.0.0.1", "localhost") and int(port or 0) == int(expected_port)


def login_item_plist(app: Path = APP_DEST) -> dict:
    return {
        "Label": LOGIN_LABEL,
        "ProgramArguments": ["/usr/bin/open", "-a", str(app), "--args", "--startup"],
        "RunAtLoad": True,
        "ProcessType": "Interactive",
    }


def install_login_item(app: Path = APP_DEST, agents_dir: Path = LAUNCH_AGENTS) -> Path:
    """7 starts with the Mac (like a desktop assistant should). Loaded at next login."""
    agents_dir = Path(agents_dir)
    agents_dir.mkdir(parents=True, exist_ok=True)
    dest = agents_dir / f"{LOGIN_LABEL}.plist"
    dest.write_bytes(plistlib.dumps(login_item_plist(app)))
    return dest


def uninstall_login_item(agents_dir: Path = LAUNCH_AGENTS) -> bool:
    dest = Path(agents_dir) / f"{LOGIN_LABEL}.plist"
    if not dest.exists():
        return False
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(dest)],
                   capture_output=True, timeout=10)
    dest.unlink()
    return True


def _http_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _start_server_detached(port: int) -> None:
    repo = REPO
    SEVEN_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, "ab", buffering=0)
    subprocess.Popen(
        [sys.executable, "-m", "live7"],
        cwd=str(repo),
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, "SEVEN_PORT": str(port)},
    )


def _log_exc(where: str) -> None:
    import traceback
    try:
        SEVEN_DIR.mkdir(parents=True, exist_ok=True)
        with open(SEVEN_DIR / "launcher.log", "a") as f:
            f.write(f"--- {where}\n{traceback.format_exc()}\n")
    except Exception:
        pass


def ensure_server(
    port: int = PORT,
    start: Optional[Callable[[], None]] = None,
    wait_s: float = 20,
    is_up: Optional[Callable[[int], bool]] = None,
) -> bool:
    """True once something serves HTTP 200 on http://127.0.0.1:<port>/.

    Nothing answering -> call start() (default: `python -m live7` detached,
    cwd = this repo, logging to ~/.seven/live7.log) and
    poll every 200 ms until it answers or wait_s runs out.
    """
    up = is_up or _http_up
    if up(port):
        return True
    (start or (lambda: _start_server_detached(port)))()
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if up(port):
            return True
        time.sleep(0.2)
    return up(port)


def window_command(
    url: str,
    chrome: str = CHROME_PATH,
    profile: Path = CHROME_PROFILE,
) -> list:
    """Chrome in app mode; plain `open` when Chrome is not installed."""
    if os.path.exists(chrome):
        return [
            chrome,
            f"--app={url}",
            f"--user-data-dir={profile}",
            "--window-size=1280,820",
            "--autoplay-policy=no-user-gesture-required",
            "--no-first-run",
        ]
    return ["open", url]


def _build_overlay(AppKit, WebKit, url: str, on_finish) -> object:
    """Transparent, mouse-through, full-screen window playing boot.html; calls
    on_finish() once when the page posts "done"."""
    frame = AppKit.NSScreen.mainScreen().frame()
    window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False)
    window.setOpaque_(False)
    window.setBackgroundColor_(AppKit.NSColor.clearColor())
    window.setLevel_(AppKit.NSStatusWindowLevel)
    window.setIgnoresMouseEvents_(True)
    window.setHasShadow_(False)
    window.setReleasedWhenClosed_(False)
    window.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces | AppKit.NSWindowCollectionBehaviorStationary)

    class _BootDone(AppKit.NSObject):  # type: ignore[misc]
        def userContentController_didReceiveScriptMessage_(self, controller, message):
            if str(message.name()) == "boot":
                on_finish()

    config = WebKit.WKWebViewConfiguration.alloc().init()
    handler = _BootDone.alloc().init()
    _KEEP.append(handler)
    config.userContentController().addScriptMessageHandler_name_(handler, "boot")
    try:
        config.setMediaTypesRequiringUserActionForPlayback_(0)   # the launch score plays unclicked
    except Exception:
        pass
    webview = WebKit.WKWebView.alloc().initWithFrame_configuration_(frame, config)
    webview.setValue_forKey_(False, "drawsBackground")
    window.setContentView_(webview)
    window.orderFrontRegardless()
    webview.loadRequest_(AppKit.NSURLRequest.requestWithURL_(AppKit.NSURL.URLWithString_(url)))
    return window


def _build_main_window(AppKit, WebKit, url: str, port: int):
    """7's own app window: dark, titled "7 Live", WKWebView on the 7 page with the
    microphone/camera granted to 7's loopback origin only. Starts invisible (alpha 0)."""
    screen = AppKit.NSScreen.mainScreen().visibleFrame()
    w, h = min(1280, screen.size.width - 80), min(820, screen.size.height - 60)
    rect = ((screen.origin.x + (screen.size.width - w) / 2, screen.origin.y + (screen.size.height - h) / 2), (w, h))
    style = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
             | AppKit.NSWindowStyleMaskMiniaturizable | AppKit.NSWindowStyleMaskResizable)
    win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        rect, style, AppKit.NSBackingStoreBuffered, False)
    win.setTitle_("7 Live")
    win.setTitlebarAppearsTransparent_(True)
    with _suppress():
        win.setAppearance_(AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameDarkAqua))
    win.setReleasedWhenClosed_(False)
    win.setBackgroundColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.03, 0.05, 0.09, 1.0))
    win.setMinSize_((420, 520))
    win.setFrameAutosaveName_("SevenLiveMain")
    win.setAlphaValue_(0.0)

    class _UI(AppKit.NSObject):  # type: ignore[misc]
        def webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
                self, webview, origin, frame, kind, decision):
            ok = media_grant_allowed(origin.host(), origin.port(), port)
            decision(1 if ok else 0)          # WKPermissionDecisionGrant / Deny

        def windowWillClose_(self, note):
            AppKit.NSApp.terminate_(None)

    config = WebKit.WKWebViewConfiguration.alloc().init()
    try:
        config.setMediaTypesRequiringUserActionForPlayback_(0)
    except Exception:
        pass
    webview = WebKit.WKWebView.alloc().initWithFrame_configuration_(((0, 0), (w, h)), config)
    webview.setValue_forKey_(False, "drawsBackground")
    webview.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
    ui = _UI.alloc().init()
    _KEEP.append(ui)
    webview.setUIDelegate_(ui)
    win.setDelegate_(ui)
    win.setContentView_(webview)
    webview.loadRequest_(AppKit.NSURLRequest.requestWithURL_(AppKit.NSURL.URLWithString_(url)))
    return win, webview


def _install_menu(AppKit) -> None:
    """App menu (Quit ⌘Q) and Edit menu so copy/paste work in the composer."""
    bar = AppKit.NSMenu.alloc().init()
    app_item = AppKit.NSMenuItem.alloc().init()
    bar.addItem_(app_item)
    app_menu = AppKit.NSMenu.alloc().initWithTitle_("7 Live")
    app_menu.addItemWithTitle_action_keyEquivalent_("Hide 7 Live", "hide:", "h")
    app_menu.addItem_(AppKit.NSMenuItem.separatorItem())
    app_menu.addItemWithTitle_action_keyEquivalent_("Quit 7 Live", "terminate:", "q")
    app_item.setSubmenu_(app_menu)
    edit_item = AppKit.NSMenuItem.alloc().init()
    bar.addItem_(edit_item)
    edit = AppKit.NSMenu.alloc().initWithTitle_("Edit")
    for title, sel, key in (("Undo", "undo:", "z"), ("Redo", "redo:", "Z"), ("Cut", "cut:", "x"),
                            ("Copy", "copy:", "c"), ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")):
        edit.addItemWithTitle_action_keyEquivalent_(title, sel, key)
    edit_item.setSubmenu_(edit)
    window_item = AppKit.NSMenuItem.alloc().init()
    bar.addItem_(window_item)
    wmenu = AppKit.NSMenu.alloc().initWithTitle_("Window")
    wmenu.addItemWithTitle_action_keyEquivalent_("Minimize", "performMiniaturize:", "m")
    wmenu.addItemWithTitle_action_keyEquivalent_("Close", "performClose:", "w")
    window_item.setSubmenu_(wmenu)
    AppKit.NSApp.setMainMenu_(bar)


def run_app(base: str, play_boot: bool = True, minimized: bool = False, timeout_s: float = 6) -> bool:
    """7 Live as a native app, one event loop for its whole life. The main window is
    created and loaded at t=0 behind the launch animation; when the animation hands
    over it fades in, the app takes focus and 7 starts listening and greets you.
    Returns False (never raises) when WebKit/AppKit are unavailable."""
    try:
        import AppKit
        import WebKit
        from PyObjCTools import AppHelper
    except Exception:
        return False
    if sys.platform != "darwin" or AppKit.NSScreen.mainScreen() is None:
        return False
    # The process is python: name it "7 Live" in the menu bar and Dock, with the ੭ icon.
    with _suppress():
        info = AppKit.NSBundle.mainBundle().infoDictionary()
        info["CFBundleName"] = "7 Live"
        info["CFBundleDisplayName"] = "7 Live"
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)   # Dock icon after the boot
    with _suppress():
        icns = APP_DEST / "Contents" / "Resources" / "7Live.icns"
        if icns.exists():
            app.setApplicationIconImage_(AppKit.NSImage.alloc().initWithContentsOfFile_(str(icns)))
    _install_menu(AppKit)
    win, webview = _build_main_window(AppKit, WebKit, f"{base}/?launch=1&native=1", PORT)
    state = {"shown": False, "overlay": None}

    def reveal(_t=None):
        if state["shown"]:
            return
        state["shown"] = True
        _log("boot handed over; showing the 7 window")
        ov = state["overlay"]
        if ov is not None:
            try:
                ov.animator().setAlphaValue_(0.0)
                AppHelper.callLater(0.4, ov.orderOut_, None)
            except Exception:
                pass
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
        with _suppress():
            icns = APP_DEST / "Contents" / "Resources" / "7Live.icns"
            if icns.exists():
                app.setApplicationIconImage_(AppKit.NSImage.alloc().initWithContentsOfFile_(str(icns)))
        app.activateIgnoringOtherApps_(True)
        win.makeKeyAndOrderFront_(None)
        if minimized:
            win.setAlphaValue_(1.0)
            win.miniaturize_(None)
            return
        AppKit.NSAnimationContext.beginGrouping()
        AppKit.NSAnimationContext.currentContext().setDuration_(0.35)
        win.animator().setAlphaValue_(1.0)
        AppKit.NSAnimationContext.endGrouping()
        start_7()

    tries = {"n": 0}

    def start_7(_t=None):
        # The page may still be loading when the animation hands over: retry until it starts.
        def done(result, error):
            r = str(result) if result is not None else f"error {error}"
            if r != "ok" and tries["n"] < 30:
                AppHelper.callLater(0.5, start_7)
            _log(f"start 7: {r} (try {tries['n']})")
        tries["n"] += 1
        webview.callAsyncJavaScript_arguments_inFrame_inContentWorld_completionHandler_(
            "if (!window.sevenNativeStart) return 'not-ready'; return await window.sevenNativeStart();",
            {}, None, WebKit.WKContentWorld.pageWorld(), done)

    if play_boot:
        try:
            state["overlay"] = _build_overlay(AppKit, WebKit, f"{base}/boot.html", reveal)
            _log("boot animation started")
        except Exception:
            _log_exc("overlay")
        AppHelper.callLater(max(0.5, timeout_s), reveal)
    else:
        AppHelper.callLater(0.6, reveal)       # give the page a moment to load
    AppHelper.runEventLoop(installInterrupt=True)
    return True


def show_overlay(url: str, timeout_s: float = 6, on_done=None) -> bool:
    """Play the boot animation on a transparent, mouse-through overlay.

    Borderless NSWindow over the whole main screen at status-window level,
    opaque off and a clear background so the desktop shows through; a
    WKWebView with drawsBackground off renders boot.html. Closes when the
    page posts "done" to the "boot" script-message handler, or on timeout
    (the window fades out over 250 ms). Returns False (never raises) when
    there is no display or PyObjC is unavailable.
    """
    try:
        import AppKit
        import WebKit
        from PyObjCTools import AppHelper
    except Exception:
        return False
    if sys.platform != "darwin" or not os.environ.get("TERM_PROGRAM") and not AppKit.NSScreen.mainScreen():
        return False

    try:
        # A real NSApplication in accessory mode (no Dock icon) — without it the window never shows.
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        screen = AppKit.NSScreen.mainScreen()
        if screen is None:
            return False
        frame = screen.frame()

        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        window.setOpaque_(False)
        window.setBackgroundColor_(AppKit.NSColor.clearColor())
        window.setLevel_(AppKit.NSStatusWindowLevel)
        window.setIgnoresMouseEvents_(True)
        window.setHasShadow_(False)
        window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
        )

        state = {"closed": False}

        def close_fade(_timer=None):
            if state["closed"]:
                return
            state["closed"] = True
            # PyObjC: stopEventLoop() after runEventLoop() TERMINATES the process, so anything that
            # must happen after the boot (opening the 7 window) runs here, before stopping.
            if on_done is not None:
                try:
                    on_done()
                except Exception:
                    _log_exc("on_done")
            try:
                window.setAlphaValue_(0.0)
                window.animator().setAlphaValue_(0.0)
            except Exception:
                pass
            AppHelper.callLater(0.25, AppHelper.stopEventLoop)

        class BootHandler(AppKit.NSObject):  # type: ignore[misc]
            def userContentController_didReceiveScriptMessage_(self, controller, message):
                if str(message.name()) == "boot":
                    close_fade()

        config = WebKit.WKWebViewConfiguration.alloc().init()
        handler = BootHandler.alloc().init()
        config.userContentController().addScriptMessageHandler_name_(handler, "boot")
        # The launch score is WebAudio started without a click: allow playback.
        try:
            config.setMediaTypesRequiringUserActionForPlayback_(0)   # WKAudiovisualMediaTypeNone
        except Exception:
            pass
        webview = WebKit.WKWebView.alloc().initWithFrame_configuration_(frame, config)
        webview.setValue_forKey_(False, "drawsBackground")

        window.setContentView_(webview)
        window.orderFrontRegardless()
        webview.loadRequest_(
            AppKit.NSURLRequest.requestWithURL_(AppKit.NSURL.URLWithString_(url))
        )
        AppHelper.callLater(max(0.5, timeout_s), close_fade)
        AppHelper.runEventLoop(installInterrupt=True)
        try:
            window.orderOut_(None)
            window.close()
        except Exception:
            pass
        return True
    except Exception:
        _log_exc("show_overlay")
        return False


def make_app(
    dest: Path = APP_DEST,
    python: str = sys.executable,
    repo: Path = REPO,
) -> Path:
    """Build a real 7 Live.app bundle (LSUIElement agent: no Dock icon)."""
    dest = Path(dest)
    macos = dest / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    info = {
        "CFBundleName": "7 Live",
        "CFBundleDisplayName": "7 Live",
        "CFBundleIdentifier": "com.seven.live",
        "CFBundleExecutable": "7Live",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSUIElement": True,
        "LSMinimumSystemVersion": "12.0",
        "CFBundleIconFile": "7Live",
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": "7 listens when you talk to it.",
        "NSCameraUsageDescription": "7 can see through the camera when you turn camera sharing on.",
    }
    (dest / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info))
    exe = macos / "7Live"
    exe.write_text(
        "#!/bin/zsh\n"
        f'cd "{repo}"\n'
        f'exec {python} -m live7.launcher "$@"\n'
    )
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    try:
        make_icon(dest / "Contents" / "Resources" / "7Live.icns")
    except Exception:
        _log_exc("make_icon")
    return dest


def make_icon(dest: Path) -> Path:
    """The ੭ mark glowing on a deep-blue rounded square, as an .icns (AppKit + iconutil)."""
    import tempfile

    import AppKit
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "7Live.iconset"
        iconset.mkdir()
        for size in (16, 32, 64, 128, 256, 512, 1024):
            rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
                None, size, size, 8, 4, True, False, AppKit.NSDeviceRGBColorSpace, 0, 0)
            AppKit.NSGraphicsContext.saveGraphicsState()
            AppKit.NSGraphicsContext.setCurrentContext_(
                AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
            inset = size * 0.1
            box = ((inset, inset), (size - 2 * inset, size - 2 * inset))
            path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(box, size * 0.18, size * 0.18)
            grad = AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(
                AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.06, 0.16, 0.30, 1),
                AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.01, 0.03, 0.08, 1))
            grad.drawInBezierPath_relativeCenterPosition_(path, (0, 0.3))
            shadow = AppKit.NSShadow.alloc().init()
            shadow.setShadowColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.49, 0.83, 0.99, 1))
            shadow.setShadowBlurRadius_(size * 0.08)
            shadow.setShadowOffset_((0, 0))
            font = AppKit.NSFont.systemFontOfSize_weight_(size * 0.62, AppKit.NSFontWeightHeavy)
            attrs = {AppKit.NSFontAttributeName: font,
                     AppKit.NSForegroundColorAttributeName: AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.92, 0.97, 1, 1),
                     AppKit.NSShadowAttributeName: shadow}
            text = AppKit.NSAttributedString.alloc().initWithString_attributes_("੭", attrs)
            ts = text.size()
            text.drawAtPoint_(((size - ts.width) / 2, (size - ts.height) / 2 + size * 0.02))
            AppKit.NSGraphicsContext.restoreGraphicsState()
            png = rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {})
            if size <= 512:
                png.writeToFile_atomically_(str(iconset / f"icon_{size}x{size}.png"), True)
            if size >= 32:
                png.writeToFile_atomically_(str(iconset / f"icon_{size // 2}x{size // 2}@2x.png"), True)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(dest)], check=True,
                       capture_output=True, timeout=30)
    return dest


def main(argv: Optional[list] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--make-app" in args:
        app = make_app()
        print(f"built {app}")
        if "--no-login" not in args:
            print(f"login item {install_login_item(app)} (7 starts with the Mac; --uninstall-login removes it)")
        return 0
    if "--uninstall-login" in args:
        print("login item removed" if uninstall_login_item() else "no login item")
        return 0
    base = f"http://127.0.0.1:{PORT}"
    startup = "--startup" in args
    _log(f"launch (startup={startup})")
    ensure_server(port=PORT)
    settings = load_settings()
    stamp = boot_stamp()
    play = should_play_boot(startup, settings, stamp)
    if startup and stamp:
        settings["last_boot_stamp"] = stamp
        with _suppress():
            save_settings(settings)
    try:
        if run_app(base, play_boot=play, minimized=startup and bool(settings.get("launch_minimized"))):
            return 0
    except Exception:
        _log_exc("run_app")
    # No WebKit: Chrome app mode after the overlay (the M15L path).
    cmd = window_command(f"{base}/?launch=1")
    opened = {"done": False}

    def open_window():
        if opened["done"]:
            return
        opened["done"] = True
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except Exception:
            _log_exc("open window")
            subprocess.Popen(["open", f"{base}/?launch=1"])

    try:
        show_overlay(f"{base}/boot.html", on_done=open_window)
    except Exception:
        _log_exc("overlay")      # the launch must never fail on the animation
    open_window()                # no-op if the overlay already opened it
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
