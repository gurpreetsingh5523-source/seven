"""
apps7.py — "7" ਦਾ ਐਪ + ਵੈਬਸਾਈਟ ਲਾਂਚਰ (cross-platform, stdlib only)।

ਐਪ ਖੋਲ੍ਹਣ ਦੀ ਕਾਬਲੀਅਤ — ਸਾਡੇ ਨਿਯਮ:
  * ਵੱਡਾ alias ਨਕਸ਼ਾ (Windows / macOS / Linux ਤਿੰਨਾਂ ਲਈ)
  * ਸਿਰਫ਼ ਜਾਣ-ਪਛਾਣ apps/URLs — PATH ਵਿੱਚ ਮੌਜੂਦ ਹੋਵੇ ਤਾਂ ਹੀ ਚੱਲੇ
  * dry_run default; ਅਸਲੀ launch = HIGH risk → ਮਨੁੱਖੀ ਮਨਜ਼ੂਰੀ
  * ਵੈਬਸਾਈਟ/ਸਰਚ = webbrowser ਨਾਲ default browser ਵਿੱਚ (LOW/MEDIUM)
"""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess
import urllib.parse
import webbrowser
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Alias ਨਕਸ਼ਾ — ਆਮ ਐਪਾਂ ਦੇ ਨਾਮ ਤਿੰਨਾਂ OS ਲਈ                                    #
# --------------------------------------------------------------------------- #

APP_ALIASES: Dict[str, Dict[str, str]] = {
    "whatsapp":    {"Windows": "WhatsApp",   "Darwin": "WhatsApp",          "Linux": "whatsapp-desktop"},
    "chrome":      {"Windows": "chrome",     "Darwin": "Google Chrome",     "Linux": "google-chrome"},
    "firefox":     {"Windows": "firefox",    "Darwin": "Firefox",           "Linux": "firefox"},
    "edge":        {"Windows": "msedge",     "Darwin": "Microsoft Edge",    "Linux": "microsoft-edge"},
    "spotify":     {"Windows": "Spotify",    "Darwin": "Spotify",           "Linux": "spotify"},
    "vscode":      {"Windows": "code",       "Darwin": "Visual Studio Code","Linux": "code"},
    "code":        {"Windows": "code",       "Darwin": "Visual Studio Code","Linux": "code"},
    "discord":     {"Windows": "Discord",    "Darwin": "Discord",           "Linux": "discord"},
    "telegram":    {"Windows": "Telegram",   "Darwin": "Telegram Desktop",  "Linux": "telegram-desktop"},
    "notepad":     {"Windows": "notepad.exe","Darwin": "TextEdit",          "Linux": "gedit"},
    "calculator":  {"Windows": "calc.exe",   "Darwin": "Calculator",        "Linux": "gnome-calculator"},
    "calc":        {"Windows": "calc.exe",   "Darwin": "Calculator",        "Linux": "gnome-calculator"},
    "terminal":    {"Windows": "cmd.exe",    "Darwin": "Terminal",          "Linux": "gnome-terminal"},
    "files":       {"Windows": "explorer.exe","Darwin": "Finder",           "Linux": "nautilus"},
    "explorer":    {"Windows": "explorer.exe","Darwin": "Finder",           "Linux": "nautilus"},
    "word":        {"Windows": "winword",    "Darwin": "Microsoft Word",    "Linux": "libreoffice --writer"},
    "excel":       {"Windows": "excel",      "Darwin": "Microsoft Excel",   "Linux": "libreoffice --calc"},
    "powerpoint":  {"Windows": "powerpnt",   "Darwin": "Microsoft PowerPoint","Linux": "libreoffice --impress"},
    "vlc":         {"Windows": "vlc",        "Darwin": "VLC",               "Linux": "vlc"},
    "paint":       {"Windows": "mspaint",    "Darwin": "Preview",           "Linux": "gimp"},
    "settings":    {"Windows": "ms-settings:","Darwin": "System Settings",  "Linux": "gnome-control-center"},
    "youtube":     {"Windows": "@YOUTUBE",   "Darwin": "@YOUTUBE",          "Linux": "@YOUTUBE"},
    "gmail":       {"Windows": "@GMAIL",     "Darwin": "@GMAIL",            "Linux": "@GMAIL"},
    "maps":        {"Windows": "@MAPS",      "Darwin": "@MAPS",             "Linux": "@MAPS"},
    "google":      {"Windows": "@GOOGLE",    "Darwin": "@GOOGLE",           "Linux": "@GOOGLE"},
    "facebook":    {"Windows": "@FACEBOOK",  "Darwin": "@FACEBOOK",         "Linux": "@FACEBOOK"},
    "instagram":   {"Windows": "@INSTAGRAM", "Darwin": "@INSTAGRAM",        "Linux": "@INSTAGRAM"},
    "github":      {"Windows": "@GITHUB",    "Darwin": "@GITHUB",           "Linux": "@GITHUB"},
    "wikipedia":   {"Windows": "@WIKIPEDIA", "Darwin": "@WIKIPEDIA",        "Linux": "@WIKIPEDIA"},
    "amazon":      {"Windows": "@AMAZON",    "Darwin": "@AMAZON",           "Linux": "@AMAZON"},
}

WEB_SHORTCUTS: Dict[str, str] = {
    "@YOUTUBE": "https://www.youtube.com",
    "@GMAIL":   "https://mail.google.com",
    "@MAPS":    "https://maps.google.com",
    "@GOOGLE":    "https://www.google.com",
    "@FACEBOOK":  "https://www.facebook.com",
    "@INSTAGRAM": "https://www.instagram.com",
    "@GITHUB":    "https://github.com",
    "@WIKIPEDIA": "https://www.wikipedia.org",
    "@AMAZON":    "https://www.amazon.com",
}

SEARCH_ENGINES = {
    "google": "https://www.google.com/search?q={q}",
    "duckduckgo": "https://duckduckgo.com/?q={q}",
    "bing": "https://www.bing.com/search?q={q}",
    "youtube": "https://www.youtube.com/results?search_query={q}",
}


def resolve_app(name: str) -> Optional[str]:
    """ਨਾਮ → OS-ਅਨੁਕੂਲ command। None ਜੇ ਨਾਪਛਾਣਤਾ।"""
    os_name = platform.system()
    key = name.strip().lower()
    entry = APP_ALIASES.get(key)
    if entry is None:
        return None
    return entry.get(os_name) or entry.get("Linux")


def _available(cmd: str) -> bool:
    """PATH ਵਿੱਚ ਹੈ ਜਾਂ macOS app ਹੈ ਜਾਂ Windows URI scheme ਹੈ।"""
    if cmd.startswith("@"):          # web shortcut — ਹਮੇਸ਼ਾ ਚੱਲ ਸਕਦਾ
        return True
    if cmd.startswith("ms-"):        # Windows URI scheme (ms-settings:)
        return True
    first = cmd.split()[0]
    if platform.system() == "Darwin":
        return True                   # `open -a` ਨਾਲ ਚਲਾਵਾਂਗੇ; system ਖੁਦ ਚੈੱਕ ਕਰੇਗਾ
    return shutil.which(first) is not None


async def app_open(payload: Dict[str, Any]) -> Dict[str, Any]:
    """ਐਪ ਖੋਲ੍ਹੋ — dry_run default, ਅਸਲੀ launch ਲਈ approved ਚਾਹੀਦਾ (orchestrator
    ਆਪਣੇ risk-gate ਨਾਲ ਮਨਜ਼ੂਰੀ ਲੈਂਦਾ ਹੈ)।"""
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError("app_open needs 'name'")
    cmd = resolve_app(name)
    if cmd is None:
        return {"error": f"'{name}' ਨਾਪਛਾਣਤਾ — alias ਨਕਸ਼ੇ ਵਿੱਚ ਨਹੀਂ",
                "known": sorted(APP_ALIASES)}
    if cmd.startswith("@"):
        return await site_open({"url": WEB_SHORTCUTS[cmd],
                                "dry_run": payload.get("dry_run", True)})
    if not _available(cmd):
        return {"error": f"'{name}' ({cmd}) ਇਸ ਮਸ਼ੀਨ 'ਤੇ PATH ਵਿੱਚ ਨਹੀਂ ਮਿਲਿਆ",
                "resolved": cmd}
    if payload.get("dry_run", True):
        return {"dry_run": True, "would_open": name, "resolved": cmd}

    os_name = platform.system()
    if os_name == "Darwin":
        proc = await asyncio.create_subprocess_exec("open", "-a", cmd)
    elif os_name == "Windows":
        proc = await asyncio.create_subprocess_shell(cmd)
    else:
        parts = cmd.split()
        proc = await asyncio.create_subprocess_exec(*parts)
    return {"opened": name, "resolved": cmd, "pid": proc.pid}


async def site_open(payload: Dict[str, Any]) -> Dict[str, Any]:
    """ਵੈਬਸਾਈਟ default browser ਵਿੱਚ ਖੋਲ੍ਹੋ।"""
    url = str(payload.get("url", "")).strip()
    if not url:
        raise ValueError("site_open needs 'url'")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if payload.get("dry_run", True):
        return {"dry_run": True, "would_open_url": url}
    await asyncio.to_thread(webbrowser.open, url)
    return {"opened_url": url}


async def web_search_open(payload: Dict[str, Any]) -> Dict[str, Any]:
    """ਸਰਚ ਕਰੋ — engine ਚੁਣੋ ਜਾਂ google default, browser ਵਿੱਚ ਖੁੱਲ੍ਹਦਾ ਹੈ।"""
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ValueError("web_search_open needs 'query'")
    engine = str(payload.get("engine", "google")).lower()
    template = SEARCH_ENGINES.get(engine, SEARCH_ENGINES["google"])
    url = template.format(q=urllib.parse.quote_plus(query))
    if payload.get("dry_run", True):
        return {"dry_run": True, "would_search": query, "url": url}
    await asyncio.to_thread(webbrowser.open, url)
    return {"searched": query, "engine": engine, "url": url}


def known_apps() -> List[str]:
    return sorted(APP_ALIASES)
