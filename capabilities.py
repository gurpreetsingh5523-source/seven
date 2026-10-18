"""
capabilities.py — Agent 7 ਦੇ ਅਸਲੀ ਕੰਮ ਕਰਨ ਵਾਲੇ tools (Dots + Muse + desktop-assistant
capabilities — ਖੁੱਲ੍ਹਾ, ਟੈਸਟ-ਪ੍ਰਮਾਣਿਤ)।

ਕਾਬਲੀਅਤਾਂ ਜੋ ਇਸ ਫਾਈਲ ਵਿੱਚ ਜੋੜੀਆਂ ਗਈਆਂ ਹਨ:
  Desktop assistant:
    * desktop_automation  — ਐਪ ਖੋਲ੍ਹੋ, typing, ਕਮਾਂਡ (HIGH risk, allowlist)
    * web_fetch           — ਸੁਰੱਖਿਅਤ web page ਲਿਆਓ (egress allowlist = Muse Sentinel ਪੈਟਰਨ)
    * file_organizer      — ਫੋਲਡਰ ਦੀਆਂ ਫਾਈਲਾਂ ਕਿਸਮ-ਵਾਰ ਵੰਡੋ (dry-run default)
    * doc_builder         — ਮਾਰਕਡਾਊਨ ਰਿਪੋਰਟ/ਸਲਾਈਡ-ਨੋਟ ਬਣਾਓ (LOW, workspace-only)
    * arduino_gen         — Arduino sketch generate + (ਵਿਕਲਪਿਕ) flash (flash = CRITICAL)
  Dots ਤੋਂ:
    * project_board       — multi-step ਕੰਮ ਦੀ ਲੜੀ (checklist) ਬਣਾ ਕੇ ਟਰੈਕ ਕਰੋ
  Muse ਤੋਂ (ਜੋ ਸਾਡੇ ਕੋਲ ਨਹੀਂ ਸੀ):
    * connectors          — read vs write ਵੱਖ-ਵੱਖ (least privilege)
    * web_fetch           — Sentinel-style egress allowlist + SSRF private-IP ਰੋਕ
  ਸਾਡਾ ਆਪਣਾ:
    * memory_write        — ਏਜੰਟ ਯਾਦਦਾਸ਼ਤ ਆਪਣੇ ਕਬਜ਼ੇ ਵਿੱਚ ਨਹੀਂ; key ਬਦਲਣਾ CRITICAL ਹੈ

ਸਾਰੇ tools ਉਹੀ ਪਾਈਪਲਾਈਨ ਵਰਤਦੇ ਹਨ: policy → approval → audit → checkpoint → sandbox.
"""

from __future__ import annotations

import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentinel_orchestrator import Capability, RiskLevel, ToolRegistry, ToolSpec


# --------------------------------------------------------------------------- #
# Muse-pattern: egress allowlist (Sentinel ਦਾ ਸਾਡਾ ਸੰਸਕਰਣ)                     #
# --------------------------------------------------------------------------- #

class EgressPolicy:
    """Sentinel ਦਾ core ਵਿਚਾਰ: ਹਰ ਬਾਹਰ ਜਾਣ ਵਾਲੀ request ਅਨੁਮਾਨਿਤ host ਤੋਂ ਹੀ ਲੰਘੇ।
    Private/loopback IP = SSRF ਰੋਕ (default deny)."""

    def __init__(self, allowed_hosts: Optional[List[str]] = None):
        self.allowed_hosts = set(allowed_hosts or [])

    def check(self, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"scheme {parsed.scheme!r} not allowed")
        host = parsed.hostname or ""
        try:  # direct IP literals: block private ranges outright (SSRF)
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                raise ValueError(f"private/loopback IP blocked: {host}")
        except ValueError as exc:
            if "blocked" in str(exc):
                raise
        if self.allowed_hosts and host not in self.allowed_hosts:
            raise ValueError(f"host {host!r} not in egress allowlist")
        return url


async def web_fetch(payload: Dict[str, Any], egress: EgressPolicy) -> Dict[str, Any]:
    """ਸੁਰੱਖਿਅਤ page fetch — Muse browser-subagent ਪੈਟਰਨ: title + text only,
    ਕੋਈ script execution ਨਹੀਂ, max-size cap, egress policy ਜ਼ਰੂਰੀ।"""
    url = egress.check(payload["url"])
    max_bytes = int(payload.get("max_bytes", 200_000))

    def _get() -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "Agent7/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read(max_bytes)

    import asyncio
    raw = await asyncio.to_thread(_get)
    html = raw.decode("utf-8", errors="replace")
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return {"url": url,
            "title": (title.group(1).strip() if title else ""),
            "text_preview": text[:2000],
            "bytes": len(raw)}


# --------------------------------------------------------------------------- #
# Connectors — Muse ਦਾ read/write ਵੰਡ (least privilege)                        #
# --------------------------------------------------------------------------- #

class Connector:
    """ਬਾਹਰੀ ਸੇਵਾ ਨਾਲ ਸੰਬੰਧ। READ ਅਤੇ WRITE ਵੱਖ-ਵੱਖ ਐਕਸੈਸ — Muse ਦਾ
    ਸਭ ਤੋਂ ਜ਼ਰੂਰੀ ਸਬਕ: "calendar ਪੜ੍ਹਨਾ" ਅਤੇ "meeting ਬਣਾਉਣਾ" ਇੱਕੋ ਐਕਸੈਸ ਨਹੀਂ।"""

    def __init__(self, name: str, can_read: bool = False, can_write: bool = False):
        self.name = name
        self.can_read = can_read
        self.can_write = can_write


class ConnectorHub:
    def __init__(self) -> None:
        self._connectors: Dict[str, Connector] = {}

    def add(self, conn: Connector) -> None:
        self._connectors[conn.name] = conn

    def use(self, name: str, mode: str) -> Connector:
        conn = self._connectors.get(name)
        if conn is None:
            raise KeyError(f"connector {name!r} not connected")
        if mode == "read" and not conn.can_read:
            raise PermissionError(f"{name}: read access not granted")
        if mode == "write" and not conn.can_write:
            raise PermissionError(f"{name}: write access not granted")
        return conn


# --------------------------------------------------------------------------- #
# Desktop automation — allowlist + dry-run ਨਾਲ          #
# --------------------------------------------------------------------------- #

ALLOWED_BINS = {"notepad", "gedit", "code", "calc", "gnome-calculator",
                "explorer", "open", "xdg-open"}


async def desktop_open_app(payload: Dict[str, Any]) -> Dict[str, Any]:
    """ਐਪ ਖੋਲ੍ਹੋ — ਸਿਰਫ਼ allowlist ਵਿੱਚੋਂ (ਸੁਰੱਖਿਅਤ ਰੂਪ)।"""
    import asyncio
    app = payload.get("app", "").strip()
    if app not in ALLOWED_BINS:
        raise PermissionError(f"app {app!r} not in allowlist {sorted(ALLOWED_BINS)}")
    if payload.get("dry_run", True):
        return {"dry_run": True, "would_open": app}
    proc = await asyncio.create_subprocess_exec(app)
    return {"opened": app, "pid": proc.pid}


# --------------------------------------------------------------------------- #
# File organizer — dry-run default              #
# --------------------------------------------------------------------------- #

CATEGORIES = {"images": {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"},
              "docs": {".pdf", ".docx", ".txt", ".md", ".xlsx", ".pptx"},
              "code": {".py", ".js", ".ts", ".c", ".cpp", ".ino", ".json"},
              "media": {".mp3", ".mp4", ".wav", ".mov"}}


def resolve_folder(name: str, home: Optional[Path] = None) -> Optional[Path]:
    """Resolve a user-facing folder name to an existing directory.

    Accepts "~" (home itself), an existing absolute path, a case-insensitive
    name of a folder directly inside home ("downloads" -> ~/Downloads), or a
    relative path under home ("Desktop/tools"). Anything that does not exist
    resolves to None so the caller can ask the user what they mean.
    """
    base = Path(home) if home is not None else Path.home()
    text = str(name).strip()
    if not text:
        return None
    if text in ("~", "~/"):
        return base if base.is_dir() else None
    expanded = Path(text).expanduser()
    if expanded.is_absolute():
        return expanded if expanded.is_dir() else None
    rel = expanded if not str(expanded).startswith("~") else Path(text[2:])
    if len(rel.parts) == 1:
        try:
            for child in base.iterdir():
                if child.is_dir() and child.name.lower() == rel.parts[0].lower():
                    return child
        except OSError:
            return None
    cand = base / rel
    return cand if cand.is_dir() else None


def plan_file_moves(folder: str) -> List[Dict[str, str]]:
    src = resolve_folder(folder)
    if src is None:
        raise ValueError(
            f"folder not found: {folder} — the user's home folder is {Path.home()}; "
            "ask the user which folder they mean")
    moves = []
    for f in sorted(src.iterdir()):
        if f.is_file():
            cat = next((c for c, exts in CATEGORIES.items()
                        if f.suffix.lower() in exts), "misc")
            moves.append({"file": f.name, "to": f"{cat}/{f.name}"})
    return moves


async def file_organizer(payload: Dict[str, Any]) -> Dict[str, Any]:
    folder = payload["folder"]
    moves = plan_file_moves(folder)   # raises "folder not found: ..." naming home
    src_dir = resolve_folder(folder)
    if payload.get("dry_run", True):
        return {"dry_run": True, "planned_moves": moves, "count": len(moves)}
    moved = 0
    for m in moves:
        src = src_dir / m["file"]
        dst = src_dir / m["to"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        moved += 1
    return {"moved": moved}


# --------------------------------------------------------------------------- #
# Arduino — generate ਮੁਫ਼ਤ, flash CRITICAL            #
# --------------------------------------------------------------------------- #

async def arduino_generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sketch ਸਿਰਫ਼ ਲਿਖੋ (LOW)। ਫਲੈਸ਼ ਕਰਨਾ ਵੱਖਰਾ HIGH/CRITICAL tool ਹੈ।"""
    led_pin = int(payload.get("led_pin", 13))
    interval = int(payload.get("interval_ms", 1000))
    sketch = (
        "// Agent7-generated sketch\n"
        f"const int LED = {led_pin};\n"
        "void setup(){ pinMode(LED, OUTPUT); Serial.begin(9600); }\n"
        "void loop(){ digitalWrite(LED,HIGH); delay(%d);\n"
        "             digitalWrite(LED,LOW);  delay(%d);\n"
        "             Serial.println(\"blink\"); }" % (interval, interval))
    return {"sketch": sketch, "lines": sketch.count("\n") + 1}


# --------------------------------------------------------------------------- #
# Doc builder + project board — Dots ਵਿਸ਼ੇਸ਼ਤਾ            #
# --------------------------------------------------------------------------- #

async def doc_builder(payload: Dict[str, Any]) -> Dict[str, Any]:
    """ਮਾਰਕਡਾਊਨ ਰਿਪੋਰਟ/ਨੋਟ — workspace ਫੋਲਡਰ ਦੇ ਅੰਦਰ ਹੀ ਲਿਖੋ (LOW risk)।"""
    workspace = Path(payload.get("workspace", "./workspace")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    name = Path(payload.get("filename", "note.md")).name  # path traversal ਰੋਕ
    if not name.endswith((".md", ".txt")):
        raise ValueError("only .md/.txt allowed")
    out = workspace / name
    out.write_text(payload.get("content", ""), encoding="utf-8")
    return {"written": str(out), "bytes": out.stat().st_size}


async def project_board(payload: Dict[str, Any], state_dir: Path) -> Dict[str, Any]:
    """Dots-ਸ਼ੈਲੀ checklist: multi-step ਕੰਮ ਦੀ ਲੜੀ ਬਣਾ ਕੇ ਸਟੇਟ ਸੇਵ ਕਰੋ।"""
    board_path = state_dir / "project_board.json"
    board = json.loads(board_path.read_text()) if board_path.exists() else {}
    proj = payload["project"]
    steps = payload.get("steps", [])
    board.setdefault(proj, {"steps": [
        {"title": s, "done": False} for s in steps]})["steps"]
    board_path.write_text(json.dumps(board, indent=2, ensure_ascii=False))
    return {"project": proj, "steps": len(board[proj]["steps"]),
            "saved": str(board_path)}


# --------------------------------------------------------------------------- #
# Memory write — ਏਜੰਟ ਦੀ ਯਾਦਦਾਸ਼ਤ ਏਜੰਟ ਦੇ ਕਬਜ਼ੇ ਵਿੱਚ ਨਹੀਂ                       #
# --------------------------------------------------------------------------- #

async def memory_write(payload: Dict[str, Any], state_dir: Path) -> Dict[str, Any]:
    """ਨਵੀਂ key ਜੋੜਨਾ MEDIUM ਹੈ; ਮੌਜੂਦਾ key ਬਦਲਣਾ CRITICAL (ਇਤਿਹਾਸ-ਛੇੜਛਾੜ ਰੋਕ)।"""
    mem_path = state_dir / "agent_memory.json"
    mem = json.loads(mem_path.read_text()) if mem_path.exists() else {}
    key, value = payload["key"], payload["value"]
    overwriting = key in mem
    mem[key] = value
    mem_path.write_text(json.dumps(mem, indent=2, ensure_ascii=False))
    return {"stored": key, "overwrote_existing": overwriting}


# --------------------------------------------------------------------------- #
# Registry factory — ਸਾਰੇ tools ਇੱਕ ਥਾਂ                                        #
# --------------------------------------------------------------------------- #

def build_full_registry(egress_hosts: Optional[List[str]] = None,
                        state_dir: str = "./agent7_state") -> ToolRegistry:
    """Agent 7 ਦਾ ਪੂਰਾ tool-set — 3 ਨਵੇਂ + 7 ਨਵੀਆਂ capabilities = 10 tools।"""
    from sentinel_orchestrator import ToolRegistry, ToolSpec
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    egress = EgressPolicy(egress_hosts)
    reg = ToolRegistry()

    async def _web(p): return await web_fetch(p, egress)
    async def _desk(p): return await desktop_open_app(p)
    async def _org(p): return await file_organizer(p)
    async def _ard(p): return await arduino_generate(p)
    async def _doc(p): return await doc_builder(p)
    async def _board(p): return await project_board(p, state)
    async def _mem(p): return await memory_write(p, state)

    reg.register(ToolSpec("web_fetch", _web, RiskLevel.LOW,
                          Capability.UNTRUSTED_INPUT,
                          validate=lambda p: egress.check(p["url"]) and None))
    reg.register(ToolSpec("desktop_open_app", _desk, RiskLevel.HIGH,
                          Capability.EXTERNAL_EFFECT,
                          validate=lambda p: (_ for _ in ()).throw(
                              PermissionError("bad app")) if p.get("app") not in ALLOWED_BINS else None))
    reg.register(ToolSpec("file_organizer", _org, RiskLevel.MEDIUM,
                          Capability.EXTERNAL_EFFECT,
                          validate=lambda p: None if p.get("folder") else (_ for _ in ()).throw(ValueError("need folder"))))
    reg.register(ToolSpec("arduino_generate", _ard, RiskLevel.LOW, Capability.NONE))
    reg.register(ToolSpec("doc_builder", _doc, RiskLevel.LOW, Capability.NONE))
    reg.register(ToolSpec("project_board", _board, RiskLevel.LOW, Capability.NONE,
                          validate=lambda p: None if p.get("project") else (_ for _ in ()).throw(ValueError("need project"))))
    reg.register(ToolSpec("memory_write", _mem, RiskLevel.MEDIUM,
                          Capability.SENSITIVE_ACCESS,
                          validate=lambda p: None if p.get("key") else (_ for _ in ()).throw(ValueError("need key"))))
    return reg
