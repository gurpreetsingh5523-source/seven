"""What 7 makes comes to the front (2026-10-07). The user: when 7 starts a task, its screen should
open in front like a desktop assistant's does. Before this, doc_create/doc_edit/doc_to_pdf saved the
file silently in Documents/7 and the user had to go and find it."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from live7 import front  # noqa: E402
from live7.office import register_office_tools  # noqa: E402
from sentinel_orchestrator import ToolRegistry  # noqa: E402


def tools(tmp_path, shown):
    reg = ToolRegistry()

    async def show(path):
        shown.append(Path(path).name)
        return True
    register_office_tools(reg, docs_dir=tmp_path / "Documents" / "7", home=tmp_path, show=show)
    return {n: reg.get(n).fn for n in ("doc_create", "doc_edit", "doc_to_pdf")}


def call(fn, payload):
    return asyncio.run(fn(payload))


def test_finished_documents_open_in_front_unless_told_not_to(tmp_path):
    shown = []
    t = tools(tmp_path, shown)
    r = call(t["doc_create"], {"kind": "docx", "name": "plan", "text": "# Plan\n- one"})
    assert r["ok"] and r["shown"] is True and shown == ["plan.docx"]
    r = call(t["doc_edit"], {"path": "plan.docx", "action": "append", "text": "more"})
    assert r["ok"] and shown[-1] == "plan.docx"
    r = call(t["doc_to_pdf"], {"path": "plan.docx"})
    assert r["ok"] and shown[-1].endswith(".pdf")
    n = len(shown)
    r = call(t["doc_create"], {"kind": "docx", "name": "quiet", "text": "x", "show": False})
    assert r["ok"] and "shown" not in r and len(shown) == n


def test_failed_tool_shows_nothing(tmp_path):
    shown = []
    t = tools(tmp_path, shown)
    assert not call(t["doc_create"], {"kind": "exe", "name": "x"})["ok"] and shown == []


def test_show_front_respects_switch_and_missing_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SEVEN_SHOW_RESULTS", "0")
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert asyncio.run(front.show_front(f)) is False
    monkeypatch.setenv("SEVEN_SHOW_RESULTS", "1")
    assert asyncio.run(front.show_front(tmp_path / "missing.txt")) is False


def test_app_registry_wires_show_front():
    src = (ROOT / "live7" / "registry.py").read_text()
    assert "show=show_front" in src
