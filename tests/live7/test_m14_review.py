"""Claude's M14 review: doc_edit backups must not litter the user's folders."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from live7.office import register_office_tools  # noqa: E402
from sentinel_orchestrator import ToolRegistry  # noqa: E402
from undo7 import UndoStack  # noqa: E402


def test_edit_backups_go_to_seven_dir_not_next_to_file(tmp_path):
    reg, undo = ToolRegistry(), UndoStack()
    register_office_tools(reg, undo=undo, docs_dir=tmp_path / "Documents" / "7", home=tmp_path)
    desk = tmp_path / "Desktop"
    desk.mkdir()
    f = desk / "notes.md"
    f.write_text("one\n")
    for t in ("two", "three"):
        assert asyncio.run(reg.get("doc_edit").fn({"path": str(f), "action": "append", "text": t}))["ok"]
    assert sorted(x.name for x in desk.iterdir()) == ["notes.md"]
    assert len(list((tmp_path / ".seven" / "doc_backups").iterdir())) == 2
    asyncio.run(undo.undo_last())
    asyncio.run(undo.undo_last())
    assert f.read_text() == "one\n"


def test_pdf_mixed_line_keeps_latin_letters(tmp_path):
    import pypdf
    reg = ToolRegistry()
    register_office_tools(reg, docs_dir=tmp_path, home=tmp_path)
    out = asyncio.run(reg.get("doc_create").fn({"kind": "pdf", "name": "mix",
                                                "text": "# 7 Live — ਟੈਸਟ ਰਿਪੋਰਟ\n- ਦਸਤਾਵੇਜ਼ (M14)"}))
    text = pypdf.PdfReader(out["path"]).pages[0].extract_text()
    assert "Live" in text and "M14" in text and "ਟੈਸਟ" in text
