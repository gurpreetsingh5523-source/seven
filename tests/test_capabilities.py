"""test_capabilities.py — Dots/Muse/desktop capability layer ਦੀਆਂ ਟੈਸਟਾਂ।"""
import asyncio
from pathlib import Path

import pytest

from capabilities import (
    Connector, ConnectorHub, EgressPolicy, build_full_registry,
    desktop_open_app, doc_builder, file_organizer, memory_write,
    plan_file_moves, project_board,
)
from sentinel_orchestrator import Capability, RiskLevel


# -- EgressPolicy (Muse Sentinel pattern) ----------------------------------- #

def test_egress_allows_allowlisted_host():
    e = EgressPolicy(["en.wikipedia.org"])
    assert e.check("https://en.wikipedia.org/wiki/Punjab").startswith("https")


@pytest.mark.parametrize("url", [
    "https://evil.com/x",                 # not in allowlist
    "http://169.254.169.254/latest",      # cloud metadata (SSRF classic)
    "http://127.0.0.1:8080/admin",        # loopback
    "http://192.168.1.1/router",          # private LAN
    "file:///etc/passwd",                 # bad scheme
])
def test_egress_blocks_ssrf_and_unknown(url):
    e = EgressPolicy(["en.wikipedia.org"])
    with pytest.raises(ValueError):
        e.check(url)


# -- Connectors (read/write split) ------------------------------------------- #

def test_connector_read_write_split():
    hub = ConnectorHub()
    hub.add(Connector("calendar", can_read=True, can_write=False))
    assert hub.use("calendar", "read").can_read
    with pytest.raises(PermissionError):
        hub.use("calendar", "write")
    with pytest.raises(KeyError):
        hub.use("gmail", "read")  # unregistered


# -- desktop_open_app (allowlist + dry-run) ----------------------------------- #

def test_desktop_open_app_dry_run_and_allowlist():
    ok = asyncio.run(desktop_open_app({"app": "notepad", "dry_run": True}))
    assert ok["dry_run"] is True and ok["would_open"] == "notepad"
    with pytest.raises(PermissionError):
        asyncio.run(desktop_open_app({"app": "rm -rf /", "dry_run": True}))


# -- file_organizer (dry-run default) ----------------------------------------- #

def test_file_organizer_plans_without_moving(tmp_path: Path):
    (tmp_path / "photo.jpg").write_text("x")
    (tmp_path / "notes.pdf").write_text("x")
    (tmp_path / "script.py").write_text("x")
    res = asyncio.run(file_organizer({"folder": str(tmp_path)}))
    assert res["dry_run"] is True and res["count"] == 3
    dests = {m["to"].split("/")[0] for m in res["planned_moves"]}
    assert dests == {"images", "docs", "code"}
    assert (tmp_path / "photo.jpg").exists()  # nothing moved


def test_plan_file_moves_unknown_goes_misc(tmp_path: Path):
    (tmp_path / "mystery.xyz").write_text("x")
    plan = plan_file_moves(str(tmp_path))
    assert plan[0]["to"].startswith("misc/")


# -- doc_builder (workspace-only, traversal safe) ----------------------------- #

def test_doc_builder_writes_and_blocks_bad_ext(tmp_path: Path):
    res = asyncio.run(doc_builder({"workspace": str(tmp_path),
                                   "filename": "r.md", "content": "hello"}))
    assert Path(res["written"]).exists()
    with pytest.raises(ValueError):
        asyncio.run(doc_builder({"workspace": str(tmp_path),
                                 "filename": "bad.exe", "content": "b"}))


def test_doc_builder_traversal_collapses_to_basename(tmp_path: Path):
    res = asyncio.run(doc_builder({"workspace": str(tmp_path),
                                   "filename": "../../evil.md", "content": "b"}))
    # Path(...).name strips the traversal → lands inside workspace
    assert Path(res["written"]).parent == tmp_path


# -- project_board / memory_write -------------------------------------------- #

def test_project_board_persists(tmp_path: Path):
    r = asyncio.run(project_board(
        {"project": "Launch", "steps": ["a", "b"]}, tmp_path))
    assert r["steps"] == 2
    assert (tmp_path / "project_board.json").exists()
    # existing project keeps its steps even if called again without new ones
    r2 = asyncio.run(project_board({"project": "Launch"}, tmp_path))
    assert r2["steps"] == 2


def test_memory_write_flags_overwrite(tmp_path: Path):
    r1 = asyncio.run(memory_write({"key": "city", "value": "Chandigarh"}, tmp_path))
    assert r1["overwrote_existing"] is False
    r2 = asyncio.run(memory_write({"key": "city", "value": "Ludhiana"}, tmp_path))
    assert r2["overwrote_existing"] is True  # history-tamper signal preserved


# -- build_full_registry wiring ---------------------------------------------- #

def test_full_registry_has_all_capabilities(tmp_path: Path):
    reg = build_full_registry(egress_hosts=["en.wikipedia.org"], state_dir=tmp_path)
    names = set(reg._tools)
    expected = {"web_fetch", "desktop_open_app", "file_organizer",
                "arduino_generate", "doc_builder", "project_board", "memory_write"}
    assert expected <= names
    # risk ladder sanity: external-effect tools must not be LOW
    assert reg._tools["desktop_open_app"].risk == RiskLevel.HIGH
    assert reg._tools["file_organizer"].risk == RiskLevel.MEDIUM
    assert reg._tools["memory_write"].risk == RiskLevel.MEDIUM
    # web_fetch handles untrusted content → capability flagged
    assert Capability.UNTRUSTED_INPUT in reg._tools["web_fetch"].capabilities
