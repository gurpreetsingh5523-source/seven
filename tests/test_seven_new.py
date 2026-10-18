"""test_seven_new.py — apps7, browser7, skills7, undo7 ਦੀਆਂ ਟੈਸਟਾਂ।"""
import asyncio
from pathlib import Path

import pytest

from apps7 import app_open, site_open, web_search_open, resolve_app, known_apps
from browser7 import smart_search, playwright_available
from capabilities import EgressPolicy
from skills7 import (system_monitor, globe_distance, circuit_schematic,
                     haversine_km, CITIES, SCHEMATICS)
from undo7 import UndoStack, make_move_undo, make_write_undo

EG = EgressPolicy(["duckduckgo.com", "wttr.in"])


# -- apps7 ------------------------------------------------------------------- #

def test_resolve_app_alias_table():
    assert resolve_app("chrome") is not None
    assert resolve_app("Notepad") is not None
    assert resolve_app("rm -rf") is None
    assert "spotify" in known_apps()


def test_app_open_dry_run_and_unknown():
    r = asyncio.run(app_open({"name": "chrome", "dry_run": True}))
    assert r.get("dry_run") is True or "error" in r  # PATH 'ਤੇ ਨਿਰਭਰ
    bad = asyncio.run(app_open({"name": "totally_fake_app", "dry_run": True}))
    assert "error" in bad and "known" in bad


def test_app_open_web_shortcut():
    r = asyncio.run(app_open({"name": "youtube", "dry_run": True}))
    assert r.get("dry_run") is True
    assert "youtube.com" in r.get("would_open_url", "")


def test_site_open_adds_https_and_dry_run():
    r = asyncio.run(site_open({"url": "github.com", "dry_run": True}))
    assert r["would_open_url"] == "https://github.com"
    with pytest.raises(ValueError):
        asyncio.run(site_open({"url": ""}))


def test_web_search_open_builds_engine_url():
    r = asyncio.run(web_search_open({"query": "ਪੰਜਾਬ ਖ਼ਬਰਾਂ", "dry_run": True}))
    assert "google.com/search" in r["url"]
    r2 = asyncio.run(web_search_open({"query": "test", "engine": "youtube",
                                      "dry_run": True}))
    assert "youtube.com/results" in r2["url"]


# -- browser7 ----------------------------------------------------------------- #

def test_smart_search_no_playwright_is_honest():
    if playwright_available():
        pytest.skip("playwright installed — live path")
    r = asyncio.run(smart_search({"query": "test"}, EG))
    assert r["mode"] == "no_playwright"
    assert "playwright" in r["note"]


def test_smart_search_respects_egress():
    bad = EgressPolicy([])  # empty allowlist + non-empty default? empty = allow-all-by-design
    # empty allowed_hosts → check() lets any non-private host; use strict list instead
    strict = EgressPolicy(["wttr.in"])
    with pytest.raises(ValueError):
        asyncio.run(smart_search({"query": "x"}, strict))  # duckduckgo not allowed


# -- skills7 ------------------------------------------------------------------ #

def test_system_monitor_returns_real_numbers():
    r = asyncio.run(system_monitor({}))
    if "error" not in r:
        assert 0 <= r["cpu_pct"] <= 100
        assert "source" in r


def test_globe_distance_known_route():
    r = asyncio.run(globe_distance({"from_city": "amritsar", "to_city": "london"}))
    # ਅਸਲ great-circle ≈ 6300–6400 km — ਸੀਮਾ ਵਿੱਚ ਹੋਣਾ ਚਾਹੀਦਾ
    assert 6000 < r["distance_km"] < 6700
    assert r["flight_hours_est"] is not None


def test_globe_distance_unknown_city_is_honest():
    r = asyncio.run(globe_distance({"from_city": "atlantis", "to_city": "london"}))
    assert "error" in r and "known_cities" in r


def test_haversine_same_point_zero():
    assert haversine_km(CITIES["delhi"], CITIES["delhi"]) < 0.01


def test_circuit_schematic_known_and_unknown():
    r = asyncio.run(circuit_schematic({"sensor": "dht11", "board": "esp32"}))
    assert "GPIO4" in r["schematic"] and "verify" in r["warning"]
    bad = asyncio.run(circuit_schematic({"sensor": "fluxcapacitor", "board": "x"}))
    assert "error" in bad and "known" in bad


# -- undo7 -------------------------------------------------------------------- #

def test_undo_empty_stack():
    r = asyncio.run(UndoStack().undo_last())
    assert "error" in r


def test_undo_move_restores_file(tmp_path: Path):
    src = tmp_path / "a.txt"
    src.write_text("data")
    dst_dir = tmp_path / "docs"
    dst_dir.mkdir()
    dst = dst_dir / "a.txt"
    src.rename(dst)
    assert not src.exists() and dst.exists()

    stack = UndoStack()
    stack.push("move a.txt", make_move_undo(src, dst))
    r = asyncio.run(stack.undo_last())
    assert r["undone"] == "move a.txt"
    assert src.exists() and not dst.exists()


def test_undo_write_restores_previous_content(tmp_path: Path):
    f = tmp_path / "n.md"
    f.write_text("old")
    stack = UndoStack()
    stack.push("write n.md", make_write_undo(f, "old"))
    f.write_text("new")
    asyncio.run(stack.undo_last())
    assert f.read_text() == "old"


def test_undo_new_file_gets_deleted(tmp_path: Path):
    f = tmp_path / "fresh.md"
    stack = UndoStack()
    stack.push("create fresh.md", make_write_undo(f, None))
    f.write_text("x")
    asyncio.run(stack.undo_last())
    assert not f.exists()


def test_new_web_shortcuts():
    from apps7 import known_apps, WEB_SHORTCUTS, APP_ALIASES
    for name in ("google", "github", "wikipedia", "facebook", "instagram", "amazon"):
        assert name in known_apps(), name
        target = APP_ALIASES[name]["Linux"]
        assert target in WEB_SHORTCUTS, target
        assert WEB_SHORTCUTS[target].startswith("https://")
