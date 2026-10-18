"""test_wisdom7.py — FailureDNA, EpisodicMemory, DreamEngine ਦੀਆਂ ਟੈਸਟਾਂ।"""
import asyncio
from pathlib import Path

from wisdom7 import (DreamEngine, EpisodicMemory, FailureDNA, RECIPES,
                     categorize_error)


# -- categorize_error -------------------------------------------------------- #

def test_categorize_common_errors():
    assert categorize_error("Connection timed out") == "timeout"
    assert categorize_error("Permission denied: /root") == "permission_denied"
    assert categorize_error("404 Not Found") == "not_found"
    assert categorize_error("connection refused") == "network_error"
    assert categorize_error("daily budget exceeded") == "resource_limit"
    assert categorize_error("Expecting value: json parse") == "parse_error"
    assert categorize_error("policy veto: rule of two") == "policy_block"
    assert categorize_error("??? weird ???") == "unknown"


# -- FailureDNA -------------------------------------------------------------- #

def test_dna_learns_and_counts(tmp_path: Path):
    dna = FailureDNA(tmp_path / "dna.json")
    e1 = dna.learn("web_fetch", "Connection timed out")
    e2 = dna.learn("web_fetch", "request timed out after 15s")
    assert e1["signature"] == e2["signature"]  # ਇੱਕੋ category = ਇੱਕੋ signature
    assert e2["count"] == 2
    assert e2["category"] == "timeout"


def test_dna_remembers_successful_fix(tmp_path: Path):
    dna = FailureDNA(tmp_path / "dna.json")
    assert dna.known_fix("app_open", "Connection timed out") is None
    dna.learn("app_open", "Connection timed out")
    dna.mark_fixed("app_open", "Connection timed out", "retry")
    assert dna.known_fix("app_open", "Connection timed out") == "retry"


def test_dna_recipe_policy_block_never_retries(tmp_path: Path):
    dna = FailureDNA(tmp_path / "dna.json")
    recipe = dna.recipe("cloud_terminal", "policy veto: rule of two")
    assert recipe[0]["action"] == "escalate"
    # ਸੁਰੱਖਿਆ ਨਿਯਮ: veto 'ਤੇ ਕਦੇ retry recipe ਨਹੀਂ
    assert all(r["action"] != "retry" for r in recipe)


def test_dna_persists_across_instances(tmp_path: Path):
    p = tmp_path / "dna.json"
    FailureDNA(p).learn("x", "timeout")
    dna2 = FailureDNA(p)
    assert dna2.report()["total_signatures"] == 1


def test_dna_report_orders_by_frequency(tmp_path: Path):
    dna = FailureDNA(tmp_path / "dna.json")
    for _ in range(3):
        dna.learn("a", "timeout")
    dna.learn("b", "not found")
    rep = dna.report()
    assert rep["by_category"]["timeout"] == 1  # signatures, not counts
    assert rep["most_frequent"][0]["tool"] == "a"
    assert rep["most_frequent"][0]["count"] == 3


# -- EpisodicMemory ---------------------------------------------------------- #

def test_episodes_record_search_tags(tmp_path: Path):
    mem = EpisodicMemory(tmp_path / "ep.json")
    mem.record("youtube ਖੋਲ੍ਹਿਆ", "site_open success", tags=["exec", "web"])
    mem.record("deploy ਵੀਟੋ", "cloud_terminal vetoed", tags=["veto"])
    assert len(mem) == 2
    assert mem.search("youtube")[0]["title"] == "youtube ਖੋਲ੍ਹਿਆ"
    assert len(mem.by_tag("veto")) == 1
    assert len(mem.recent(1)) == 1


def test_episodes_persist_and_cap(tmp_path: Path):
    p = tmp_path / "ep.json"
    mem = EpisodicMemory(p, capacity=5)
    for i in range(8):
        mem.record(f"ep {i}", "x")
    assert len(EpisodicMemory(p, capacity=5)) == 5  # cap ਲਾਗੂ + persist


# -- DreamEngine ------------------------------------------------------------- #

def test_dream_deterministic_without_llm(tmp_path: Path):
    dna = FailureDNA(tmp_path / "d.json")
    mem = EpisodicMemory(tmp_path / "e.json")
    for _ in range(3):
        dna.learn("web_fetch", "timeout")
    mem.record("a", "b")
    dream = DreamEngine(mem, dna, router=None)
    out = asyncio.run(dream.dream())
    assert out["creative"] is None
    assert any("web_fetch" in i for i in out["insights"])


def test_dream_calm_when_no_patterns(tmp_path: Path):
    dream = DreamEngine(EpisodicMemory(tmp_path / "e.json"),
                        FailureDNA(tmp_path / "d.json"))
    out = asyncio.run(dream.dream())
    assert "ਸੁਚਾਰੂ" in out["insights"][0]


def test_dream_veto_pattern_insight(tmp_path: Path):
    mem = EpisodicMemory(tmp_path / "e.json")
    for i in range(3):
        mem.record(f"veto {i}", "policy veto", tags=["veto"])
    dream = DreamEngine(mem, FailureDNA(tmp_path / "d.json"))
    out = asyncio.run(dream.dream())
    assert any("ਵੀਟੋ" in i for i in out["insights"])
