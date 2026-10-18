"""rsi7 + prolepsis ਦੇ ਟੈਸਟ"""
from pathlib import Path

import pytest

from rsi7 import (ConsensusError, EvolutionLog, apply_proposal, board_vote,
                  propose, safety_check, settle, score_delta)
from prolepsis import Prolepsis


# ---- rsi7 ----

def test_forbidden_targets_blocked():
    p = propose("ਬਜਟ ਵਧਾਓ", "ਸਸਤਾ ਪਵੇਗਾ", "budget: unlimited", "ledger",
                rollback_patch="budget: 1.0")
    assert safety_check(p) is not None  # budget/ledger → ਰੋਕ
    p2 = propose("ਵਿਹਾਰ ਸੁਧਾਰ", "DNA ਸਬੂਤ", "persona_hint: ਜਾਂਚ ਪਹਿਲਾਂ",
                 "khoji", rollback_patch="persona_hint: old")
    assert safety_check(p2) is None


def test_no_rollback_no_proposal():
    p = propose("x", "y", "patch", "team", rollback_patch="")
    assert safety_check(p) is not None


def test_default_deny_then_approved(tmp_path: Path):
    log = EvolutionLog(tmp_path / "evo.json")
    p = propose("ਵਿਹਾਰ", "ਸਬੂਤ", "hint: new", "khoji", rollback_patch="hint: old")
    p = apply_proposal(p, log, approved=False)
    assert p.status == "pending"          # default-deny
    p = propose("ਵਿਹਾਰ", "ਸਬੂਤ", "hint: new", "khoji", rollback_patch="hint: old")
    p = apply_proposal(p, log, approved=True)
    assert p.status == "applied"


def test_settle_verified_vs_rolled_back(tmp_path: Path):
    log = EvolutionLog(tmp_path / "evo.json")
    p = propose("ਵਿਹਾਰ", "ਸਬੂਤ", "hint: new", "khoji", rollback_patch="hint: old")
    p.baseline_score = 0.5
    p = apply_proposal(p, log, approved=True)
    settle(p, log, after_score=0.7)
    assert log.proposals[-1]["status"] == "verified"
    p2 = propose("ਵਿਹਾਰ2", "ਸਬੂਤ", "hint: x", "coder", rollback_patch="hint: old")
    p2.baseline_score = 0.5
    apply_proposal(p2, log, approved=True)
    settle(p2, log, after_score=0.4)
    assert log.proposals[-1]["status"] == "rolled_back"
    assert log.verified_count() == 1
    assert score_delta(0.5, 0.7) == 0.2


def test_board_vote_majority_and_tie():
    assert board_vote({"a": "stdlib", "b": "stdlib", "c": "forge"}) == "stdlib"
    with pytest.raises(ConsensusError):
        board_vote({"a": "x", "b": "y", "c": "z", "d": "x"})  # ਬਹੁਮਤ ਨਹੀਂ
    with pytest.raises(ConsensusError):
        board_vote({})
    with pytest.raises(ConsensusError):
        board_vote({"a": "x"}, allowed=["y"])  # ਅਵੈਧ ਵੋਟ


# ---- prolepsis ----

def test_split_and_kind():
    p = Prolepsis(Path("/tmp/nonexistent_fore.json"))
    parts = p.split("ਪੰਜਾਬ ਖ਼ਬਰ ਲੱਭੋ, ਫਿਰ ਰਿਪੋਰਟ ਲਿਖੋ")
    assert len(parts) == 2
    assert p.kind_of("ਖ਼ਬਰ ਲੱਭੋ") == "search"
    assert p.kind_of("ਰਿਪੋਰਟ ਲਿਖੋ") == "build"


def test_predict_has_routes_and_winner(tmp_path: Path):
    p = Prolepsis(tmp_path / "f.json")
    f = p.predict("weather ਲੁਧਿਆਣਾ", known_tools=["weather_report"])
    assert f.steps and f.predicted_effort_min > 0
    for step in f.steps:
        assert len(step["routes"]) >= 2            # ਤੁਲਨਾ ਲਾਜ਼ਮੀ
        assert step["winner"] == step["routes"][0]["name"]  # ਘੱਟ-ਸਕੋਰ ਜੇਤੂ
    # ਮੌਜੂਦਾ ਟੂਲ ਵਾਲਾ ਹਿੱਸਾ reuse ਰਸਤਾ ਚੁਣੇ
    assert f.steps[0]["winner"] == "ਮੌਜੂਦਾ ਟੂਲ ਸਿੱਧਾ"


def test_settle_learns_calibration(tmp_path: Path):
    p = Prolepsis(tmp_path / "f.json")
    f = p.predict("ਕੋਡ ਲਿਖੋ")
    p.settle(f.id, actual_effort_min=f.predicted_effort_min * 2)  # ਅੱਧਾ ਅਨੁਮਾਨ ਸੀ
    assert p.calibration() > 1.0                     # ਅਗਲੀ ਵਾਰ ਵੱਧ ਅਨੁਮਾਨ
    f2 = p.predict("ਕੋਡ ਲਿਖੋ")
    assert f2.predicted_effort_min > f.predicted_effort_min
    rep = p.report()
    assert rep["settled"] == 1 and rep["mean_prediction_error"] > 1


def test_persistence(tmp_path: Path):
    path = tmp_path / "f.json"
    p1 = Prolepsis(path)
    f = p1.predict("ਕੰਮ")
    p2 = Prolepsis(path)
    assert len(p2.history) == 1 and p2.history[0]["id"] == f.id
