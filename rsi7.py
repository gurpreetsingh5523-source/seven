"""rsi7 — 7 ਦਾ ਸਵੈ-ਸੁਧਾਰ ਘੇਰਾ (RSI loop) + ਮਾਹਿਰ-ਬੋਰਡ ਵੋਟਿੰਗ।

ਸਿਧਾਂਤ (ਸਖ਼ਤ):
  ਦਿਮਾਗ਼ ਤੱਕ ਹੀ — *ਵਿਹਾਰ* ਸੁਧਾਰ ਸਕਦਾ ਹੈ, *ਅਧਿਕਾਰ* ਕਦੇ ਨਹੀਂ।
  - ਪ੍ਰਸਤਾਵ → ਮਨਜ਼ੂਰੀ → ਲਾਗੂ (+ਰੋਲਬੈਕ-ਪੁਆਇੰਟ) → ਨਤੀਜੇ ਨਾਲ ਆਪੇ ਸਾਬਤ/ਰੱਦ
  - ਕਦੇ ਸੁਝਾਅ ਦਾ ਵਿਸ਼ਾ ਨਹੀਂ: ਟੂਲ-ਅਧਿਕਾਰ, ਪਾਲਿਸੀ ਨਿਯਮ, ਬਜਟ-ਹੱਦ, ਆਡਿਟ-ਬੰਦ ਕਰਨਾ
  - ਸੁਧਾਰ ਦਾ ਸਕੋਰ ਪਹਿਲਾਂ/ਬਾਅਦ ਦੀ ਸਫਲਤਾ-ਦਰ ਨਾਲ — ਮਹਿਸੂਸ ਨਾਲ ਨਹੀਂ

ਵੋਟਿੰਗ: ਮਾਹਿਰ-ਬੋਰਡ — ਟਾਇ ਹੋਵੇ ਜਾਂ ਬਹੁਮਤ ਨਾ ਬਣੇ ਤਾਂ ConsensusError।
ਉੱਪਰ ਵਾਲੀ ਪਰਤ (prolepsis/ਸਰਵਿਸ) ਇਸਨੂੰ ਫੜ ਕੇ ਘੱਟ-ਜੋਖ਼ਮ ਵਿਕਲਪ ਲੈਂਦੀ ਹੈ।
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional


class ConsensusError(Exception):
    """ਬੋਰਡ ਇੱਕ ਮਤ ਨਹੀਂ ਹੋ ਸਕਿਆ — ਘੱਟ-ਜੋਖ਼ਮ ਵਿਕਲਪ ਲਓ।"""


# ਇਹ ਵਿਸ਼ੇ ਕਦੇ ਸਵੈ-ਸੁਧਾਰ ਦੇ ਦਾਇਰੇ ਵਿਚ ਨਹੀਂ ਆਉਣਗੇ
FORBIDDEN_TARGETS = (
    "policy", "approval", "budget", "ledger", "audit", "egress",
    "sandbox", "secrets", "sentinel", "capabilit",
)


@dataclass
class Proposal:
    """ਵਿਹਾਰ-ਸੁਧਾਰ ਦਾ ਪ੍ਰਸਤਾਵ — ਲਾਗੂ ਹੋਣ ਤੋਂ ਪਹਿਲਾਂ ਮਨਜ਼ੂਰੀ ਲਾਜ਼ਮੀ।"""
    id: str
    created_at: float
    title: str
    rationale: str            # ਕਿਉਂ — DNA/dream ਸਬੂਤ ਸਮੇਤ
    patch: str                # ਵਿਹਾਰ-ਬਦਲਾਅ (persona ਟੈਕਸਟ / ਰੂਟਿੰਗ-ਸੰਕੇਤ)
    target: str               # ਕਿਹੜਾ ਏਜੰਟ/ਹਿੱਸਾ
    requires_approval: bool = True
    status: str = "pending"   # pending → approved → applied → verified | rolled_back | rejected
    rollback_patch: str = ""
    baseline_score: float = 0.0
    after_score: float = 0.0


class EvolutionLog:
    """ਸਾਰੇ ਪ੍ਰਸਤਾਵਾਂ ਦੀ ਅਟੱਲ ਕਿਤਾਬ — hash-chain ਆਡਿਟ ਤੋਂ ਵੱਖ, ਸਿੱਖ ਲਈ।"""

    def __init__(self, path: Path):
        self.path = path
        self.proposals: List[dict] = []
        if path.exists():
            try:
                self.proposals = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.proposals = []

    def record(self, p: Proposal) -> None:
        self.proposals.append(asdict(p))
        self.path.write_text(json.dumps(self.proposals, ensure_ascii=False,
                                        indent=1), encoding="utf-8")

    def update(self, pid: str, **fields) -> Optional[dict]:
        for p in self.proposals:
            if p["id"] == pid:
                p.update(fields)
                self.path.write_text(json.dumps(
                    self.proposals, ensure_ascii=False, indent=1),
                    encoding="utf-8")
                return p
        return None

    def verified_count(self) -> int:
        return sum(1 for p in self.proposals if p["status"] == "verified")


def safety_check(p: Proposal) -> Optional[str]:
    """ਪ੍ਰਸਤਾਵ ਵਿਹਾਰ-ਦਾਇਰੇ ਵਿਚ ਹੈ? ਨਹੀਂ ਤਾਂ ਕਾਰਨ ਦੇ ਕੇ ਰੋਕੋ।"""
    hay = f"{p.title} {p.patch} {p.target}".lower()
    for bad in FORBIDDEN_TARGETS:
        if bad in hay:
            return (f"'{bad}' ਸਵੈ-ਸੁਧਾਰ ਤੋਂ ਬਾਹਰ ਹੈ — ਸਿਰਫ਼ ਵਿਹਾਰ ਬਦਲਿਆ "
                    f"ਜਾ ਸਕਦਾ, ਅਧਿਕਾਰ ਨਹੀਂ")
    if not p.rollback_patch:
        return "ਰੋਲਬੈਕ-ਪੁਆਇੰਟ ਬਿਨਾਂ ਕੋਈ ਪ੍ਰਸਤਾਵ ਨਹੀਂ"
    return None


def propose(title: str, rationale: str, patch: str, target: str,
            rollback_patch: str) -> Proposal:
    return Proposal(id=f"evo_{uuid.uuid4().hex[:10]}",
                    created_at=time.time(), title=title,
                    rationale=rationale, patch=patch, target=target,
                    rollback_patch=rollback_patch)


def apply_proposal(p: Proposal, log: EvolutionLog,
                   approved: bool = False) -> Proposal:
    """default-deny: ਮਨਜ਼ੂਰੀ ਬਿਨਾਂ ਲਾਗੂ ਨਹੀਂ; ਸੁਰੱਖਿਆ-ਜਾਂਚ ਬਿਨਾਂ ਕਦੇ ਨਹੀਂ।"""
    reason = safety_check(p)
    if reason:
        p.status = "rejected"
        p.rationale += f" | ਰੱਦ: {reason}"
        log.record(p)
        return p
    if p.requires_approval and not approved:
        p.status = "pending"
        log.record(p)
        return p
    p.status = "applied"
    log.record(p)
    return p


def score_delta(before: float, after: float) -> float:
    """ਸੁਧਾਰ = ਬਾਅਦ − ਪਹਿਲਾਂ। ਸਕੋਰ ਹਮੇਸ਼ਾ ਅਸਲੀ ਸਫਲਤਾ-ਦਰ ਤੋਂ।"""
    return round(after - before, 4)


def settle(p: Proposal, log: EvolutionLog, after_score: float,
           threshold: float = 0.0) -> Proposal:
    """ਨਤੀਜੇ ਨਾਲ ਫੈਸਲਾ: ਸਕੋਰ ਵਧਿਆ → verified, ਨਹੀਂ → rolled_back।"""
    p.after_score = after_score
    if score_delta(p.baseline_score, after_score) > threshold:
        p.status = "verified"
    else:
        p.status = "rolled_back"
    log.update(p.id, status=p.status, after_score=after_score)
    return p


# --------------------------------------------------------------------------- #
# ਮਾਹਿਰ-ਬੋਰਡ ਵੋਟਿੰਗ — ਵਿਕਲਪਾਂ ਵਿਚ ਫੈਸਲਾ                                        #
# --------------------------------------------------------------------------- #

def board_vote(votes: Dict[str, str],
               allowed: Optional[List[str]] = None) -> str:
    """votes: {specialist_key: choice} → ਜੇਤੂ ਵਿਕਲਪ।

    ਟਾਇ ਜਾਂ ਬਹੁਮਤ-ਘਾਟ → ConsensusError (ਉੱਤੇ ਫੇਲ-ਸੇਫ਼ ਘੱਟ-ਜੋਖ਼ਮ ਵਿਕਲਪ)।
    """
    if not votes:
        raise ConsensusError("ਕੋਈ ਵੋਟ ਨਹੀਂ")
    tally: Dict[str, int] = {}
    for _, choice in votes.items():
        if allowed is not None and choice not in allowed:
            continue
        tally[choice] = tally.get(choice, 0) + 1
    if not tally:
        raise ConsensusError("ਸਾਰੀਆਂ ਵੋਟਾਂ ਅਵੈਧ")
    best = max(tally.values())
    winners = [c for c, n in tally.items() if n == best]
    if len(winners) != 1 or best < (len(votes) // 2 + 1) and len(votes) >= 3:
        raise ConsensusError(f"ਬਹੁਮਤ ਨਹੀਂ: {tally}")
    return winners[0]
