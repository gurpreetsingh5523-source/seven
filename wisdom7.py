"""
wisdom7.py — "7" ਦਾ ਅਕਲ ਪੱਧਰ: ਅਸਫਲਤਾ ਤੋਂ ਸਿੱਖਣਾ, ਯਾਦ ਰੱਖਣਾ, ਸੁਪਨੇ ਵੇਖਣਾ।

Amrit GODMODE ਤੋਂ ਸਿੱਖੇ ਤਿੰਨ ਵਿਚਾਰ (MIT — ਤੁਹਾਡੀ ਆਪਣੀ repo; ਅੰਦਰੂਨੀ ਬਣਤਰ
ਸਾਡੀ ਆਪਣੀ, ਟੈਸਟ-ਪ੍ਰਮਾਣਿਤ, stdlib-only):

  1. FailureDNA (failure/failure_dna.py ਤੋਂ ਵਿਚਾਰ)
     ਹਰ ਫੇਲ ਹੋਏ ਕਦਮ ਦੀ "ਸਿੱਖ": error signature → category → ਜੋ ਠੀਕ ਕਰਨ
     ਵਿੱਚ ਕਾਮਯਾਬ ਹੋਇਆ ਉਹ ਯਾਦ। ਅਗਲੀ ਵਾਰ ਉਹੀ ਫੇਲ ਹੋਵੇ ਤਾਂ ਪਹਿਲਾਂ ਹੀ ਪਤਾ —
     ਮਨੁੱਖ ਨੂੰ ਦੱਸਣ ਤੋਂ ਪਹਿਲਾਂ ਖੁਦ recipe ਅਜ਼ਮਾਓ।
  2. EpisodicMemory (memory/episodic_memory.py ਤੋਂ ਵਿਚਾਰ)
     ਨਾਮ-ਵਾਲੇ episodes: "ਕੱਲ੍ਹ youtube ਖੋਲ੍ਹਿਆ", "deploy ਵੀਟੋ ਹੋਇਆ" — ਟੈਗ
     ਨਾਲ ਖੋਜ ਯੋਗ। Audit (ਕਾਨੂੰਨੀ ਰਿਕਾਰਡ) ਤੋਂ ਵੱਖ — ਇਹ ਯਾਦਦਾਸ਼ਤ ਹੈ।
  3. DreamEngine (dream_engine.py ਤੋਂ ਵਿਚਾਰ)
     idle ਵੇਲੇ ਪੁਰਾਣੀਆਂ ਘਟਨਾਵਾਂ ਮਿਲਾ ਕੇ ਸਿੱਖ ਕੱਢੋ: "ਇਹ tool ਅੱਜ 4 ਵਾਰ
     timeout ਹੋਇਆ" → ਸਲਾਹ। LLM ਮਿਲੇ ਤਾਂ ਰਚਨਾਤਮਕ recombination,
     ਨਹੀਂ ਤਾਂ deterministic insights — ਦੋਵੇਂ ਇਮਾਨਦਾਰ।
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# 1. FailureDNA — ਫੇਲ → ਸਿੱਖ → ਅਗਲੀ ਵਾਰ ਖੁਦ ਠੀਕ                                #
# --------------------------------------------------------------------------- #

def categorize_error(error: str) -> str:
    e = error.lower()
    if "timeout" in e or "timed out" in e:
        return "timeout"
    if "permission" in e or "denied" in e:
        return "permission_denied"
    if "not found" in e or "no such" in e or "404" in e:
        return "not_found"
    if "connection" in e or "network" in e or "refused" in e:
        return "network_error"
    if "budget" in e or "rate limit" in e or "429" in e:
        return "resource_limit"
    if "json" in e or "parse" in e:
        return "parse_error"
    if "validation" in e or "valueerror" in e or "schema" in e:
        return "validation_error"
    if "veto" in e or "policy" in e:
        return "policy_block"
    return "unknown"


# ਹਰ category ਲਈ recovery recipe — ਕ੍ਰਮ ਵਿੱਚ ਅਜ਼ਮਾਓ
RECIPES: Dict[str, List[Dict[str, Any]]] = {
    "timeout":          [{"action": "retry", "note": "backoff ਨਾਲ ਮੁੜ ਕੋਸ਼ਿਸ਼"},
                         {"action": "escalate", "note": "ਦੋ ਵਾਰ ਬਾਅਦ ਮਨੁੱਖ ਨੂੰ ਦੱਸੋ"}],
    "network_error":    [{"action": "retry", "note": "ਨੈੱਟਵਰਕ ਲਹਿਰ ਹੋ ਸਕਦੀ"},
                         {"action": "switch_provider",
                          "note": "ਲੋਕਲ↔API ਬਦਲੋ (HybridRouter ਆਪੇ ਕਰਦਾ)"}],
    "resource_limit":   [{"action": "switch_provider",
                          "note": "ਬਜਟ ਪਾਰ → local-only ਮੋਡ"}],
    "not_found":        [{"action": "escalate",
                          "note": "path/ਨਾਮ ਗਲਤ ਹੋ ਸਕਦਾ — ਮਨੁੱਖ ਪੁੱਛੋ"}],
    "validation_error": [{"action": "escalate",
                          "note": "payload ਠੀਕ ਕਰਨ ਲਈ ਮਨੁੱਖ ਚਾਹੀਦਾ"}],
    "policy_block":     [{"action": "escalate",
                          "note": "ਵੀਟੋ ਜਾਣਬੁੱਝ ਕੇ ਹੈ — ਕਦੇ bypass ਨਹੀਂ"}],
    "parse_error":      [{"action": "retry", "note": "LLM output ਫਿਰ ਮੰਗੋ"}],
}


class FailureDNA:
    """error signature → {ਵਾਰ, category, ਕਿਹੜਾ fix ਕਾਮਯਾਬ ਹੋਇਆ}।"""

    def __init__(self, path: Path):
        self.path = path
        self._dna: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            try:
                self._dna = json.loads(path.read_text())
            except Exception:
                self._dna = {}

    @staticmethod
    def signature(tool: str, error: str) -> str:
        cat = categorize_error(error)
        return hashlib.sha256(f"{tool}:{cat}".encode()).hexdigest()[:12]

    def learn(self, tool: str, error: str) -> Dict[str, Any]:
        sig = self.signature(tool, error)
        entry = self._dna.setdefault(sig, {
            "tool": tool, "category": categorize_error(error),
            "count": 0, "first_seen": time.time(),
            "successful_fixes": []})
        entry["count"] += 1
        entry["last_seen"] = time.time()
        self._save()
        return {"signature": sig, **entry}

    def known_fix(self, tool: str, error: str) -> Optional[str]:
        entry = self._dna.get(self.signature(tool, error))
        if entry and entry.get("successful_fixes"):
            return entry["successful_fixes"][-1]
        return None

    def mark_fixed(self, tool: str, error: str, fix: str) -> None:
        sig = self.signature(tool, error)
        if sig in self._dna:
            fixes = self._dna[sig].setdefault("successful_fixes", [])
            if fix not in fixes:
                fixes.append(fix)
            self._save()

    def recipe(self, tool: str, error: str) -> List[Dict[str, Any]]:
        return RECIPES.get(categorize_error(error),
                           [{"action": "escalate", "note": "ਅਣਜਾਣ ਫੇਲ — ਮਨੁੱਖ"}])

    def report(self) -> Dict[str, Any]:
        cats = Counter(e["category"] for e in self._dna.values())
        worst = sorted(self._dna.values(), key=lambda e: -e["count"])[:5]
        return {"total_signatures": len(self._dna),
                "by_category": dict(cats),
                "most_frequent": [{"tool": e["tool"], "category": e["category"],
                                   "count": e["count"]} for e in worst]}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._dna, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# 2. EpisodicMemory — ਨਾਮ-ਵਾਲੀਆਂ ਘਟਨਾਵਾਂ, ਟੈਗ ਨਾਲ ਖੋਜ                           #
# --------------------------------------------------------------------------- #

class EpisodicMemory:
    def __init__(self, path: Path, capacity: int = 500):
        self.path = path
        self.capacity = capacity
        self._episodes: List[Dict[str, Any]] = []
        if path.exists():
            try:
                self._episodes = json.loads(path.read_text())
            except Exception:
                self._episodes = []

    def record(self, title: str, content: str,
               tags: Optional[List[str]] = None,
               scope: str = "shared") -> Dict[str, Any]:
        # scope: "shared" = ਸਭ ਮਾਹਿਰ ਵੇਖ ਸਕਦੇ; ਕਿਸੇ specialist-key = ਸਿਰਫ਼ ਉਹੀ
        ep = {"id": len(self._episodes) + 1, "title": title,
              "content": content[:600], "tags": tags or [],
              "scope": scope,
              "timestamp": datetime.now().isoformat(timespec="seconds")}
        self._episodes.append(ep)
        if len(self._episodes) > self.capacity:
            self._episodes = self._episodes[-self.capacity:]
        self._save()
        return ep

    def search(self, query: str,
               scope: Optional[str] = None) -> List[Dict[str, Any]]:
        q = query.lower()
        return [e for e in self._episodes
                if (q in e["title"].lower() or q in e["content"].lower())
                and self._visible(e, scope)]

    @staticmethod
    def _visible(ep: Dict[str, Any], scope: Optional[str]) -> bool:
        # scope=None → ਸਭ (ਮਾਲਕ ਦਾ ਨਜ਼ਰੀਆ); specialist-key → own + shared
        if scope is None:
            return True
        return ep.get("scope", "shared") in ("shared", scope)

    def by_tag(self, tag: str) -> List[Dict[str, Any]]:
        return [e for e in self._episodes if tag in e.get("tags", [])]

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return self._episodes[-n:]

    def __len__(self) -> int:
        return len(self._episodes)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._episodes, indent=2,
                                        ensure_ascii=False))


# --------------------------------------------------------------------------- #
# 3. DreamEngine — idle ਵੇਲੇ ਮਿਲਾ-ਜੁਲਾ ਕੇ ਸਿੱਖ                                   #
# --------------------------------------------------------------------------- #

class DreamEngine:
    """Episodes + FailureDNA ਪੜ੍ਹ ਕੇ insights ਕੱਢਦਾ ਹੈ।
    LLM ਹੋਵੇ ਤਾਂ ਰਚਨਾਤਮਕ "ਸੁਪਨਾ"; ਨਹੀਂ ਤਾਂ deterministic ਸਿੱਖ — ਦੋਵੇਂ ਸੱਚੇ।"""

    def __init__(self, episodes: EpisodicMemory, dna: FailureDNA,
                 router=None):
        self.episodes = episodes
        self.dna = dna
        self.router = router  # HybridRouter (optional)

    def deterministic_insights(self) -> List[str]:
        insights: List[str] = []
        rep = self.dna.report()
        for e in rep["most_frequent"]:
            if e["count"] >= 2:
                cat = e["category"]
                fix_hint = {
                    "timeout": "timeout ਵਧਾਉਣ ਜਾਂ ਕੰਮ ਛੋਟੇ ਟੁਕੜਿਆਂ ਵਿੱਚ ਵੰਡਣ ਬਾਰੇ ਸੋਚੋ",
                    "network_error": "ਨੈੱਟਵਰਕ ਅਸਥਿਰ ਲੱਗਦਾ — local ਮੋਡ ਵਧੇਰੇ ਭਰੋਸੇਮੰਦ ਰਹੇਗਾ",
                    "resource_limit": "API ਖਰਚਾ ਵੱਧ ਰਿਹਾ — local ਮਾਡਲ 'ਤੇ ਜ਼ਿਆਦਾ ਭਾਰ ਪਾਓ",
                }.get(cat, "ਇਸ ਪੈਟਰਨ 'ਤੇ ਨਜ਼ਰ ਰੱਖੋ")
                insights.append(f"'{e['tool']}' {e['count']} ਵਾਰ {cat} — {fix_hint}")
        vetoes = self.episodes.by_tag("veto")
        if len(vetoes) >= 3:
            insights.append(f"{len(vetoes)} ਵੀਟੋ ਹੋ ਚੁੱਕੇ — policy ਬਿਲਕੁਲ ਸਹੀ ਕੰਮ "
                            f"ਕਰ ਰਹੀ; ਜੇ ਇਹ ਅਕਸਰ ਰੁਕਾਵਟ ਬਣੇ ਤਾਂ risk ਲੈਵਲ review ਕਰੋ")
        if not insights:
            insights.append("ਸਭ ਸੁਚਾਰੂ — ਕੋਈ ਦੁਹਰਾਓ ਵਾਲਾ ਪੈਟਰਨ ਨਹੀਂ।")
        return insights

    async def dream(self, trigger: str = "manual") -> Dict[str, Any]:
        """ਇੱਕ 'ਸੁਪਨਾ': deterministic insights + (LLM ਹੋਵੇ ਤਾਂ) recombination।"""
        base = self.deterministic_insights()
        creative = None
        if self.router is not None:
            seeds = [e["title"] for e in self.episodes.recent(8)]
            if seeds:
                prompt = ("ਇਹ ਏਜੰਟ ਦੀਆਂ ਤਾਜ਼ਾ ਘਟਨਾਵਾਂ ਹਨ:\n- " + "\n- ".join(seeds)
                          + "\n\nਇੱਕ ਛੋਟਾ, ਠੋਸ ਸੁਝਾਅ ਦਿਓ (1-2 ਲਾਈਨਾਂ) ਕਿ ਏਜੰਟ "
                            "ਅੱਗੇ ਕੀ ਸੁਧਾਰ ਸਕਦਾ ਹੈ। ਪੰਜਾਬੀ ਵਿੱਚ।")
                try:
                    resp = await self.router.complete(
                        "You are a concise improvement advisor.", prompt,
                        trigger=trigger, n_tools=0)
                    creative = resp.text.strip()[:400]
                except Exception:
                    creative = None
        return {"insights": base, "creative": creative,
                "episodes_seen": len(self.episodes),
                "failure_signatures": self.dna.report()["total_signatures"]}
