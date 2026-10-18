"""prolepsis — ਭਵਿੱਖੀ ਸੰਗਰਚਨਾ ਇੰਜਣ (ਲਿਖਣ ਤੋਂ ਪਹਿਲਾਂ ਸੋਚਣਾ)।

ਤੁਹਾਡੀ ਗੱਲ: "ਰਿਜ਼ਲਟ-ਪ੍ਰੀਡਿਕਸ਼ਨ ਦੇ ਆਧਾਰ ਤੇ ਕੋਡ — ponytail ਵਾਂਗ 'ਲੋੜ ਹੈ ਇਹ
ਲਿਖਣ ਦੀ' ਪਰ ਇੱਕ ਕਦਮ ਅੱਗੇ: ਕੋਡ ਲਿਖਣ ਤੋਂ ਪਹਿਲਾਂ ਹੀ ਤੁਲਨਾਤਮਕ ਅਧਿਐਨ ਜੋ ਭਵਿੱਖੀ
ਬਣਤਰ (future architecture) ਨੂੰ ਜਨਮ ਦੇਵੇ।"

ਲੂਪ:
  1. ਰੁਕਾਵਟ ਨੂੰ ਹਿੱਸਿਆਂ ਵਿਚ ਤੋੜੋ
  2. ਹਰ ਹਿੱਸੇ ਲਈ ਰਸਤੇ (ਘੱਟੋ-ਘੱਟ 2 — ponytail-ਸਿਖਰ ਦੀ ਲੋੜ ਤੋਂ ਨਵਾਂ ਸਕਿੱਲ ਤੱਕ)
  3. ਸਕੋਰ: ਲਾਗਤ + ਕੋਡ-ਮਾਤਰਾ + ਬਰਕਾਰ-ਹੋਣਾ + ਜੋਖ਼ਮ
  4. ਜੇਤੂ ਰਸਤਾ → "ਕਿਸਮਤ" (foresight) ਲਿਖੋ: ਬਣਤਰ, ਅਨੁਮਾਨਤ ਯਤਨ, ਭਰੋਸਾ
  5. ਕੰਮ ਹੋਣ ਮਗਰੋਂ ਅਸਲੀਅਤ ਨਾਲ ਤੁਲਨਾ → ਪੂਰਵਾਨੁਮਾਨ-ਗ਼ਲਤੀ ਦਰਜ
     → ਅਗਲੀ ਵਾਰ ਸਕੋਰਿੰਗ ਆਪੇ ਸਿਆਣੀ (ਹਰ ਵਾਰ ਘਟਦੀ ਗ਼ਲਤੀ = ਲਗਾਤਾਰ ਸਿੱਖਣਾ)

ਸਭ ਲੋਕਲ — ਕੋਈ API ਨਹੀਂ। router ਹੋਵੇ ਤਾਂ ਰਸਤੇ-ਸੁਝਾਅ creative ਵੀ ਹੋ ਸਕਦੇ।
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

# ਯਤਨ-ਇਕਾਈਆਂ (ਮਿੰਟ): ਟੂਲ ਦੀ ਕਿਸਮ ਤੋਂ ਹਿੱਸੇ ਦੀ ਲਾਗਤ ਦਾ ਅਨੁਮਾਨ
_EFFORT = {
    "chat": 1, "search": 3, "read": 2, "compute": 4, "build": 15,
    "reverse": 10, "hardware": 20, "browse": 8, "write": 10,
}

_KEY_KIND = [
    (("search", "ਸਰਚ", "ਲੱਭ", "ਖੋਜ", "weather", "ਮੌਸਮ"), "search"),
    (("reverse", "ਰਿਵਰਸ", "ਉਲਟ", "binary", "apk"), "reverse"),
    (("circuit", "arduino", "esp32", "ਸਰਕਟ"), "hardware"),
    (("browse", "ਵੈਬਸਾਈਟ", "form", "ਫਾਰਮ", "ਖੋਲ੍ਹ"), "browse"),
    (("ਲਿਖ", "ਬਣਾ", "code", "ਕੋਡ", "script", "build", "document"), "build"),
    (("ਪੜ੍ਹ", "read", "ਵੇਖ", "summar"), "read"),
    (("ਗਿਣ", "ਹਿਸਾਬ", "calculate", "ਦੂਰੀ", "distance"), "compute"),
]


@dataclass
class Route:
    name: str                 # ਜਿਵੇਂ "stdlib ਸਿੱਧਾ" / "ਨਵਾਂ ਸਕਿੱਲ ਫੋਰਜ"
    kind: str
    est_effort_min: float
    code_volume: float        # 0-1 — ਕਿੰਨਾ ਕੋਡ ਲਿਖਣਾ ਪਏਗਾ
    reuse: float              # 0-1 — ਕਿੰਨਾ ਮੌਜੂਦਾ ਟੂਲ/ਸਕਿੱਲ ਵਰਤਿਆ ਜਾਂਦਾ
    risk: float               # 0-1 — ਬਾਹਰੀ ਅਸਰ/ਮਨਜ਼ੂਰੀ-ਲੋੜ
    notes: str = ""

    def score(self, cal: float = 1.0) -> float:
        """ਘੱਟ = ਵਧੀਆ। cal = ਅਤੀਤ ਦੀ ਗ਼ਲਤੀ ਤੋਂ ਸਿੱਖਿਆ ਕੈਲੀਬ੍ਰੇਸ਼ਨ।"""
        raw = (self.est_effort_min * 0.5 + self.code_volume * 20
               + self.risk * 15 - self.reuse * 25)
        return round(raw * cal, 2)


@dataclass
class Foresight:
    """ਕਿਸਮਤ — ਲਿਖਣ ਤੋਂ ਪਹਿਲਾਂ ਦਾ ਨਕਸ਼ਾ।"""
    id: str
    created_at: float
    task: str
    steps: List[dict]          # [{part, kind, routes:[{name,score}], winner}]
    architecture: str          # ਭਵਿੱਖੀ ਬਣਤਰ — ਇੱਕ ਪੰਗਤੀ ਨਕਸ਼ਾ
    predicted_effort_min: float
    confidence: float          # 0-1
    actual_effort_min: Optional[float] = None
    prediction_error: Optional[float] = None


class Prolepsis:
    def __init__(self, path: Path):
        self.path = path
        self.history: List[dict] = []
        if path.exists():
            try:
                self.history = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.history = []

    # --- ਅਤੀਤ ਤੋਂ ਕੈਲੀਬ੍ਰੇਸ਼ਨ: ਮੇਰੇ ਅਨੁਮਾਨ ਕਿੰਨੇ ਮਾਰਦੇ ਹਨ? ---
    def calibration(self) -> float:
        errs = [h["prediction_error"] for h in self.history
                if h.get("prediction_error") is not None]
        if not errs:
            return 1.0
        mean_err = sum(errs) / len(errs)          # >1 = ਘੱਟ ਅਨੁਮਾਨ ਲਾਉਂਦਾ ਸੀ
        return round(min(max(mean_err, 0.5), 2.0), 3)

    def mean_error(self) -> Optional[float]:
        errs = [h["prediction_error"] for h in self.history
                if h.get("prediction_error") is not None]
        return round(sum(errs) / len(errs), 3) if errs else None

    # --- ਰੁਕਾਵਟ → ਹਿੱਸੇ ---
    def split(self, task: str) -> List[str]:
        parts = [p.strip() for p in task.replace("ਅਤੇ", ",").replace(
            "ਫਿਰ", ",").replace("then", ",").replace("and", ",").split(",")
                 if p.strip()]
        return parts or [task]

    def kind_of(self, part: str) -> str:
        low = part.lower()
        for keys, kind in _KEY_KIND:
            if any(k in low for k in keys):
                return kind
        return "chat"

    # --- ਹਿੱਸੇ → ਰਸਤੇ (ਘੱਟੋ-ਘੱਟ 2, ਤੁਲਨਾ ਲਾਜ਼ਮੀ) ---
    def routes_for(self, kind: str, reuse_hint: float) -> List[Route]:
        base = _EFFORT.get(kind, 5)
        return [
            Route("ਮੌਜੂਦਾ ਟੂਲ ਸਿੱਧਾ", kind, base * (1 - reuse_hint * 0.6),
                  code_volume=0.05, reuse=reuse_hint, risk=0.1,
                  notes="ਸਿਖਰ: ਪਹਿਲਾਂ ਹੀ ਹੈ ਤਾਂ ਲਿਖੋ ਨਾ"),
            Route("ਨਵਾਂ ਸਕਿੱਲ ਫੋਰਜ", kind, base + 12, code_volume=0.7,
                  reuse=0.1, risk=0.4,
                  notes="ਬਾਰ-ਬਾਰ ਲੋੜ ਪਵੇ ਤਾਂ ਹੀ — ਲੋੜ ਕਾਂਡ ਦੀ ਮਾਂ"),
            Route("ਹੱਥੀਂ/ਅਰਧ-ਆਟੋ", kind, base * 2.5, code_volume=0.0,
                  reuse=0.0, risk=0.05, notes="ਇੱਕ-ਵਾਰੀ ਕੰਮ ਲਈ ਸਭ ਤੋਂ ਸਸਤਾ"),
        ]

    def predict(self, task: str, known_tools: Optional[List[str]] = None
                ) -> Foresight:
        cal = self.calibration()
        known = known_tools or []
        steps = []
        total = 0.0
        for part in self.split(task):
            kind = self.kind_of(part)
            # reuse-ਸੰਕੇਤ: ਇਸ ਕਿਸਮ ਦਾ ਟੂਲ ਪਹਿਲਾਂ ਹੀ ਹੈ?
            reuse = 0.9 if any(kind in t for t in known) else 0.3
            routes = self.routes_for(kind, reuse)
            scored = sorted(routes, key=lambda r: r.score(cal))
            winner = scored[0]
            total += winner.est_effort_min * cal
            steps.append({"part": part, "kind": kind,
                          "routes": [{"name": r.name, "score": r.score(cal)}
                                     for r in scored],
                          "winner": winner.name})
        conf = round(0.55 + 0.35 * (1 if self.history else 0)
                     + 0.1 * (1 if cal != 1.0 else 0), 2)
        arch = " → ".join(f"[{s['kind']}:{s['winner']}]" for s in steps)
        f = Foresight(id=f"fore_{uuid.uuid4().hex[:10]}",
                      created_at=time.time(), task=task, steps=steps,
                      architecture=arch,
                      predicted_effort_min=round(total, 1),
                      confidence=min(conf, 0.95))
        self.history.append(asdict(f))
        self._save()
        return f

    # --- ਅਸਲੀਅਤ ਨਾਲ ਤੁਲਨਾ → ਸਿੱਖ ---
    def settle(self, foresight_id: str, actual_effort_min: float
               ) -> Optional[dict]:
        for h in self.history:
            if h["id"] == foresight_id:
                pred = h["predicted_effort_min"] or 1.0
                h["actual_effort_min"] = actual_effort_min
                h["prediction_error"] = round(actual_effort_min / pred, 3)
                self._save()
                return h
        return None

    def report(self) -> dict:
        settled = [h for h in self.history
                   if h.get("prediction_error") is not None]
        return {"foresights": len(self.history), "settled": len(settled),
                "calibration": self.calibration(),
                "mean_prediction_error": self.mean_error(),
                "learning": ("ਹਰ ਵਾਰ ਸਿਆਣਾ" if settled else
                             "ਅਜੇ ਸਿੱਖਣ ਲਈ ਨਤੀਜੇ ਚਾਹੀਦੇ")}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.history, ensure_ascii=False,
                                        indent=1), encoding="utf-8")
