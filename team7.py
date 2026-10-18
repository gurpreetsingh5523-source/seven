"""team7 — 7 ਦੀ ਵਿਸ਼ੇਸ਼ ਏਜੰਟ-ਟੀਮ (specialist agent team).

ਵਿਚਾਰ: Dots ਵਾਂਗ ਵੱਖ-ਵੱਖ ਹੁਨਰਮੰਦ ਏਜੰਟ + Amrit (19 ਏਜੰਟ) ਦੀ ਟੀਮ-ਸੋਚ,
ਪਰ Sentinel ਦੇ ਉੱਤੇ — ਏਜੰਟ *ਸਲਾਹ* ਦਿੰਦੇ ਹਨ, ਫੈਸਲਾ PolicyEngine ਹੀ ਕਰਦਾ ਹੈ।

ਹਰ Specialist:
  - ਆਪਣਾ ਟੂਲ-ਦਾਇਰਾ (whitelist) — ਦੂਜਿਆਂ ਦੇ ਟੂਲ ਨਹੀਂ ਛੂਹ ਸਕਦਾ
  - ਆਪਣਾ ਜੋਖ਼ਮ-ਛੱਤ (max_risk) — ਇਸ ਤੋਂ ਉੱਤੇ ਦਾ ਕਦਮ ਮਨਜ਼ੂਰੀ ਮੰਗੇਗਾ
  - ਆਪਣਾ ਵਿਹਾਰ-ਨਿਯਮ (persona) — ਜਿਵੇਂ ponytail ਦੀ "sustained ਆਲਸੀ ਸੀਨੀਅਰ" ਸਿਖਲਾਈ

Ponytail ਤੋਂ ਲਿਆ ਮੂਲ ਫਿਕਰਾ (MIT, ਆਪਣਾ ਕੋਡ): "ਕੋਡ ਘੱਟ ਲਿਖੋ" prompt ਨਹੀਂ —
ਅਸਲੀ static ਜਾਂਚ (review_code) ਜੋ over-engineering ਫੜਦੀ ਹੈ।
"""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ponytail ਦੀ ਸਿਖਰ (ladder) — ਆਪਣੇ ਸ਼ਬਦਾਂ ਵਿਚ; Coder/Critic ਦੋਹਾਂ ਦੇ ਵਿਹਾਰ ਵਿਚ
LAZY_LADDER: List[str] = [
    "ਕੀ ਇਹ ਚੀਜ਼ ਚਾਹੀਦੀ ਹੀ ਹੈ? (YAGNI — ਅੰਦਾਜ਼ੇ ਤੇ ਕੁਝ ਨਾ ਬਣਾਓ)",
    "ਕੋਡਬੇਸ ਵਿਚ ਪਹਿਲਾਂ ਹੀ ਮੌਜੂਦ? → ਦੁਬਾਰਾ ਵਰਤੋ",
    "stdlib ਕਰਦੀ ਹੈ? → stdlib",
    "ਪਲੇਟਫਾਰਮ/ਬ੍ਰਾਊਜ਼ਰ ਆਪੇ ਕਰਦਾ ਹੈ? → ਨੇਟਿਵ",
    "ਪਹਿਲਾਂ-ਇੰਸਟਾਲ ਡਿਪੈਂਡੈਂਸੀ? → ਉਹੀ, ਨਵੀਂ ਨਹੀਂ",
    "ਇੱਕ ਸਤਰ ਚੱਲਦੀ ਹੈ? → ਇੱਕ ਸਤਰ",
    "ਫਿਰ ਹੀ ਘੱਟੋ-ਘੱਟ ਕੰਮ ਕਰਨ ਵਾਲਾ ਕੋਡ",
]

STDLIB = set(__import__("sys").stdlib_module_names)


@dataclass(frozen=True)
class Specialist:
    key: str
    name: str            # ਪੰਜਾਬੀ ਨਾਮ — UI ਵਿਚ ਦਿੱਸੇਗਾ
    role: str
    tools: tuple         # whitelist — ਸਿਰਫ਼ ਇਹ ਟੂਲ
    max_risk: str        # "low" | "medium" | "high"
    persona: str
    ladder: bool = False  # ponytail-ਸਿਖਰ ਲਾਗੂ?


SPECIALISTS: Dict[str, Specialist] = {
    "soofi": Specialist(
        "soofi", "ਸੂਫ਼ੀ", "ਮੁੱਖ ਚੇਹਰਾ — ਆਮ ਗੱਲਬਾਤ ਤੇ ਟੀਮ-ਰੁਤਬਾ",
        ("chat_reply", "team_status"), "low",
        "ਤੂੰ 7 ਦਾ ਚੇਹਰਾ ਹੈਂ। ਸਾਫ਼, ਨਿਮਰ, ਪੰਜਾਬੀ-ਪਹਿਲਾਂ।"),
    "khoji": Specialist(
        "khoji", "ਖੋਜੀ", "ਖੋਜ, ਰਿਸਰਚ, ਵੈਬ-ਬ੍ਰਾਊਜ਼ਿੰਗ, ਮੌਸਮ",
        ("web_search", "smart_search", "browser_task", "weather_report",
         "site_open", "recall"), "medium",
        "ਸਰੋਤ ਦੱਸ, ਅੰਦਾਜ਼ਾ ਨਾ ਮਾਰ। ਨਹੀਂ ਲਭਿਆ ਤਾਂ ਸਿੱਧਾ ਕਹਿ।"),
    "coder": Specialist(
        "coder", "ਕੋਡਰ", "ਨਵਾਂ ਕੋਡ/ਸਕਿੱਲ — ਸਿਖਰ-ਨਿਯਮ ਨਾਲ",
        ("skill_forge_draft", "code_review"), "medium",
        "ਸਿਖਰ ਚੜ੍ਹ: " + " → ".join(LAZY_LADDER[:4]) + " …", ladder=True),
    "samikhia": Specialist(
        "samikhia", "ਸਮੀਖਕ", "ਕੋਡ-ਸਮੀਖਿਆ — ਘੱਟੋ-ਘੱਟ diff, ਜੜ੍ਹ-ਕਾਰਣ ਫਿਕਸ",
        ("code_review",), "low",
        "ਆਲਸੀ ਸੀਨੀਅਰ ਡਿਵ ਵਰਗਾ: 50 ਸਤਰਾਂ ਵੇਖ ਕੇ 1 ਲਿਖ। ਸਿਫ਼ਤ ਨਹੀਂ, ਸੱਚ।",
        ladder=True),
    "ulat": Specialist(
        "ulat", "ਉਲਟ-ਇੰਜੀਨੀਅਰ", "ਰਿਵਰਸ-ਇੰਜੀਨੀਅਰਿੰਗ — ਬਾਈਨਰੀ/APK ਜਾਂਚ (read-only)",
        ("reverse_inspect", "reverse_strings"), "low",
        "ਨਮੂਨਾ *ਕਦੇ ਨਹੀਂ* ਚਲਾਉਂਦਾ — ਸਿਰਫ਼ ਪੜ੍ਹਦਾ ਹੈ। ਹਰ ਦਾਅਵੇ ਨਾਲ ਸਬੂਤ।"),
    "chaukidar": Specialist(
        "chaukidar", "ਚੌਕੀਦਾਰ", "ਸੁਰੱਖਿਆ — ਆਡਿਟ, ਪਾਲਿਸੀ, ਅਸਫਲਤਾ-ਰਿਪੋਰਟ",
        ("failure_report", "recall"), "low",
        "policy_block ਕਦੇ retry ਨਹੀਂ। ਸ਼ੱਕ ਹੋਵੇ ਤਾਂ ਰੋਕ — ਬਾਅਦ ਵਿਚ ਪੁੱਛ ਲੈਂਗੇ।"),
    "karigar": Specialist(
        "karigar", "ਕਾਰੀਗਰ", "ਹਾਰਡਵੇਅਰ — Arduino/ESP32, ਸਕੀਮੈਟਿਕ, ਸਿਸਟਮ",
        ("circuit_schematic", "arduino_generate", "system_monitor"), "low",
        "ਅਸਲੀ ਹਾਰਡਵੇਅਰ ਅਦਾਕਾਰੇ ਤੋਂ ਭਟਕਦਾ ਹੈ — ਕੈਲੀਬ੍ਰੇਸ਼ਨ ਦੀ ਥਾਂ ਛੱਡ।"),
    "sutradhar": Specialist(
        "sutradhar", "ਸੂਤਰਧਾਰ", "ਯਾਦਦਾਸ਼ਤ — episodes, ਸੁਪਨੇ, ਸਿੱਖ",
        ("recall", "dream", "memory_write", "failure_report"), "low",
        "ਹਰ ਕੰਮ ਤੋਂ ਸਿੱਖ ਬਚਾਉਂਦਾ ਹੈ; ਖਾਲੀ ਵੇਲੇ ਜੋੜਦਾ ਹੈ।"),
    "jantri": Specialist(
        "jantri", "ਜੰਤਰੀ", "ਫਾਈਲਾਂ, ਦਸਤਾਵੇਜ਼, ਪ੍ਰੋਜੈਕਟ-ਬੋਰਡ, undo",
        ("file_organizer", "doc_builder", "project_board", "undo",
         "app_open"), "high",
        "ਉਲਟੇ ਜਾ ਸਕਣ ਵਾਲਾ ਪਹਿਲਾਂ; undo-ਐਂਟਰੀ ਬਿਨਾਂ ਕੁਝ ਨਹੀਂ।"),
}

# ਸ਼ਬਦ → specialist (ਕ੍ਰਮ ਮਹੱਤਵਪੂਰਨ: ਪਹਿਲਾਂ ਮਿਲਦਾ ਜਿੱਤਦਾ)
_ROUTE = [
    (("reverse", "ਰਿਵਰਸ", "ਉਲਟ", "binary", "ਬਾਈਨਰੀ", "apk", "exe",
      "disassembl", "ਮੈਲਵੇਅਰ", "malware"), "ulat"),
    (("review", "ਰਿਵਿਊ", "ਸਮੀਖਿਆ", "ਕੋਡ ਵੇਖ", "check code", "ਸੁਧਾਰ ਕੋਡ"),
     "samikhia"),
    (("code", "ਕੋਡ", "script", "ਸਕ੍ਰਿਪਟ", "function", "ਬਣਾ ਦੇ ਕੋਡ"), "coder"),
    (("search", "ਸਰਚ", "ਲੱਭ", "ਖੋਜ", "weather", "ਮੌਸਮ", "news", "ਖ਼ਬਰ"),
     "khoji"),
    (("circuit", "ਸਰਕਟ", "arduino", "esp32", "ਸਕੀਮੈਟਿਕ", "sensor",
      "ਸੈਂਸਰ"), "karigar"),
    (("ਯਾਦ", "recall", "ਸੁਪਨਾ", "dream", "ਸਿੱਖ", "memory"), "sutradhar"),
    (("audit", "ਆਡਿਟ", "ਫੇਲ", "failure", "ਸੁਰੱਖਿਆ", "security",
      "policy"), "chaukidar"),
    (("file", "ਫਾਈਲ", "organize", "document", "ਦਸਤਾਵੇਜ਼", "board",
      "ਖੋਲ੍ਹ", "ਖੋਲੋ", "open ", "undo"), "jantri"),
]


def route(message: str) -> Specialist:
    """ਸੁਨੇਹਾ ਕਿਸ ਮਾਹਿਰ ਕੋਲ ਜਾਵੇ — ਡੀਫਾਲਟ ਸੂਫ਼ੀ।"""
    low = message.lower()
    for keys, spec in _ROUTE:
        if any(k in low for k in keys):
            return SPECIALISTS[spec]
    return SPECIALISTS["soofi"]


class Team:
    """ਟੀਮ-ਰਜਿਸਟਰੀ: ਕੌਣ ਹੈ, ਕੀ ਕਰ ਸਕਦਾ ਹੈ, ਰੁਤਬਾ।"""

    def __init__(self) -> None:
        self.agents = SPECIALISTS

    def roster(self) -> List[dict]:
        return [{"key": a.key, "name": a.name, "role": a.role,
                 "tools": list(a.tools), "max_risk": a.max_risk,
                 "ladder": a.ladder} for a in self.agents.values()]

    def assign(self, message: str) -> dict:
        a = route(message)
        return {"specialist": a.key, "name": a.name, "role": a.role,
                "allowed_tools": list(a.tools), "max_risk": a.max_risk,
                "persona": a.persona}

    def can_use(self, spec_key: str, tool: str) -> bool:
        a = self.agents.get(spec_key)
        return bool(a) and tool in a.tools


# --------------------------------------------------------------------------- #
# code_review — ponytail-ਸਿਖਰ ਦੀ ਅਸਲੀ static ਜਾਂਚ (prompt ਨਹੀਂ, ਕੋਡ)          #
# --------------------------------------------------------------------------- #

def review_code(code: str, max_fn_lines: int = 40) -> dict:
    """ਓਵਰ-ਇੰਜੀਨੀਅਰਿੰਗ ਤੇ ਖ਼ਤਰਨਾਕ ਪੈਟਰਨ ਫੜੋ।

    ਨਿਯਮ (ਪਹਿਲਾਂ-ਬਾਤਰੇ):
      - non-stdlib import → ਨਵੀਂ ਡਿਪੈਂਡੈਂਸੀ, ਲੋੜ ਸਾਬਤ ਕਰੋ
      - ਇੱਕ-ਤਰੀਕੇ ਵਾਲੀ class → ਬਿਨਾਂ-ਮੰਗੀ abstraction (ladder rung 1)
      - ਲੰਬਾ function → ਵੰਡੋ
      - bare except / eval / exec → ਖ਼ਤਰਨਾਕ, ਛੱਡੋ ਨਹੀਂ
      - TODO ਲਈ ਬਣਾਇਆ 'later' ਸਕੈਫ਼ੋਲਡਿੰਗ → ਹਟਾਓ
    """
    issues: List[dict] = []
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError as e:
        return {"ok": False, "issues": [{"rule": "syntax", "line": e.lineno,
                                          "msg": f"ਪਾਰਸ ਨਹੀਂ ਹੋਇਆ: {e.msg}"}],
                "ladder": LAZY_LADDER}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = (node.names[0].name if isinstance(node, ast.Import)
                   else (node.module or "")).split(".")[0]
            if mod and mod not in STDLIB and not mod.startswith("_"):
                issues.append({"rule": "new_dependency", "line": node.lineno,
                               "msg": f"'{mod}' non-stdlib ਹੈ — ਕੀ stdlib/ਮੌਜੂਦਾ "
                                      f"ਡਿਪੈਂਡੈਂਸੀ ਨਾਲ ਨਹੀਂ ਬਣਦਾ? (ਸਿਖਰ 3/5)"})
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)
                       and not n.name.startswith("_")]
            if len(methods) == 1:
                issues.append({"rule": "single_method_class",
                               "line": node.lineno,
                               "msg": f"class '{node.name}' ਵਿਚ ਇੱਕੋ ਤਰੀਕਾ — "
                                      f"function ਕਾਫ਼ੀ ਹੈ (ਬਿਨਾਂ-ਮੰਗੀ abstraction)"})
        elif isinstance(node, ast.FunctionDef):
            if node.body:
                span = (node.body[-1].end_lineno or node.lineno) - node.lineno
                if span > max_fn_lines:
                    issues.append({"rule": "long_function", "line": node.lineno,
                                   "msg": f"'{node.name}' {span} ਸਤਰਾਂ — ਵੰਡੋ"})
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append({"rule": "bare_except", "line": node.lineno,
                           "msg": "bare except — ਗੱਡਬੱਡੀ ਨਿਗਲ ਲਏਗਾ; ਕਿਸਮ ਲਿਖੋ"})
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("eval", "exec"):
            issues.append({"rule": "dynamic_exec", "line": node.lineno,
                           "msg": f"{node.func.id}() — skill_forge ਵੀ ਇਹਨਾਂ ਨੂੰ "
                                  f"ਰੋਕਦਾ ਹੈ; ਬਦਲੋ"})

    skipped = len([i for i in issues if i["rule"] == "new_dependency"])
    return {"ok": not issues, "issues": issues, "ladder": LAZY_LADDER,
            "summary": f"{len(issues)} ਸੁਝਾਅ"
                       + (f" ({skipped} ਨਵੀਆਂ ਡਿਪੈਂਡੈਂਸੀਆਂ)" if skipped else "")}
