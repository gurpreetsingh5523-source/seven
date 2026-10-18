"""
seven_ui.py — "7" ਦਾ ਮੁੱਖ ਫੇਸ: voice-first orb ਇੰਟਰਫੇਸ (typing ਵੀ ਨਾਲ)।

ਚਲਾਓ:  python seven_ui.py   →  http://127.0.0.1:7878

ਡਿਜ਼ਾਈਨ:
  * ਵੱਡਾ ਚਮਕਦਾਰ orb — ਆਵਾਜ਼ ਸੁਣਦਾ (ਹਰਾ pulse), ਬੋਲਦਾ (ਨੀਲਾ ripple)
  * 🎙️ ਮੁੱਖ ਬਟਨ — ਬੋਲੋ; ਲਿਖਣ ਲਈ ਹੇਠ input (ਦੋਵੇਂ ਬਰਾਬਰ)
  * ਜਵਾਬ ਬੋਲਕੇ ਵੀ ਸੁਣਾਉਂਦਾ (Web Speech API — ਮੁਫ਼ਤ, ਲੋਕਲ)
  * ਖ਼ਤਰਨਾਕ ਕਦਮ approval ਕਾਰਡ ਨਾਲ ਰੁਕਦਾ; undo ਬਟਨ; audit + forge ਅੰਦਰ ਹੀ
  * ਕੋਈ dependency ਨਹੀਂ — stdlib server + ਇੱਕ HTML ਪੇਜ
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent7 import (CostLedger, HybridRouter, LLMPlanner, LocalOpenAICompatProvider,
                    OllamaProvider, OpenAIProvider)
from capabilities import build_full_registry, EgressPolicy
from apps7 import app_open, site_open, web_search_open, known_apps
from browser7 import browse_task, smart_search, playwright_available
from skills7 import (system_monitor, globe_distance, circuit_schematic,
                     weather_report, CITIES, SCHEMATICS)
from undo7 import UndoStack, make_move_undo
from stt7 import LocalSTT, STTError, MAX_AUDIO_BYTES
from wisdom7 import DreamEngine, EpisodicMemory, FailureDNA
from team7 import Team, review_code, route as team_route
from reverse7 import inspect_file as reverse_inspect, extract_strings
from prolepsis import Prolepsis
from rsi7 import (ConsensusError, EvolutionLog, apply_proposal, board_vote,
                    propose, settle as rsi_settle)
from pages7 import Pages, PageError, ConflictError
from guard7 import scan_untrusted, wrap_tool_output
from sentinel_orchestrator import (
    ActionIntent, ApprovalGate, AuditLog, Capability, CheckpointStore,
    LocalSandbox, Orchestrator, PolicyEngine, RiskLevel, SecretsBroker,
    ToolRegistry, ToolSpec,
)
from skill_forge import DRAFTER_SYSTEM, ForgeError, static_check, verify_in_subprocess

import sys
import time
from collections import deque

PORT = 7878
STATE = Path("./seven_state")
HUD_LOG = Path("/tmp/seven_hud.log")


class RingLog:
    """HUD ਲਈ ਘਟਨਾਵਾਂ ਦੀ ਰਿੰਗ — ਬੂਟ + runtime ਇੱਕੋ ਥਾਂ।"""
    def __init__(self, cap: int = 200):
        self.q = deque(maxlen=cap)
    def add(self, kind: str, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {kind.upper():8} {msg}"
        self.q.append(line)
        try:
            with HUD_LOG.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        print(line)
    def lines(self) -> list:
        return list(self.q)


HUD = RingLog()
STT = LocalSTT()   # loads models lazily on first mic use

EGRESS_HOSTS = ["en.wikipedia.org", "pa.wikipedia.org", "github.com",
                "raw.githubusercontent.com", "api.github.com",
                "wttr.in", "duckduckgo.com"]


# --------------------------------------------------------------------------- #
# SevenService — orb ਪਿੱਛੇ ਅਸਲੀ ਏਜੰਟ                                            #
# --------------------------------------------------------------------------- #

class SevenService:
    def __init__(self, state_dir: Path):
        state_dir.mkdir(parents=True, exist_ok=True)
        self.egress = EgressPolicy(EGRESS_HOSTS)
        self.broker = SecretsBroker(vault={"ci_token": "DEMO-TOKEN-123"})
        self.registry = self._build_registry()
        self.audit = AuditLog(state_dir / "audit.jsonl")
        self.checkpoints = CheckpointStore(state_dir / "checkpoints.json")
        self.policy = PolicyEngine()
        self.sandbox = LocalSandbox(self.broker, simulated_latency_s=0.05)
        self.undo = UndoStack(audit=self.audit)
        self.dna = FailureDNA(state_dir / "failure_dna.json")
        self.episodes = EpisodicMemory(state_dir / "episodes.json")
        self.team = Team()
        self.prolepsis = Prolepsis(state_dir / "foresights.json")
        self.evo_log = EvolutionLog(state_dir / "evolution.json")
        self.pages = Pages(state_dir / "pages.json")

        # Local brain: an OpenAI-compatible server on loopback (mlx_lm.server)
        # if one is running, else Ollama. Both are free and offline.
        self.local_mlx = LocalOpenAICompatProvider(
            host=os.environ.get("SEVEN_LOCAL_URL", "http://127.0.0.1:8095"))
        # Optional second local model that writes Gurmukhi replies (the planner
        # model is good at JSON/tools but writes broken Gurmukhi prose).
        self.writer = LocalOpenAICompatProvider(
            host=os.environ.get("SEVEN_WRITER_URL", "http://127.0.0.1:8096"))
        self.local = OllamaProvider(model="qwen3:8b")
        self.remote = OpenAIProvider()
        self.ledger = CostLedger(daily_budget_usd=1.00)
        self.router = HybridRouter(self.local, self.remote, self.ledger)
        self.dreams = DreamEngine(self.episodes, self.dna, router=self.router)

        self.pending: Dict[str, ActionIntent] = {}
        self.approved_ids: set[str] = set()
        self._ollama_ok: Optional[bool] = None
        self._brain_name = "demo"

        async def ui_gate(intent: ActionIntent, decision) -> bool:
            return intent.action_id in self.approved_ids

        self.agent = Orchestrator(
            name="7", planner=None, registry=self.registry,
            sandbox=self.sandbox,
            approval_gate=ApprovalGate(ui_gate, timeout_s=600.0),
            audit=self.audit, checkpoints=self.checkpoints, policy=self.policy)

        for b in ("ਸੁਰੱਖਿਆ ਕੋਰ .......... OK",
                  "ਟੂਲ ਰਜਿਸਟਰੀ ......... %d tools" % len(self.registry._tools),
                  "ਬਜਟ ਲੈਜ਼ਰ ........... $%.2f/day" % self.ledger.daily_budget_usd,
                  "ਆਡਿਟ ਚੇਨ ............ valid",
                  "voice ਤਿਆਰ ਹੈ"):
            HUD.add("boot", b)

    def _build_registry(self) -> ToolRegistry:
        reg = build_full_registry(egress_hosts=EGRESS_HOSTS, state_dir=STATE)

        async def chat_reply(p):
            return {"reply": p.get("text", "")}

        async def _app(p):
            return await app_open(p)

        async def _site(p):
            return await site_open(p)

        async def _search(p):
            return await web_search_open(p)

        async def _sysmon(p):
            return await system_monitor(p)

        async def _globe(p):
            return await globe_distance(p)

        async def _circuit(p):
            return await circuit_schematic(p)

        async def _weather(p):
            return await weather_report(p, self.egress)

        async def _browse(p):
            return await browse_task(p, self.egress)

        async def _smart(p):
            return await smart_search(p, self.egress)

        async def _undo(p):
            return await self.undo.undo_last()

        def app_schema(p):
            if not isinstance(p.get("name"), str) or not p.get("name"):
                raise ValueError("app_open needs string 'name'")

        reg.register(ToolSpec("chat_reply", chat_reply, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("app_open", _app, RiskLevel.HIGH,
                              Capability.EXTERNAL_EFFECT, validate=app_schema))
        reg.register(ToolSpec("site_open", _site, RiskLevel.MEDIUM,
                              Capability.EXTERNAL_EFFECT))
        reg.register(ToolSpec("web_search", _search, RiskLevel.LOW,
                              Capability.UNTRUSTED_INPUT))
        reg.register(ToolSpec("system_monitor", _sysmon, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("globe_distance", _globe, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("circuit_schematic", _circuit, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("weather_report", _weather, RiskLevel.LOW,
                              Capability.UNTRUSTED_INPUT))
        reg.register(ToolSpec("browser_task", _browse, RiskLevel.HIGH,
                              Capability.UNTRUSTED_INPUT | Capability.EXTERNAL_EFFECT))
        reg.register(ToolSpec("smart_search", _smart, RiskLevel.LOW,
                              Capability.UNTRUSTED_INPUT))
        reg.register(ToolSpec("undo", _undo, RiskLevel.MEDIUM,
                              Capability.EXTERNAL_EFFECT))

        async def _recall(p):
            q = str(p.get("query", ""))
            hits = self.episodes.search(q) if q else self.episodes.recent(8)
            return {"matches": len(hits), "episodes": hits[-8:]}

        async def _dream(p):
            HUD.add("dream", "ਸੁਪਨੇ ਦਾ ਦੌਰ ਸ਼ੁਰੂ…")
            d = await self.dreams.dream()
            HUD.add("dream", f"{len(d['insights'])} insights")
            return d

        async def _failures(p):
            return self.dna.report()

        reg.register(ToolSpec("recall", _recall, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("dream", _dream, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("failure_report", _failures, RiskLevel.LOW,
                              Capability.NONE))

        async def _team_status(p):
            return {"agents": self.team.roster(), "count": len(self.team.roster())}

        async def _team_ask(p):
            msg = str(p.get("message", ""))
            a = self.team.assign(msg)
            HUD.add("team", f"{a['name']} ਨੂੰ ਸੌਂਪਿਆ")
            return a

        async def _code_review(p):
            code = str(p.get("code", ""))
            r = review_code(code)
            HUD.add("team", f"ਸਮੀਖਕ: {r['summary']}")
            return r

        async def _reverse(p):
            path = str(p.get("path", ""))
            if not path:
                raise ValueError("reverse_inspect needs 'path'")
            r = reverse_inspect(path)
            HUD.add("team", f"ਉਲਟ: {r['format']} ({r['size']} bytes)")
            return r

        def path_schema(p):
            if not isinstance(p.get("path"), str) or not p.get("path"):
                raise ValueError("reverse_inspect needs string 'path'")

        reg.register(ToolSpec("team_status", _team_status, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("team_ask", _team_ask, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("code_review", _code_review, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("reverse_inspect", _reverse, RiskLevel.LOW,
                              Capability.NONE, validate=path_schema))

        async def _foresee(p):
            task = str(p.get("task", ""))
            if not task:
                raise ValueError("foresee needs 'task'")
            known = list(self.registry._tools.keys())
            f = self.prolepsis.predict(task, known)
            HUD.add("fore", f"ਕਿਸਮਤ: {f.predicted_effort_min} ਮਿੰਟ (ਭਰੋਸਾ {f.confidence})")
            return {"foresight_id": f.id, "architecture": f.architecture,
                    "predicted_effort_min": f.predicted_effort_min,
                    "confidence": f.confidence, "steps": f.steps,
                    "calibration": self.prolepsis.calibration()}

        async def _evolve(p):
            # DNA ਤੋਂ ਸਬ ਤੋਂ ਵੱਖਰੀ ਅਸਫਲਤਾ → ਵਿਹਾਰ-ਸੁਧਾਰ ਪ੍ਰਸਤਾਵ
            rep = self.dna.report()
            top = rep.get("most_frequent", [])
            if not top:
                return {"note": "ਸੁਧਾਰ ਲਈ ਕੋਈ ਅਸਫਲਤਾ-ਪੈਟਰਨ ਨਹੀਂ — ਸਭ ਸੁਚਾਰੂ",
                        "evolution": self.evo_log.verified_count()}
            sig = top[0]
            prop = propose(
                title=f"ਵਿਹਾਰ-ਸੁਧਾਰ: {sig.get('category', 'unknown')} ਘਟਾਓ",
                rationale=f"DNA: {sig.get('count', 0)} ਵਾਰ {sig.get('category')} "
                          f"— {sig.get('tool', '?')} ਵਿਚ",
                patch="persona_hint: ਇਸ ਕਿਸਮ ਦੀ ਗੱਲਬੱਲ ਤੋਂ ਪਹਿਲਾਂ ਜਾਂਚ-ਪੜਤਾਲ",
                target=sig.get("tool", "team"),
                rollback_patch="persona_hint: <ਪੁਰਾਣਾ>")
            applied = apply_proposal(prop, self.evo_log,
                                     approved=bool(p.get("approved")))
            HUD.add("evo", f"ਪ੍ਰਸਤਾਵ {applied.id}: {applied.status}")
            return {"proposal": applied.id, "status": applied.status,
                    "title": applied.title,
                    "needs_approval": applied.status == "pending",
                    "verified_so_far": self.evo_log.verified_count()}

        async def _board(p):
            votes = p.get("votes")
            if isinstance(votes, dict):
                try:
                    return {"decision": board_vote(votes)}
                except ConsensusError as e:
                    return {"decision": None, "fallback": "ਘੱਟ-ਜੋਖ਼ਮ ਵਿਕਲਪ",
                            "reason": str(e)}
            # ਡਿਫਾਲਟ: ਮੌਜੂਦਾ ਟੀਮ ਤੋਂ ਨਮੂਨਾ ਵੋਟ
            roster = [a["key"] for a in self.team.roster()]
            return {"note": "votes={'ਏਜੰਟ':'ਵਿਕਲਪ'} ਦਿਓ", "roster": roster}

        reg.register(ToolSpec("foresee", _foresee, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("evolve", _evolve, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("board", _board, RiskLevel.LOW, Capability.NONE))

        async def _page_draft(p):
            d = self.pages.draft(str(p.get("title", "")), str(p.get("content", "")),
                                 p.get("page_id"))
            HUD.add("page", f"ਡਰਾਫ਼ਟ: {d['title']}")
            return {"draft_id": d["id"], "title": d["title"],
                    "status": "pending",
                    "note": "ਸਮੀਖਿਆ ਮਗਰੋਂ ਹੀ ਸੇਵ — page_approve ਨਾਲ"}
        async def _page_approve(p):
            return self.pages.approve(str(p.get("draft_id", "")))
        async def _page_reject(p):
            return self.pages.reject(str(p.get("draft_id", "")))
        async def _page_read(p):
            return self.pages.read(str(p.get("page_id", "")))
        async def _page_list(p):
            return {"pages": self.pages.list_pages(),
                    "pending_drafts": self.pages.pending_drafts()}
        async def _guard_scan(p):
            return scan_untrusted(str(p.get("text", "")))

        reg.register(ToolSpec("page_draft", _page_draft, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("page_approve", _page_approve, RiskLevel.MEDIUM,
                              Capability.EXTERNAL_EFFECT))
        reg.register(ToolSpec("page_reject", _page_reject, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("page_read", _page_read, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("page_list", _page_list, RiskLevel.LOW, Capability.NONE))
        reg.register(ToolSpec("guard_scan", _guard_scan, RiskLevel.LOW, Capability.NONE))
        return reg

    async def _ollama_available(self) -> bool:
        """True if any local brain is up. Re-checked every call (cheap ping), so
        starting the server after 7 boots works without a restart."""
        if await self.local_mlx.available():
            if self.router.local is not self.local_mlx:
                self.router.local = self.local_mlx
            self._brain_name = "MLX ਲੋਕਲ"
            self._ollama_ok = True
            return True
        ok = await self.local.available()
        if ok and self.router.local is not self.local:
            self.router.local = self.local
        self._brain_name = "Ollama ਲੋਕਲ" if ok else "demo"
        self._ollama_ok = ok
        return ok

    # -- planning ---------------------------------------------------------- #

    async def make_plan(self, message: str) -> Dict[str, Any]:
        if await self._ollama_available():
            planner = LLMPlanner(self.router, self.registry, self.broker, self.audit)
            provider = f"{self._brain_name} (ਮੁਫ਼ਤ, ਔਫਲਾਈਨ)"
            try:
                intents = await planner.plan(message, self.agent.memory.context)
            except Exception as exc:
                intents = self._fallback_plan(message)
                provider = f"demo (LLM ਫੇਲ: {exc})"
        else:
            intents = self._fallback_plan(message)
            provider = "demo-planner (ਅਸਲੀ ਦਿਮਾਗ਼ ਲਈ: mlx_lm.server --port 8095 ਜਾਂ ollama serve)"

        intents = await self._gurmukhi_reply(message, intents)
        steps = []
        for intent in intents:
            decision = self.policy.evaluate(intent)
            self.pending[intent.action_id] = intent
            steps.append({
                "action_id": intent.action_id, "tool": intent.tool,
                "risk": intent.risk.value, "payload": intent.payload,
                "policy_allowed": decision.allowed,
                "policy_reason": decision.reason,
                "needs_approval": decision.needs_approval,
                "status": "pending" if decision.allowed else "vetoed_by_policy",
            })
        HUD.add("plan", f"'{message[:42]}' → {len(intents)} ਕਦਮ")
        self.audit.record("plan_accepted", trigger=message,
                          steps=[s["action_id"] for s in steps])
        return {"provider": provider, "steps": steps}

    WRITER_SYSTEM = ("You are 7, a private assistant running locally on the user's own Mac. "
                     "Reply in clear, natural Gurmukhi Punjabi, briefly (1-3 sentences). "
                     "If you are not sure of a fact, say so instead of guessing.")

    async def _gurmukhi_reply(self, message: str, intents: List[ActionIntent]) -> List[ActionIntent]:
        """If the user wrote Gurmukhi and the plan is just a chat reply, let the
        writer model produce the text. Falls back silently to the planner's text."""
        if not (len(intents) == 1 and intents[0].tool == "chat_reply"):
            return intents
        if not any("\u0a00" <= ch <= "\u0a7f" for ch in message):
            return intents
        try:
            if not await self.writer.available():
                return intents
            resp = await self.writer.complete(self.WRITER_SYSTEM, message, max_tokens=300)
            text = resp.text.strip()
            if text:
                intents[0].payload = {"text": text}
        except Exception:
            pass
        return intents

    def _fallback_plan(self, message: str) -> List[ActionIntent]:
        low = message.lower()

        def I(tool, payload, risk, caps):
            return ActionIntent(tool=tool, payload=payload, risk=risk,
                                capabilities=caps)

        # ਐਪ/ਵੈਬਸਾਈਟ ਖੋਲ੍ਹੋ — "open X", "X ਖੋਲ੍ਹੋ", "X kholo", "X ਚਲਾਓ"
        for word in ("open ", "ਖੋਲ੍ਹੋ", "ਖੋਲੋ", "kholo", "ਚਲਾਓ", "chala"):
            if word in low:
                name = (low.replace("ਖੋਲ੍ਹੋ", "").replace("ਖੋਲੋ", "").replace("kholo", "")
                        .replace("open ", "").replace("ਚਲਾਓ", "")
                        .replace("chala", "").strip())
                if name in ("app", ""):
                    break
                if "." in name:  # domain ਲਗਦਾ → ਵੈਬਸਾਈਟ
                    return [I("site_open", {"url": name, "dry_run": False},
                              RiskLevel.MEDIUM, Capability.EXTERNAL_EFFECT)]
                if name in known_apps():
                    return [I("app_open", {"name": name, "dry_run": False},
                              RiskLevel.HIGH, Capability.EXTERNAL_EFFECT)]

        if any(k in low for k in ("ਪੇਜ ਲਿਖ", "ਨੋਟ ਸੇਵ", "draft", "ਡਰਾਫ਼ਟ",
                                  "page ")):
            # "ਪੇਜ ਲਿਖ <title> : <content>" — content ਨਾ ਹੋਵੇ ਤਾਂ ਸੂਝਾ ਹੀ
            body = message
            for kw in ("ਪੇਜ ਲਿਖ", "ਨੋਟ ਸੇਵ", "draft", "ਡਰਾਫ਼ਟ", "page ", "ਕਰੋ", "ਬਣਾ"):
                body = body.replace(kw, "")
            title, _, content = body.partition(":")
            return [I("page_draft", {"title": title.strip() or "ਨੋਟ",
                                     "content": content.strip() or message},
                      RiskLevel.LOW, Capability.NONE)]
        if any(k in low for k in ("ਪੇਜ ਸੂਚੀ", "page list", "ਮੇਰੇ ਪੇਜ",
                                  "drafts", "ਡਰਾਫ਼ਟਾਂ")):
            return [I("page_list", {}, RiskLevel.LOW, Capability.NONE)]
        if any(k in low for k in ("foresee", "ਸੋਚ ਪਹਿਲਾਂ", "ਭਵਿੱਖੀ", "ਪ੍ਰੀਡਿਕਸ਼ਨ",
                                  "prediction", "ਪਲਾਨ ਬਣਾ")):
            task = message
            for kw in ("foresee", "ਸੋਚ ਪਹਿਲਾਂ", "ਭਵਿੱਖੀ", "ਪ੍ਰੀਡਿਕਸ਼ਨ",
                       "prediction", "ਪਲਾਨ ਬਣਾ", "ਕਰੋ"):
                task = task.replace(kw, "")
            return [I("foresee", {"task": task.strip() or message},
                      RiskLevel.LOW, Capability.NONE)]
        if any(k in low for k in ("ਸਰਚ", "search", "ਲੱਭੋ", "find")):
            q = (low.replace("ਸਰਚ", "").replace("search", "")
                 .replace("ਲੱਭੋ", "").replace("find", "").strip(": "))
            return [I("web_search", {"query": q or message, "dry_run": False},
                      RiskLevel.LOW, Capability.UNTRUSTED_INPUT)]
        if any(k in low for k in ("ਯਾਦ", "recall", "ਕੀ ਕੀਤਾ", "history")):
            q = (low.replace("ਯਾਦ", "").replace("recall", "")
                 .replace("ਕੀ ਕੀਤਾ", "").replace("history", "").strip())
            return [I("recall", {"query": q}, RiskLevel.LOW, Capability.NONE)]
        if any(k in low for k in ("ਸੁਪਨਾ", "dream", "ਸਿੱਖ", "ਸੁਝਾਅ", "improve")):
            return [I("dream", {}, RiskLevel.LOW, Capability.NONE)]
        if any(k in low for k in ("ਫੇਲ", "failures", "ਗਲਤੀਆਂ", "mistakes")):
            return [I("failure_report", {}, RiskLevel.LOW, Capability.NONE)]
        if any(k in low for k in ("reverse", "ਰਿਵਰਸ", "ਉਲਟ-ਇੰਜੀਨ", "ਉਲਟ ਇੰਜੀਨ",
                                  "binary", "ਬਾਈਨਰੀ", "apk")):
            words = [w for w in message.split() if "/" in w or "." in w.split("/")[-1]]
            path = words[-1] if words else ""
            return [I("reverse_inspect", {"path": path}, RiskLevel.LOW,
                      Capability.NONE)]
        if any(k in low for k in ("ਸੋਚ ਪਹਿਲਾਂ", "foresee", "ਭਵਿੱਖ", "ਪ੍ਰੀਡਿਕਸ਼ਨ",
                                  "prediction", "ਪਲਾਨ ਬਣਾ", "ਅਨੁਮਾਨ")):
            task = message
            for kw in ("ਸੋਚ ਪਹਿਲਾਂ", "foresee", "ਭਵਿੱਖ", "ਪ੍ਰੀਡਿਕਸ਼ਨ",
                       "prediction", "ਪਲਾਨ ਬਣਾ", "ਅਨੁਮਾਨ", "ਕਰੋ", "ਦੇ"):
                task = task.replace(kw, "")
            return [I("foresee", {"task": task.strip() or message},
                      RiskLevel.LOW, Capability.NONE)]
        if any(k in low for k in ("ਸੁਧਾਰ ਕਰ", "ਸਵੈ-ਸੁਧਾਰ", "evolve", "rsi",
                                  "ਆਪਣੇ ਆਪ ਸੁਧਾਰ", "ਸੁਧਰ")):
            return [I("evolve", {"approved": False}, RiskLevel.LOW,
                      Capability.NONE)]
        if any(k in low for k in ("ਟੀਮ", "team", "ਏਜੰਟ ਕੌਣ", "roster", "ਮਾਹਿਰ")):
            return [I("team_status", {}, RiskLevel.LOW, Capability.NONE)]

        if any(k in low for k in ("review", "ਰਿਵਿਊ", "ਸਮੀਖਿਆ")):
            # ਸੁਨੇਹੇ ਵਿਚ ਹੀ ਕੋਡ ਪੇਸਟ ਕੀਤਾ ਹੋਵੇ ਤਾਂ ਸਿੱਧੀ ਸਮੀਖਿਆ
            code = message
            for kw in ("review", "ਰਿਵਿਊ", "ਸਮੀਖਿਆ", "ਕਰੋ", "ਕਰ ਦੇ"):
                code = code.replace(kw, "")
            if any(t in code for t in ("def ", "import ", "class ", "function")):
                return [I("code_review", {"code": code.strip()}, RiskLevel.LOW,
                          Capability.NONE)]
            return [I("team_ask", {"message": message}, RiskLevel.LOW,
                      Capability.NONE)]

        if any(k in low for k in ("weather", "ਮੌਸਮ", "mausam")):
            city = (low.replace("weather", "").replace("ਮੌਸਮ", "")
                    .replace("mausam", "").replace(" ਦਾ", "").strip() or "Chandigarh")
            return [I("weather_report", {"city": city.title()}, RiskLevel.LOW,
                      Capability.UNTRUSTED_INPUT)]
        if any(k in low for k in ("system", "ਸਿਸਟਮ", "cpu", "ram")):
            return [I("system_monitor", {}, RiskLevel.LOW, Capability.NONE)]
        if "distance" in low or "ਦੂਰੀ" in low:
            words = [w.strip("?.,!") for w in low.split()]
            cities = [w for w in words if w in CITIES]
            if len(cities) >= 2:
                return [I("globe_distance", {"from_city": cities[0],
                                             "to_city": cities[1]},
                          RiskLevel.LOW, Capability.NONE)]
            return [I("chat_reply", {"text": "ਦੋ ਸ਼ਹਿਰ ਦੱਸੋ, ਜਿਵੇਂ: "
                      "'distance amritsar to london'"}, RiskLevel.LOW,
                      Capability.NONE)]
        if any(k in low for k in ("circuit", "ਸਰਕਟ", "wiring", "ਵਾਇਰਿੰਗ", "schematic")):
            sensor = ("dht11" if "dht" in low else
                      "soil" if "soil" in low else
                      "hcsr04" if ("ultra" in low or "hc" in low) else "led")
            board = "esp32" if "esp32" in low else "arduino"
            return [I("circuit_schematic", {"sensor": sensor, "board": board},
                      RiskLevel.LOW, Capability.NONE)]
        if any(k in low for k in ("undo", "ਵਾਪਸ", "ulta")):
            return [I("undo", {}, RiskLevel.MEDIUM, Capability.EXTERNAL_EFFECT)]
        if any(k in low for k in ("organize", "ਵੰਡ", "sort")):
            return [I("file_organizer", {"folder": str(Path.home() / "Downloads"),
                                         "dry_run": True},
                      RiskLevel.MEDIUM, Capability.SENSITIVE_ACCESS)]
        if any(k in low for k in ("arduino", "ਆਰਡਿਨੋ", "blink")):
            return [I("arduino_generate", {"led_pin": 13, "interval_ms": 1000},
                      RiskLevel.LOW, Capability.NONE)]
        return [I("chat_reply", {"text": (
            "ਮੈਂ ਸੁਣਿਆ: “" + message + "”\n"
            "ਕੋਸ਼ਿਸ਼ ਕਰੋ: open chrome · youtube.com ਖੋਲ੍ਹੋ · search ਪੰਜਾਬ · "
            "ਮੌਸਮ ਲੁਧਿਆਣਾ · system · distance amritsar to london · "
            "circuit dht11 esp32 · organize files · undo")},
            RiskLevel.LOW, Capability.NONE)]

    # -- execution ---------------------------------------------------------- #

    async def execute_step(self, action_id: str, approved: bool) -> Dict[str, Any]:
        intent = self.pending.pop(action_id, None)
        if intent is None:
            return {"error": "unknown or already-handled action"}
        if approved:
            self.approved_ids.add(action_id)
        HUD.add("exec", f"{intent.tool} ({intent.risk.value}) approved={approved}")
        result = await self.agent._execute_step(intent)
        HUD.add("exec", f"{intent.tool} → {result.status.value}")
        self.checkpoints.save_result(result)
        self.episodes.record(
            f"{intent.tool} — {result.status.value}",
            f"payload={json.dumps(intent.payload, ensure_ascii=False)[:200]} "
            f"error={result.error or '—'}",
            tags=["exec", intent.tool,
                  "veto" if "veto" in result.status.value else result.status.value])
        if result.status.value == "failed" and result.error:
            learned = self.dna.learn(intent.tool, result.error)
            prior_fix = self.dna.known_fix(intent.tool, result.error)
            recipe = self.dna.recipe(intent.tool, result.error)
            HUD.add("dna", f"{intent.tool}: {learned['category']} "
                           f"(#{learned['count']}) recipe={recipe[0]['action']}")
            result.output = {"failure_dna": {
                "category": learned["category"],
                "times_seen": learned["count"],
                "known_fix": prior_fix,
                "recipe": recipe}}
        elif result.status.value == "success" and intent.payload.get("_retried_from"):
            self.dna.mark_fixed(intent.tool,
                                intent.payload["_retried_from"], "retry")
        # guard7: ਬਾਹਰੀ-ਸਮਗਰੀ ਟੂਲਾਂ ਦਾ ਨਤੀਜਾ ਆਪੇ ਸਕੈਨ (injection ਝੰਡਾ)
        if result.status.value == "success" and isinstance(result.output, dict):
            spec = self.registry._tools.get(intent.tool)
            if spec and (spec.capabilities & Capability.UNTRUSTED_INPUT):
                result.output = wrap_tool_output(result.output)
                if result.output.get("injection_guard"):
                    HUD.add("guard", f"{intent.tool}: injection-ਪੈਟਰਨ ਮਿਲਿਆ!")
        return {"action_id": action_id, "status": result.status.value,
                "output": result.output, "error": result.error,
                "attempts": result.attempts,
                "latency_s": round(result.latency_s, 3)}

    def status(self) -> Dict[str, Any]:
        return {"tools": sorted(self.registry._tools.keys()),
                "spent_today_usd": round(self.ledger.spent_today, 6),
                "daily_budget_usd": self.ledger.daily_budget_usd,
                "ollama": self._ollama_ok,
                "brain": self._brain_name,
                "playwright": playwright_available(),
                "undo_pending": self.undo.pending(),
                "episodes": len(self.episodes),
                "failure_signatures": self.dna.report()["total_signatures"],
                "agents": len(self.team.roster()),
                "pages": self.pages.stats(),
                "audit_entries": len(self.audit),
                "audit_chain_valid": self.audit.verify()}

    def hud_lines(self) -> Dict[str, Any]:
        return {"lines": HUD.lines()[-160:]}

    def audit_tail(self, n: int = 12) -> Dict[str, Any]:
        return {"entries": self.audit._entries[-n:],
                "chain_valid": self.audit.verify()}


SERVICE: Optional[SevenService] = None


def run_async(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# ਮੁੱਖ ਫੇਸ — voice-first orb                                                    #
# --------------------------------------------------------------------------- #

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="pa">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>7 — ਤੁਹਾਡਾ ਏਜੰਟ</title>
<style>
  :root { --deep:#0a0e1a; --mid:#101424; --surface:#151a2e; --card:#1c2238;
          --gold:#f0b429; --gold2:#ffd166; --cyan:#4ecdc4; --violet:#a78bfa;
          --green:#51cf66; --red:#ff6b6b; --orange:#fd7e14;
          --text:#e8eaf0; --muted:#7a80a0; --border:rgba(255,255,255,.08); }
  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; background:var(--deep); color:var(--text);
              font-family:'Outfit','Noto Sans Gurmukhi',system-ui,sans-serif; overflow:hidden; }

  /* ਤਾਰੇ */
  .stars { position:fixed; inset:0; pointer-events:none; z-index:0; }
  .star { position:absolute; width:2px; height:2px; background:#fff; border-radius:50%;
          animation:twinkle var(--d,3s) ease-in-out infinite var(--dl,0s); }
  @keyframes twinkle { 0%,100%{opacity:.08} 50%{opacity:.7} }

  .shell { position:relative; z-index:1; display:flex; flex-direction:column; height:100vh; }

  /* header — gold ring avatar */
  .hdr { display:flex; align-items:center; justify-content:space-between; padding:12px 22px;
         border-bottom:.5px solid var(--border); background:rgba(10,14,26,.85);
         backdrop-filter:blur(12px); }
  .hdr-left { display:flex; align-items:center; gap:12px; }
  .avatar-ring { position:relative; width:46px; height:46px; }
  .avatar-ring svg { width:100%; height:100%; }
  @keyframes spin { to { transform:rotate(360deg) } }
  .ring-anim { animation:spin 9s linear infinite; transform-origin:23px 23px; }
  .avatar-face { position:absolute; inset:8px; border-radius:50%;
                 background:radial-gradient(circle at 40% 35%, #ffe9b0, #f0b429 55%, #8a6116);
                 display:flex; align-items:center; justify-content:center;
                 font-size:15px; font-weight:800; color:#0a0e1a; }
  .brand-name { font-size:19px; font-weight:700; letter-spacing:.5px;
                background:linear-gradient(90deg,var(--gold),var(--gold2));
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .brand-sub { font-size:10.5px; color:var(--muted); letter-spacing:1.6px; text-transform:uppercase; }
  .pill { display:flex; align-items:center; gap:6px; border-radius:20px; padding:4px 12px;
          font-size:12px; border:.5px solid; }
  .pill.ok { color:var(--green); border-color:rgba(81,207,102,.35); background:rgba(81,207,102,.08); }
  .dot { width:6px; height:6px; border-radius:50%; background:var(--green);
         animation:pg 1.6s ease-in-out infinite; }
  @keyframes pg { 50% { box-shadow:0 0 0 6px rgba(81,207,102,0) } }
  .hdr-right { display:flex; gap:8px; }
  .btn-sm { background:var(--card); border:.5px solid var(--border); border-radius:8px;
            color:var(--muted); font-size:12px; padding:6px 13px; cursor:pointer; transition:.2s; }
  .btn-sm:hover { border-color:var(--gold); color:var(--gold); }

  /* 3-ਪੈਨਲ HUD */
  .main { display:grid; grid-template-columns:1fr 285px; flex:1; min-height:0; }
  .center { display:flex; flex-direction:column; min-width:0; border-right:.5px solid var(--border); }
  .side { background:var(--mid); display:flex; flex-direction:column; min-height:0; }

  /* orb — ਤਾਰੇ + HUD */
  #stage { padding:16px 0 4px; display:flex; flex-direction:column; align-items:center; gap:5px; }
  #orb { width:220px; height:220px; cursor:pointer; display:block; }
  #orbstate { font-size:12px; color:var(--gold); min-height:16px; }

  #chat { flex:1; overflow-y:auto; padding:10px 22px; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:76%; padding:9px 14px; border-radius:14px; white-space:pre-wrap; word-break:break-word; }
  .me  { align-self:flex-end; background:#233a1a; border:1px solid rgba(81,207,102,.25);
         border-bottom-right-radius:4px; }
  .bot { align-self:flex-start; background:var(--surface); border:1px solid var(--border);
         border-bottom-left-radius:4px; }
  .meta { font-size:11px; color:var(--muted); margin-top:5px; }
  .step { border:1px solid var(--border); border-left:3px solid var(--cyan); border-radius:10px;
          padding:9px 12px; margin-top:8px; background:#0d1226; font-size:13.5px; }
  .step.critical { border-left-color:var(--red); }
  .step.high { border-left-color:var(--orange); }
  .step .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .tag { font-size:11px; padding:1px 8px; border-radius:99px; border:1px solid var(--border); color:var(--muted); }
  .tag.ok { color:var(--green); border-color:var(--green); }
  .tag.bad { color:var(--red); border-color:var(--red); }
  .tag.warn { color:var(--orange); border-color:var(--orange); }
  button { cursor:pointer; border:none; border-radius:9px; padding:7px 14px; font-size:13.5px; font-weight:600; }
  .yes { background:var(--green); color:#04170b; }
  .no  { background:#2a3350; color:var(--text); }
  .out { margin-top:8px; font-family:ui-monospace,monospace; font-size:12.5px; color:#b8f5d0;
         background:#071009; border:1px solid #14532d; border-radius:8px; padding:8px 10px;
         white-space:pre-wrap; max-height:220px; overflow-y:auto; }
  .err { color:#ffc9c9; background:#1a0b0b; border-color:#7f1d1d; }
  .reply { margin-top:6px; font-size:16px; line-height:1.55; white-space:pre-wrap; color:var(--text); }
  .reply.err { color:#ffc9c9; background:none; }
  details.raw { margin-top:6px; } details.raw summary { cursor:pointer; color:#7a80a0; font-size:12px; }
  code { font-family:ui-monospace,monospace; font-size:12px; color:#cbd5e1; }

  /* command bar */
  #cmdbar { display:flex; gap:8px; padding:11px 18px; align-items:center;
            border-top:.5px solid var(--border); background:rgba(16,20,36,.92); }
  #mic { background:#b3223e; color:#fff; font-size:17px; padding:8px 13px; }
  #lang { background:#121a2e; color:var(--text); border:1px solid #2a3350; border-radius:8px; padding:6px 8px; font-size:14px; }
  #mic.live { background:var(--green); animation:pulseG 1s infinite; }
  #spk { background:#2a3350; color:var(--text); padding:8px 11px; }
  #in { flex:1; background:#0b0f1e; color:var(--text); border:1px solid var(--border);
        border-radius:10px; padding:11px 14px; font-size:15px; outline:none; }
  #in:focus { border-color:var(--gold); }
  #send { background:linear-gradient(135deg,#c89b20,#f0b429); color:#0a0e1a; font-weight:700; }
  #status { display:flex; gap:16px; padding:4px 22px; font-size:11.5px; color:var(--muted); }
  #status b { color:var(--text); }

  /* ਸਜਸ਼ਨ ਚਿੱਪਸ */
  #chips { display:flex; gap:7px; flex-wrap:wrap; padding:0 22px 8px; }
  .chip { font-size:12px; color:var(--cyan); border:.5px solid rgba(78,205,196,.3);
          background:rgba(78,205,196,.06); border-radius:20px; padding:4px 12px; cursor:pointer; }
  .chip:hover { background:rgba(78,205,196,.15); }

  /* ਸਾਈਡ ਪੈਨਲ — HUD log */
  .side-hdr { padding:14px 16px 8px; font-size:11px; letter-spacing:1.5px; color:var(--muted);
              text-transform:uppercase; border-bottom:.5px solid var(--border); }
  #hud { flex:1; overflow-y:auto; padding:10px 12px; font-family:ui-monospace,monospace;
         font-size:11.5px; line-height:1.7; color:#9db2ce; }
  #hud .boot { color:var(--gold2); }
  #hud .plan { color:var(--cyan); }
  #hud .exec { color:var(--green); }
  #hud .err  { color:var(--red); }
  .side-ftr { padding:10px 14px; border-top:.5px solid var(--border); font-size:11px; color:var(--muted); }

  /* boot overlay */
  #boot { position:fixed; inset:0; z-index:50; background:var(--deep); display:flex;
          flex-direction:column; align-items:center; justify-content:center; gap:18px;
          transition:opacity .6s; }
  #boot.gone { opacity:0; pointer-events:none; }
  #boot .b-title { font-size:30px; font-weight:800; letter-spacing:3px;
                   background:linear-gradient(90deg,var(--gold),var(--gold2),var(--cyan));
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  #bootlog { font-family:ui-monospace,monospace; font-size:13px; color:var(--muted);
             min-height:120px; text-align:center; line-height:2; }
  #bootlog .ok { color:var(--green); }
</style>
</head>
<body>
<div class="stars" id="stars"></div>

<div id="boot">
  <div class="b-title">7</div>
  <div id="bootlog"></div>
</div>

<div class="shell">
  <div class="hdr">
    <div class="hdr-left">
      <div class="avatar-ring">
        <svg viewBox="0 0 46 46"><g class="ring-anim">
          <circle cx="23" cy="23" r="21" fill="none" stroke="#f0b429" stroke-width="1.4"
                  stroke-dasharray="22 8" opacity=".9"/>
        </g></svg>
        <div class="avatar-face">7</div>
      </div>
      <div><div class="brand-name">7</div>
      <div class="brand-sub">Voice · Local · Sovereign</div></div>
    </div>
    <div class="hdr-right">
      <div class="pill ok"><span class="dot"></span> Live</div>
      <button class="btn-sm" id="forge">+ Skill</button>
      <button class="btn-sm" id="undobtn">↩ Undo</button>
      <button class="btn-sm" id="audit">Audit</button>
    </div>
  </div>

  <div class="main">
    <div class="center">
      <div id="stage">
        <canvas id="orb" width="440" height="440" title="ਬੋਲਣ ਲਈ ਦਬਾਓ"></canvas>
        <div id="orbstate"></div>
      </div>
      <div id="chat"></div>
      <div id="chips"></div>
      <div id="cmdbar">
        <button id="mic" title="ਬੋਲੋ (ਦਬਾਓ = ਸ਼ੁਰੂ, ਫੇਰ ਦਬਾਓ = ਬੰਦ)">🎙️</button>
        <select id="lang" title="ਕਿਹੜੀ ਬੋਲੀ ਬੋਲ ਰਹੇ ਹੋ"><option value="pa">ਪੰਜਾਬੀ</option><option value="en">English</option></select>
        <button id="spk" title="ਬੋਲਣਾ on/off">🔊</button>
        <input id="in" placeholder="ਬੋਲੋ ਜਾਂ ਲਿਖੋ…" autofocus>
        <button id="send">ਭੇਜੋ ➤</button>
      </div>
      <div id="status">
        <span>ਦਿਮਾਗ਼: <b id="brain">…</b></span><span>ਖਰਚ: <b id="spent">$0</b></span>
        <span>tools: <b id="tools">…</b></span><span>audit: <b id="auditok">…</b></span>
        <span>undo: <b id="undon">0</b></span><span>auto: <b id="pw">…</b></span>
      </div>
    </div>
    <div class="side">
      <div class="side-hdr">HUD — ਸਿਸਟਮ ਲੌਗ</div>
      <div id="hud"></div>
      <div class="side-ftr">ਹਰ ਫੈਸਲਾ ਇੱਥੇ + tamper-evident audit ਵਿੱਚ</div>
    </div>
  </div>
</div>

<script>
/* ਤਾਰੇ */
const stars = document.getElementById('stars');
for (let i = 0; i < 90; i++) {
  const s = document.createElement('div'); s.className = 'star';
  s.style.left = Math.random()*100+'%'; s.style.top = Math.random()*100+'%';
  s.style.setProperty('--d', (2+Math.random()*4)+'s');
  s.style.setProperty('--dl', (Math.random()*4)+'s');
  stars.appendChild(s);
}

const chat = document.getElementById('chat');
const inp = document.getElementById('in');
const orb = document.getElementById('orb');
const orbstate = document.getElementById('orbstate');
const hud = document.getElementById('hud');

const esc = s => String(s).replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));

/* ਬੂਟ ਸੀਕਵੈਂਸ */
const bootEl = document.getElementById('boot');
const bootlog = document.getElementById('bootlog');
const bootLines = ['ਸੁਰੱਖਿਆ ਕੋਰ ਚਾਲੂ …','ਟੂਲ ਰਜਿਸਟਰੀ ਲੋਡ …','ਵੌਇਸ ਇੰਜਨ ਤਿਆਰ …','7 ਲਾਈਵ'];
let bi = 0;
(function bootTick(){
  if (bi < bootLines.length) {
    const d = document.createElement('div');
    d.innerHTML = esc(bootLines[bi]) + ' <span class="ok">✔</span>';
    bootlog.appendChild(d); bi++;
    setTimeout(bootTick, 260);
  } else setTimeout(() => bootEl.classList.add('gone'), 420);
})();

function add(role, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + role; d.innerHTML = html;
  chat.appendChild(d); chat.scrollTop = chat.scrollHeight; return d;
}
async function api(path, body) {
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                               body: JSON.stringify(body||{})});
  return r.json();
}

async function refreshStatus() {
  const s = await (await fetch('/api/status')).json();
  document.getElementById('brain').textContent = s.brain || (s.ollama ? 'ਲੋਕਲ' : 'demo/API');
  document.getElementById('spent').textContent = '$' + s.spent_today_usd;
  document.getElementById('tools').textContent = s.tools.length;
  document.getElementById('auditok').textContent = (s.audit_chain_valid?'✔':'✘')+' '+s.audit_entries;
  document.getElementById('undon').textContent = (s.undo_pending||[]).length;
  document.getElementById('pw').textContent = s.playwright ? '✔ playwright' : 'basic';
}

let hudSeen = 0;
async function refreshHud() {
  const h = await (await fetch('/api/hud')).json();
  const lines = h.lines || [];
  if (lines.length === hudSeen) return;
  hud.innerHTML = lines.map(l => {
    const cls = l.includes('BOOT') ? 'boot' : l.includes('PLAN') ? 'plan'
              : l.includes('EXEC') ? 'exec' : '';
    return `<div class="${cls}">${esc(l)}</div>`;
  }).join('');
  hudSeen = lines.length;
  hud.scrollTop = hud.scrollHeight;
}
setInterval(refreshHud, 1500);

function stepCard(st) {
  const card = document.createElement('div');
  card.className = 'step ' + st.risk;
  card.innerHTML = `<div class="row"><b>${esc(st.tool)}</b>
      <span class="tag ${st.risk==='critical'?'bad':st.risk==='high'?'warn':''}">${st.risk}</span>
      ${st.policy_allowed?'':'<span class="tag bad">policy veto</span>'}
      <span class="tag">${esc(st.action_id)}</span></div>
    <div style="margin-top:5px;color:#7a80a0"><code>${esc(JSON.stringify(st.payload))}</code></div>`;
  if (!st.policy_allowed) {
    card.insertAdjacentHTML('beforeend',
      `<div class="meta" style="color:#ff6b6b">⛔ ${esc(st.policy_reason)}</div>`);
    return card;
  }
  const row = document.createElement('div'); row.className='row'; row.style.marginTop='8px';
  if (st.needs_approval) {
    row.innerHTML = `<span class="meta">ਮਨਜ਼ੂਰੀ ਚਾਹੀਦੀ ਹੈ:</span>`;
    const yes=document.createElement('button'); yes.className='yes'; yes.textContent='✔ ਮਨਜ਼ੂਰ';
    const no =document.createElement('button'); no.className='no';  no.textContent='✘ ਰੱਦ';
    yes.onclick=()=>execute(st.action_id,true,card,row);
    no.onclick =()=>execute(st.action_id,false,card,row);
    row.append(yes,no);
  } else {
    const go=document.createElement('button'); go.className='yes'; go.textContent='▶ ਚਲਾਓ';
    go.onclick=()=>execute(st.action_id,true,card,row); row.append(go);
  }
  card.appendChild(row); return card;
}

async function execute(id, approved, card, row) {
  row.innerHTML = '<span class="meta">ਚੱਲ ਰਿਹਾ…</span>';
  const res = await api('/api/execute', {action_id:id, approved});
  row.innerHTML = `<span class="tag ${res.status==='success'?'ok':'bad'}">${res.status}</span>
                   <span class="meta">${res.latency_s ?? ''}s</span>`;
  if (res.output || res.error) {
    const sum = document.createElement('div'); sum.className = 'reply' + (res.error ? ' err' : '');
    sum.textContent = res.error ? String(res.error) : summarize(res.output);
    card.appendChild(sum);
    const d = document.createElement('details'); d.className = 'raw';
    d.innerHTML = '<summary>ਵੇਰਵਾ (JSON)</summary>';
    const o = document.createElement('div'); o.className = 'out';
    o.textContent = JSON.stringify(res.output || res.error, null, 1);
    d.appendChild(o); card.appendChild(d);
    if (!res.error && res.output) speak(sum.textContent);
  }
  refreshStatus(); refreshHud();
}

function summarize(out) {
  if (typeof out !== 'object') return String(out);
  if (out.reply) return out.reply;
  if (out.opened_url) return 'ਵੈਬਸਾਈਟ ਖੋਲ੍ਹ ਦਿੱਤੀ ਜੀ';
  if (out.opened) return out.opened + ' ਖੋਲ੍ਹ ਦਿੱਤਾ';
  if (out.searched) return 'ਸਰਚ ਖੋਲ੍ਹ ਦਿੱਤੀ';
  if (out.temp_c) return 'ਤਾਪਮਾਨ ' + out.temp_c + ' ਡਿਗਰੀ, ' + (out.desc || '');
  if (out.cpu_pct !== undefined) return 'CPU ' + out.cpu_pct + ' ਫੀਸਦੀ, RAM ' + (out.ram_used_pct ?? '?') + ' ਫੀਸਦੀ';
  if (out.distance_km) return out.from + ' ਤੋਂ ' + out.to + ' ' + out.distance_km + ' ਕਿਲੋਮੀਟਰ';
  if (out.insights) return 'ਸੁਪਨੇ ਤੋਂ ਸਿੱਖ: ' + out.insights[0];
  if (out.episodes) return out.matches + ' ਘਟਨਾਵਾਂ ਮਿਲੀਆਂ';
  if (out.total_signatures !== undefined) return out.total_signatures + ' ਫੇਲ-ਸਿੱਖ ਯਾਦ ਹਨ';
  if (out.error) return out.error;
  return 'ਹੋ ਗਿਆ ਜੀ';
}

async function send() {
  const text = inp.value.trim(); if (!text) return;
  inp.value = '';
  add('me', esc(text));
  const bubble = add('bot', '<span class="meta">ਸੋਚ ਰਿਹਾ…</span>');
  setMood(/ਸੁਪਨਾ|dream/.test(text) ? 'dreaming' : 'thinking');
  const plan = await api('/api/chat', {message: text});
  bubble.innerHTML = `<div class="meta">planner: ${esc(plan.provider)}</div>`;
  if (plan.error) { bubble.innerHTML += `<div class="out err">${esc(plan.error)}</div>`; setMood('error', 2200); return; }
  if (!plan.steps.length) { bubble.innerHTML += '<div class="meta">ਕੋਈ ਕਦਮ ਨਹੀਂ</div>'; setMood('idle'); return; }
  setMood('executing');
  setTimeout(() => setMood('success', 1600), 700);
  plan.steps.forEach(st => {
    // Plain conversation: show the answer as text, no card / button / raw JSON
    if (st.tool === 'chat_reply' && st.policy_allowed) {
      const t = document.createElement('div'); t.className = 'reply';
      t.textContent = (st.payload && st.payload.text) || '';
      bubble.appendChild(t);
      speak(t.textContent);
      api('/api/execute', {action_id: st.action_id, approved: false});  // keep the audit trail
      return;
    }
    const card = stepCard(st);
    bubble.appendChild(card);
    // Read-only LOW-risk tools run straight away; anything else waits for a click
    if (st.policy_allowed && !st.needs_approval && st.risk === 'low') {
      const row = card.querySelector('.row:last-child');
      execute(st.action_id, false, card, row);
    }
  });
  chat.scrollTop = chat.scrollHeight; refreshHud();
}

/* voice */
let speakOn = true;
document.getElementById('spk').onclick = e => { speakOn = !speakOn; e.target.style.opacity = speakOn?'1':'.4'; };

// ======================= QUANTUM ORB (7) =======================
// ਮੂਡ ਨਾਲ ਰੰਗ, ਆਕਾਰ, ਗਤੀ ਤੇ pulse ਬਦਲਦਾ ਘੁੰਮਦਾ orb — pure canvas, ਬਿਨਾਂ ਲਾਇਬ੍ਰੇਰੀ।
const orbC = document.getElementById('orb');
const octx = orbC.getContext('2d');
const OC = 220; // ਕੇਂਦਰ (440/2)

const MOODS = {
  idle:      { hot:'#fff6dd', mid:'#f0b429', dim:'#6b531a', speed:0.5, pulse:1.0, scale:1.00, lobes:4, label:'' },
  listening: { hot:'#d2fff4', mid:'#4ecdc4', dim:'#0f5c56', speed:1.4, pulse:1.5, scale:1.06, lobes:5, label:'ਸੁਣ ਰਿਹਾ…' },
  thinking:  { hot:'#ffe3b3', mid:'#ff9100', dim:'#8a4b00', speed:2.6, pulse:1.8, scale:1.03, lobes:7, label:'ਸੋਚ ਰਿਹਾ…' },
  executing: { hot:'#ecd6ff', mid:'#9b5cff', dim:'#3d1a78', speed:2.2, pulse:2.0, scale:1.08, lobes:6, label:'ਕਰ ਰਿਹਾ…' },
  speaking:  { hot:'#ffffff', mid:'#bfefff', dim:'#2e8ea6', speed:1.8, pulse:2.3, scale:1.10, lobes:5, label:'ਬੋਲ ਰਿਹਾ…' },
  dreaming:  { hot:'#e2d8ff', mid:'#7c6cff', dim:'#2a2350', speed:0.8, pulse:1.2, scale:0.98, lobes:8, label:'ਸੁਪਨਾ ਵੇਖ ਰਿਹਾ…' },
  success:   { hot:'#e2ffe9', mid:'#51cf66', dim:'#1b5c2e', speed:1.1, pulse:2.0, scale:1.07, lobes:4, label:'ਹੋ ਗਿਆ ✓' },
  error:     { hot:'#ffdcdc', mid:'#ff3b30', dim:'#5c0f0f', speed:1.3, pulse:2.6, scale:1.02, lobes:3, label:'ਰੁਕਾਵਟ ✖' },
};

// ਰੰਗ-lerp (hex → rgb → mix)
function hx(h){ h=h.replace('#',''); return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)]; }
function mixc(a,b,t){ return a.map((v,i)=> v+(b[i]-v)*t ); }
function css(c,a=1){ return `rgba(${c[0]|0},${c[1]|0},${c[2]|0},${a})`; }

let orbMood = 'idle';
const cur = { hot:hx(MOODS.idle.hot), mid:hx(MOODS.idle.mid), dim:hx(MOODS.idle.dim),
              speed:0.5, pulse:1.0, scale:1.0, lobes:4 };
let orbT = 0, gazeX = 0, gazeY = 0;

// ਕਰਸਰ-ਨਜ਼ਰ: orb ਹੌਲੀ ਨਜ਼ਰ ਨਾਲ ਝੁਕਦਾ
window.addEventListener('pointermove', e => {
  gazeX = ((e.clientX / Math.max(1, innerWidth)) * 2 - 1) * 8;
  gazeY = ((e.clientY / Math.max(1, innerHeight)) * 2 - 1) * 8;
});

// ਘੁੰਮਦੇ ਕਣ (particles) — ਹਰ ਇੱਕ ਦਾ ਆਪਣਾ ਘੇਰਾ ਤੇ ਗਤੀ
const ORB_DOTS = Array.from({length: 70}, (_, i) => ({
  r: 92 + Math.random() * 78, a: Math.random() * Math.PI * 2,
  v: (0.0018 + Math.random() * 0.004) * (Math.random() < .5 ? -1 : 1),
  s: 0.8 + Math.random() * 1.8, tw: Math.random() * Math.PI * 2,
}));

let moodTimer = null;
function setMood(m, holdMs = 0) {
  if (!MOODS[m]) m = 'idle';
  orbMood = m;
  orbstate.textContent = MOODS[m].label;
  if (moodTimer) { clearTimeout(moodTimer); moodTimer = null; }
  if (holdMs > 0) moodTimer = setTimeout(() => setMood('idle'), holdMs);
}

function drawOrb() {
  const tgt = MOODS[orbMood];
  // ਹੌਲੀ-ਹੌਲੀ ਬਦਲਾਅ — ਹਰ ਪੈਰਾਮੀਟਰ ਟਾਰਗਿਟ ਵੱਲ
  cur.hot = mixc(cur.hot, hx(tgt.hot), 0.06);
  cur.mid = mixc(cur.mid, hx(tgt.mid), 0.06);
  cur.dim = mixc(cur.dim, hx(tgt.dim), 0.06);
  cur.speed += (tgt.speed - cur.speed) * 0.06;
  cur.pulse += (tgt.pulse - cur.pulse) * 0.06;
  cur.scale += (tgt.scale - cur.scale) * 0.06;
  cur.lobes += (tgt.lobes - cur.lobes) * 0.06;

  orbT += cur.speed * 0.016;
  const beat = 1 + Math.sin(orbT * 3.2) * 0.05 * cur.pulse;

  octx.clearRect(0, 0, 440, 440);
  octx.save();
  octx.translate(OC + gazeX * 0.4, OC + gazeY * 0.4);
  octx.scale(cur.scale, cur.scale);

  // — ਬਾਹਰੀ halo (bloom ਦੀ ਬਜਾਏ ਡਬਲ ਗਲੋ) —
  const halo = octx.createRadialGradient(0, 0, 10, 0, 0, 165);
  halo.addColorStop(0, css(cur.mid, 0.30 * beat));
  halo.addColorStop(0.55, css(cur.dim, 0.12));
  halo.addColorStop(1, 'rgba(0,0,0,0)');
  octx.fillStyle = halo;
  octx.beginPath(); octx.arc(0, 0, 165, 0, Math.PI * 2); octx.fill();

  // — ਕੋਰ: ਚਮਕਦਾਰ ਦਿਲ —
  const core = octx.createRadialGradient(-8, -10, 2, 0, 0, 58 * beat);
  core.addColorStop(0, css(cur.hot, 0.95));
  core.addColorStop(0.35, css(cur.mid, 0.75));
  core.addColorStop(1, css(cur.dim, 0));
  octx.fillStyle = core;
  octx.beginPath(); octx.arc(0, 0, 58 * beat, 0, Math.PI * 2); octx.fill();

  // — ਲਹਿਰ-ਵਾਲੀਆਂ ਦੋ ਲਹਿਰੀਆਂ (standing waves) —
  for (const [ph, col, wid, al] of [[0, cur.mid, 2.2, 0.85], [Math.PI, cur.dim, 1.4, 0.5]]) {
    octx.beginPath();
    for (let a = 0; a <= Math.PI * 2 + 0.01; a += 0.05) {
      const r = 78 * beat + Math.sin(a * cur.lobes + orbT * 4 + ph) * 9 * cur.pulse * 0.5;
      const x = Math.cos(a + orbT * 0.6) * r, y = Math.sin(a + orbT * 0.6) * r;
      a === 0 ? octx.moveTo(x, y) : octx.lineTo(x, y);
    }
    octx.strokeStyle = css(col, al);
    octx.lineWidth = wid;
    octx.shadowColor = css(col, 0.8); octx.shadowBlur = 14;
    octx.stroke(); octx.shadowBlur = 0;
  }

  // — ਘੁੰਮਦੇ ਕਣ —
  for (const d of ORB_DOTS) {
    d.a += d.v * cur.speed * 2;
    const tw = 0.35 + 0.65 * Math.abs(Math.sin(orbT * 2 + d.tw));
    const x = Math.cos(d.a) * d.r * cur.scale, y = Math.sin(d.a) * d.r * cur.scale;
    octx.fillStyle = css(cur.mid, 0.65 * tw);
    octx.beginPath(); octx.arc(x, y, d.s, 0, Math.PI * 2); octx.fill();
  }

  // — ਘੁੰਮਦਾ dash-ਘੇਰਾ (quantum dial) —
  octx.save();
  octx.rotate(-orbT * 0.35);
  octx.setLineDash([14, 22]);
  octx.strokeStyle = css(cur.dim, 0.55);
  octx.lineWidth = 1.6;
  octx.beginPath(); octx.arc(0, 0, 128 * beat, 0, Math.PI * 2); octx.stroke();
  octx.restore();

  octx.restore();
  requestAnimationFrame(drawOrb);
}
requestAnimationFrame(drawOrb);
// ===================== /QUANTUM ORB =====================

function speak(text) {
  if (!speakOn || !window.speechSynthesis || !text) return;
  const u = new SpeechSynthesisUtterance(String(text).slice(0, 400));
  const vs = speechSynthesis.getVoices();
  const paVoice = vs.find(v=>v.lang.startsWith('pa'));
  if (/[\u0A00-\u0A7F]/.test(String(text)) && !paVoice) return;  // no Punjabi voice installed
  u.voice = paVoice || vs.find(v=>v.lang.startsWith('en')) || vs[0];
  u.lang = (u.voice && u.voice.lang) || 'pa-IN';
  setMood('speaking');
  u.onend = () => setMood('idle');
  speechSynthesis.cancel(); speechSynthesis.speak(u);
}
/* mic: record in the browser, transcribe LOCALLY on this Mac (/api/stt).
   Web Speech was dropped: it sends audio to Apple/Google and Safari has no Punjabi. */
const micBtn = document.getElementById('mic');
const langSel = document.getElementById('lang');
try { langSel.value = localStorage.getItem('seven_lang') || 'pa'; } catch (e) {}
langSel.onchange = () => { try { localStorage.setItem('seven_lang', langSel.value); } catch (e) {} };
let mediaRec = null, chunks = [], live = false;

function micNote(html, isErr) {
  add('bot', `<div class="${isErr ? 'reply err' : 'meta'}">${html}</div>`);
}

async function startRec() {
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    micNote('ਇਸ ਬ੍ਰਾਊਜ਼ਰ ਵਿੱਚ ਮਾਈਕ ਰਿਕਾਰਡਿੰਗ ਨਹੀਂ। Chrome ਜਾਂ ਨਵਾਂ Safari ਵਰਤੋ।', true); return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: true});
  } catch (e) {
    micNote('ਮਾਈਕ ਦੀ ਇਜਾਜ਼ਤ ਨਹੀਂ ਮਿਲੀ (' + esc(e.name) + ')। ਬ੍ਰਾਊਜ਼ਰ ਦੀ address bar ਵਿੱਚ ਮਾਈਕ ਆਈਕਨ ਤੋਂ Allow ਕਰੋ, ਤੇ System Settings → Privacy → Microphone ਵਿੱਚ ਬ੍ਰਾਊਜ਼ਰ ਚਾਲੂ ਕਰੋ।', true);
    setMood('error', 1800); return;
  }
  chunks = [];
  mediaRec = new MediaRecorder(stream);
  mediaRec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
  mediaRec.onstop = async () => {
    stream.getTracks().forEach(t => t.stop());
    live = false; micBtn.classList.remove('live');
    const blob = new Blob(chunks, {type: mediaRec.mimeType || 'audio/webm'});
    if (blob.size < 2000) { micNote('ਕੁਝ ਸੁਣਿਆ ਨਹੀਂ — ਦੁਬਾਰਾ ਬੋਲੋ', true); setMood('idle'); return; }
    setMood('thinking');
    inp.placeholder = 'ਸੁਣ ਕੇ ਲਿਖ ਰਿਹਾ…';
    try {
      const r = await fetch('/api/stt', {method: 'POST', body: blob,
                 headers: {'Content-Type': blob.type, 'X-Lang': langSel.value}});
      const res = await r.json();
      inp.placeholder = 'ਬੋਲੋ ਜਾਂ ਲਿਖੋ...';
      if (res.error) { micNote('STT: ' + esc(res.error), true); setMood('error', 1800); return; }
      if (!res.text) { micNote('ਕੁਝ ਸਮਝ ਨਹੀਂ ਆਇਆ — ਦੁਬਾਰਾ ਬੋਲੋ', true); setMood('idle'); return; }
      inp.value = res.text;
      send();
    } catch (e) {
      inp.placeholder = 'ਬੋਲੋ ਜਾਂ ਲਿਖੋ...';
      micNote('STT ਸਰਵਰ ਨਾਲ ਗੱਲ ਨਹੀਂ ਹੋਈ: ' + esc(String(e)), true); setMood('error', 1800);
    }
  };
  mediaRec.start();
  live = true; micBtn.classList.add('live'); setMood('listening');
}

function toggleMic() { if (live && mediaRec) mediaRec.stop(); else startRec(); }
micBtn.onclick = toggleMic; orb.onclick = toggleMic;

/* chips */
const chipDefs = ['ਸੁਪਨਾ ਵੇਖ','ਯਾਦ ਕਰ','ਫੇਲ ਰਿਪੋਰਟ','ਟੀਮ','open chrome','youtube.com ਖੋਲ੍ਹੋ','search ਪੰਜਾਬ ਖ਼ਬਰਾਂ','ਮੌਸਮ ਲੁਧਿਆਣਾ','system','distance amritsar to london','circuit dht11 esp32','organize files'];
const chips = document.getElementById('chips');
chipDefs.forEach(c => {
  const d = document.createElement('div'); d.className='chip'; d.textContent=c;
  d.onclick = () => { inp.value = c; send(); };
  chips.appendChild(d);
});

async function forge() {
  const req = prompt('ਕਿਹੜਾ skill ਬਣਾਉਣਾ ਹੈ? (English ਵਧੀਆ: count words in text)');
  if (!req) return;
  const bubble = add('bot', '<span class="meta">skill ਲਿਖਿਆ ਜਾ ਰਿਹਾ…</span>');
  const res = await api('/api/forge', {request: req});
  if (res.error) { bubble.innerHTML = `<div class="out err">⛔ ${esc(res.error)}</div>`; return; }
  bubble.innerHTML = `<b>ਨਵਾਂ skill:</b> ${esc(res.name)}
      <div class="out">${esc(res.code)}</div>
      <div class="meta">verify: ${esc(JSON.stringify(res.verify_result))} · ${esc(res.drafter)}</div>`;
  const row = document.createElement('div'); row.className='row'; row.style.marginTop='8px';
  const yes=document.createElement('button'); yes.className='yes'; yes.textContent='✔ ਰਜਿਸਟਰ';
  const no =document.createElement('button'); no.className='no';  no.textContent='✘ ਰੱਦ';
  yes.onclick = async () => {
    const r2 = await api('/api/forge/approve', {forge_id: res.forge_id, approved: true});
    row.innerHTML = `<span class="tag ${r2.registered?'ok':'bad'}">${r2.registered?'ਰਜਿਸਟਰ ✔':'ਰੱਦ'}</span>`;
    refreshStatus(); };
  no.onclick = async () => { await api('/api/forge/approve', {forge_id: res.forge_id, approved: false});
    row.innerHTML = '<span class="tag bad">ਰੱਦ</span>'; };
  row.append(yes,no); bubble.appendChild(row); chat.scrollTop = chat.scrollHeight;
}
async function doUndo() {
  const res = await api('/api/undo', {});
  add('bot', `<b>↩ Undo</b><div class="out ${res.error?'err':''}">${esc(JSON.stringify(res))}</div>`);
  speak(res.undone ? 'ਵਾਪਸ ਕਰ ਦਿੱਤਾ' : (res.error || ''));
  refreshStatus();
}
async function showAudit() {
  const a = await (await fetch('/api/audit')).json();
  const lines = a.entries.map(e => `${new Date(e.ts*1000).toLocaleTimeString()}  ${e.event}`).join('\n');
  add('bot', `<b>Audit</b> — chain ${a.chain_valid?'✔':'✘ TAMPERED'}<div class="out">${esc(lines||'(empty)')}</div>`);
}

document.getElementById('send').onclick = send;
document.getElementById('forge').onclick = forge;
document.getElementById('undobtn').onclick = doUndo;
document.getElementById('audit').onclick = showAudit;
inp.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

add('bot', `ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ <b>7</b> — orb ਦਬਾ ਕੇ ਬੋਲੋ ਜਾਂ ਹੇਠ ਲਿਖੋ।
ਖ਼ਤਰਨਾਕ ਕਦਮ ਮਨਜ਼ੂਰੀ ਬਿਨਾਂ ਨਹੀਂ; ਉਲਟੇ ਜਾ ਸਕਣ ਵਾਲੇ <b>↩ Undo</b> ਨਾਲ ਵਾਪਸ।`);
if (!new URLSearchParams(location.search).get('mood')) setTimeout(() => speak('ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਜੀ, ਮੈਂ ਸੱਤ ਹਾਜ਼ਰ ਹਾਂ'), 1800);
refreshStatus(); refreshHud();

// ?mood=thinking — ਝਲਕ/ਟੈਸਟ ਲਈ (ਸਭ ਤੋਂ ਅਖੀਰ, greeting ਦੇ speak ਤੋਂ ਬਾਅਦ)
const _qm = new URLSearchParams(location.search).get('mood');
if (_qm) setMood(_qm);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    _forge_pending: Dict[str, Any] = {}

    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        base = self.path.split("?", 1)[0]
        if base == "/api/hud":
            self._json(SERVICE.hud_lines())
        elif base == "/":
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif base == "/api/status":
            self._json(SERVICE.status())
        elif base == "/api/audit":
            self._json(SERVICE.audit_tail())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            if self.path.split("?", 1)[0] == "/api/stt":
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0 or length > MAX_AUDIO_BYTES:
                    self._json({"error": "ਆਡੀਓ ਦਾ ਆਕਾਰ ਗਲਤ"}, 400)
                    return
                data = self.rfile.read(length)
                lang = self.headers.get("X-Lang", "pa")
                try:
                    res = STT.transcribe(data, lang)
                    HUD.add("voice", f"STT {res['engine']} {res['seconds']}s → '{res['text'][:40]}'")
                    self._json(res)
                except STTError as exc:
                    self._json({"error": str(exc)}, 400)
                return
            if self.path == "/api/chat":
                self._json(run_async(SERVICE.make_plan(self._body().get("message", ""))))
            elif self.path == "/api/execute":
                b = self._body()
                self._json(run_async(SERVICE.execute_step(b.get("action_id", ""),
                                                          bool(b.get("approved")))))
            elif self.path == "/api/undo":
                self._json(run_async(SERVICE.undo.undo_last()))
            elif self.path == "/api/forge":
                self._handle_forge(self._body())
            elif self.path == "/api/forge/approve":
                self._handle_forge_approve(self._body())
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _handle_forge(self, body: Dict[str, Any]) -> None:
        request = body.get("request", "")
        if not request:
            return self._json({"error": "empty request"})

        async def draft_and_verify():
            resp = await SERVICE.router.complete(
                DRAFTER_SYSTEM, f"Capability request: {request}\n\nPython code only:",
                trigger=request, n_tools=len(SERVICE.registry._tools))
            code = resp.text.strip()
            fence = code.find("```")
            if fence != -1:
                parts = code[fence:].split("```")
                code = parts[1] if len(parts) > 1 else code
                code = code.removeprefix("python").strip()
            static_check(code)
            result = await verify_in_subprocess(code, {})
            return resp, code, result

        try:
            resp, code, result = run_async(draft_and_verify())
        except ForgeError as exc:
            SERVICE.audit.record("forge_rejected", reason=str(exc), stage="pre-approval")
            return self._json({"error": f"ਸੁਰੱਖਿਆ/verification ਨਾਲ ਰੱਦ: {exc}"})
        except Exception as exc:
            return self._json({"error": f"{type(exc).__name__}: {exc}"})

        forge_id = f"forge_{uuid.uuid4().hex[:10]}"
        name = "skill_" + "".join(c if c.isalnum() else "_" for c in request.lower())[:28]
        Handler._forge_pending[forge_id] = {"code": code, "name": name,
                                            "request": request, "drafter": resp.provider}
        self._json({"forge_id": forge_id, "name": name, "code": code,
                    "verify_result": result, "drafter": resp.provider})

    def _handle_forge_approve(self, body: Dict[str, Any]) -> None:
        cand = Handler._forge_pending.pop(body.get("forge_id", ""), None)
        if cand is None:
            return self._json({"error": "unknown forge_id"}, 404)
        if not bool(body.get("approved")):
            SERVICE.audit.record("forge_rejected", name=cand["name"], reason="human veto")
            return self._json({"registered": False})
        if cand["name"] in SERVICE.registry._tools:
            return self._json({"error": "name collision", "registered": False})
        code_snapshot = cand["code"]

        async def skill_fn(payload):
            return await verify_in_subprocess(code_snapshot, payload)

        SERVICE.registry.register(ToolSpec(cand["name"], skill_fn, RiskLevel.LOW,
                                           Capability.NONE))
        SERVICE.audit.record("forge_registered", name=cand["name"], via="seven_ui")
        self._json({"registered": True, "name": cand["name"]})


class _Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)
    def write(self, s):
        self.f.write(s)
        sys.__stdout__.write(s)
    def flush(self):
        self.f.flush(); sys.__stdout__.flush()


def main() -> None:
    global SERVICE
    SERVICE = SevenService(STATE)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"  ┌───────────────────────────────────────────┐")
    print(f"  │   7  →  http://127.0.0.1:{PORT}              │")
    print(f"  │   ਆਵਾਜ਼ + ਲਿਖਤ · Ctrl+C ਨਾਲ ਬੰਦ            │")
    print(f"  └───────────────────────────────────────────┘")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n7 ਬੰਦ। ਫਿਰ ਮਿਲਦੇ ਹਾਂ!")


if __name__ == "__main__":
    main()
