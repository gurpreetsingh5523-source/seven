"""
Agent 7 (ਏਜੰਟ ਸੱਤ) — a hybrid local+API autonomous agent, built on
sentinel_orchestrator.py.

Why "7": seven layers, each one learned from a documented strength or failure
of OpenAI Dots and Meta Muse (see ਏਜੰਟ_ਰਿਸਰਚ_ਰਿਪੋਰਟ.md):

  1. SENSE    — triggers (event/schedule) enter the loop          [Dots: always-on]
  2. THINK    — a REAL LLM planner (not hardcoded steps)          [v1's fatal flaw]
  3. ROUTE    — cost-aware hybrid brain: local model (free) first,
                escalate to a paid API only when needed           [the equity layer:
                neither Dots ($100–$500/mo) nor Muse (Meta-locked) offers this]
  4. GUARD    — PolicyEngine: Agents Rule of Two                  [Muse]
  5. PERMIT   — out-of-band, time-boxed, default-deny approvals   [Muse Sentinel]
  6. ACT      — sandbox + idempotency + retries + checkpoints     [v2]
  7. REMEMBER — compacting memory + tamper-evident audit          [v2]

Equity design goals ("ਅਮੀਰ ਅਤੇ ਗਰੀਬ ਦੋਹਾਂ ਲਾਭ ਲੈ ਸਕਣ"):
  - Fully functional OFFLINE with a local model (Ollama: llama/qwen/deepseek).
    Zero spend, zero lock-in, data never leaves the machine.
  - Paid API is an OPTIONAL accelerator, engaged only by the router when the
    task's complexity exceeds the local model's proven ability — and hard-capped
    by a daily budget. Budget exhausted => graceful local-only mode, never a crash.
  - Stdlib only for the provider clients (urllib) — no SDK dependency, so it
    runs on a $35 Raspberry Pi as easily as on a workstation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from sentinel_orchestrator import (
    ActionIntent, ApprovalGate, AuditLog, Capability, CheckpointStore,
    LocalSandbox, Orchestrator, PlanValidator, PolicyEngine, RiskLevel,
    SecretsBroker, StepStatus, ToolRegistry, ToolSpec,
)

log = logging.getLogger("agent7")


# --------------------------------------------------------------------------- #
# LAYER 3a. LLM providers — local (free) and API (paid), one interface        #
# --------------------------------------------------------------------------- #

@dataclass
class LLMResponse:
    text: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class LLMProvider(Protocol):
    name: str
    cost_per_1k_in: float
    cost_per_1k_out: float
    local: bool

    async def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> LLMResponse:
        ...

    async def available(self) -> bool:
        ...


class ProviderUnavailable(Exception):
    pass


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough heuristic; providers may report exact


class OllamaProvider:
    """
    Free, private, offline brain. Talks to a local Ollama server over stdlib
    urllib (no SDK). Default model: a small open-weight instruct model.
    """

    name = "ollama-local"
    cost_per_1k_in = 0.0
    cost_per_1k_out = 0.0
    local = True

    def __init__(self, model: str = "qwen3:8b", host: str = "http://localhost:11434",
                 timeout_s: float = 60.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    async def available(self) -> bool:
        def _ping() -> bool:
            try:
                with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                    return r.status == 200
            except Exception:
                return False
        return await asyncio.to_thread(_ping)

    async def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> LLMResponse:
        body = json.dumps({
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1},
        }).encode()

        def _call() -> Dict[str, Any]:
            req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return json.loads(r.read())
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                raise ProviderUnavailable(f"ollama unreachable: {exc}") from exc

        data = await asyncio.to_thread(_call)
        text = data.get("response", "")
        return LLMResponse(
            text=text,
            provider=self.name,
            input_tokens=data.get("prompt_eval_count", _estimate_tokens(prompt)),
            output_tokens=data.get("eval_count", _estimate_tokens(text)),
            cost_usd=0.0,
        )


class LocalOpenAICompatProvider:
    """
    Free, private, offline brain served by any local OpenAI-compatible server
    (mlx_lm.server on Apple Silicon, llama.cpp server, LM Studio). Stdlib only.
    Default: mlx_lm.server on 127.0.0.1:8095. Only loopback hosts are allowed,
    so this can never become a paid/remote call by misconfiguration.
    """

    name = "local-mlx"
    cost_per_1k_in = 0.0
    cost_per_1k_out = 0.0
    local = True

    def __init__(self, host: str = "http://127.0.0.1:8095", model: str = "default_model",
                 timeout_s: float = 120.0):
        parsed = urllib.parse.urlparse(host)
        if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(f"local provider must be loopback, got {parsed.hostname!r}")
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    async def available(self) -> bool:
        def _ping() -> bool:
            try:
                with urllib.request.urlopen(f"{self.host}/v1/models", timeout=3) as r:
                    return r.status == 200
            except Exception:
                return False
        return await asyncio.to_thread(_ping)

    async def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> LLMResponse:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }).encode()

        def _call() -> Dict[str, Any]:
            req = urllib.request.Request(f"{self.host}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return json.loads(r.read())
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                raise ProviderUnavailable(f"local server unreachable: {exc}") from exc

        data = await asyncio.to_thread(_call)
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=text, provider=self.name,
            input_tokens=usage.get("prompt_tokens", _estimate_tokens(prompt)),
            output_tokens=usage.get("completion_tokens", _estimate_tokens(text)),
            cost_usd=0.0,
        )


class OpenAIProvider:
    """
    Paid, powerful brain — OPTIONAL accelerator. Key comes from the env, never
    from the plan/context (surrogation discipline applies to providers too).
    """

    name = "openai-api"
    local = False

    def __init__(self, model: str = "gpt-4o-mini",
                 cost_per_1k_in: float = 0.00015, cost_per_1k_out: float = 0.0006,
                 timeout_s: float = 60.0):
        self.model = model
        self.cost_per_1k_in = cost_per_1k_in
        self.cost_per_1k_out = cost_per_1k_out
        self.timeout_s = timeout_s
        self._key = os.environ.get("OPENAI_API_KEY", "")

    async def available(self) -> bool:
        return bool(self._key)

    async def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> LLMResponse:
        if not self._key:
            raise ProviderUnavailable("OPENAI_API_KEY not set")
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }).encode()

        def _call() -> Dict[str, Any]:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self._key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return json.loads(r.read())
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                raise ProviderUnavailable(f"openai unreachable: {exc}") from exc

        data = await asyncio.to_thread(_call)
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        ti = usage.get("prompt_tokens", _estimate_tokens(prompt))
        to = usage.get("completion_tokens", _estimate_tokens(text))
        cost = ti / 1000 * self.cost_per_1k_in + to / 1000 * self.cost_per_1k_out
        return LLMResponse(text=text, provider=self.name, input_tokens=ti,
                           output_tokens=to, cost_usd=cost)


# --------------------------------------------------------------------------- #
# LAYER 3b. Cost ledger + equity router                                       #
# --------------------------------------------------------------------------- #

class BudgetExhausted(Exception):
    pass


@dataclass
class CostLedger:
    """Hard daily cap on paid-API spend. Local inference is always free."""
    daily_budget_usd: float = 1.00
    _spent: float = 0.0
    _day: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))

    def _roll(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day, self._spent = today, 0.0

    def can_spend(self) -> bool:
        self._roll()
        return self._spent < self.daily_budget_usd

    def record(self, cost_usd: float) -> None:
        self._roll()
        self._spent += cost_usd

    @property
    def spent_today(self) -> float:
        self._roll()
        return self._spent


class HybridRouter:
    """
    The equity layer — neither Dots nor Muse has one.

    Policy:
      * Every request is OFFERED to the local (free) brain first when the task's
        complexity score is within the local model's ability band.
      * Escalation to the paid API happens only when: (a) complexity exceeds the
        band, or (b) the local brain's output fails to parse/validate, or
        (c) the local brain is unreachable — and ONLY while budget remains.
      * Budget exhausted or key missing => local-only mode. The agent DEGRADES,
        it never dies. That is the whole point.
    """

    def __init__(self, local: LLMProvider, remote: Optional[LLMProvider],
                 ledger: CostLedger, local_complexity_ceiling: int = 40):
        self.local = local
        self.remote = remote
        self.ledger = ledger
        self.ceiling = local_complexity_ceiling

    @staticmethod
    def complexity(trigger: str, n_tools: int) -> int:
        """
        Transparent heuristic (documented, tunable): longer, multi-entity
        triggers and larger tool surfaces push the score up. A real deployment
        can replace this with a learned router without touching the rest.
        """
        words = len(trigger.split())
        digits = len(re.findall(r"\d", trigger))
        multi = len(re.findall(r"\band\b|,|;", trigger))
        return words + digits + 3 * multi + 2 * n_tools

    async def complete(self, system: str, prompt: str, *, trigger: str,
                       n_tools: int, max_tokens: int = 2048) -> LLMResponse:
        score = self.complexity(trigger, n_tools)
        prefer_local = score <= self.ceiling
        order: List[LLMProvider] = ([self.local, self.remote] if prefer_local
                                    else [self.remote, self.local])
        order = [p for p in order if p is not None]

        errors: List[str] = []
        for provider in order:
            if not provider.local and not self.ledger.can_spend():
                errors.append(f"{provider.name}: daily budget exhausted")
                continue
            if not await provider.available():
                errors.append(f"{provider.name}: unavailable")
                continue
            try:
                resp = await provider.complete(system, prompt, max_tokens)
            except ProviderUnavailable as exc:
                errors.append(str(exc))
                continue
            if not provider.local:
                self.ledger.record(resp.cost_usd)
            log.info("router -> %s (complexity=%d, ceiling=%d, spent=$%.4f)",
                     provider.name, score, self.ceiling, self.ledger.spent_today)
            return resp
        raise ProviderUnavailable("all providers failed: " + "; ".join(errors))


# --------------------------------------------------------------------------- #
# LAYER 2. The real planner — LLM proposes, PlanValidator disposes            #
# --------------------------------------------------------------------------- #

PLANNER_SYSTEM = """You are the planning component of an autonomous agent.
Convert the trigger into a JSON array of steps. Rules:
- Use ONLY tools from the provided registry; never invent tools.
- Each step: {"tool": str, "payload": object, "risk": "low|medium|high|critical",
  "capabilities": ["untrusted_input","sensitive_access","external_effect"]}
- Payloads must satisfy the tool's schema hints. Use "${SECRET:name}" surrogates
  for credentials — NEVER real values.
- Split any step that would need all three capabilities into separate steps.
- If the trigger is conversation or a question that needs no tool (greetings,
  "who are you", explanations), answer it yourself with ONE chat_reply step:
  {"tool": "chat_reply", "payload": {"text": "<your answer>"}, "risk": "low", "capabilities": []}
  Reply in the user's language and script: Gurmukhi Punjabi if they wrote
  Gurmukhi, romanised PUNJABI (not Hindi!) if they wrote romanised Punjabi
  (e.g. "Main 7 haan, tuhada local assistant." — not "Main 7 hoon, aapki"),
  else English. You are "7", a
  local, private assistant running on the user's own Mac.
- Output ONLY the JSON array. No prose, no markdown fences."""

_CAPS = {
    "untrusted_input": Capability.UNTRUSTED_INPUT,
    "sensitive_access": Capability.SENSITIVE_ACCESS,
    "external_effect": Capability.EXTERNAL_EFFECT,
}
_RISKS = {r.value: r for r in RiskLevel}


# Payload shapes the planner must follow (keys read by the tool functions).
TOOL_HINTS: Dict[str, str] = {
    "chat_reply": '{"text": "<answer to the user>"}',
    "weather_report": '{"city": "<city name, e.g. Ludhiana>"}',
    "globe_distance": '{"from_city": "<city>", "to_city": "<city>"}',
    "system_monitor": "{}  (CPU / RAM / disk usage)",
    "app_open": '{"name": "<app, e.g. chrome>", "dry_run": false}',
    "site_open": '{"url": "https://...", "dry_run": false}',
    "web_search": '{"query": "<search text>", "engine": "<google|bing|duckduckgo>", "dry_run": false}',
    "smart_search": '{"query": "<search text>"}',
    "web_fetch": '{"url": "https://...", "max_bytes": 200000}',
    "circuit_schematic": '{"sensor": "<e.g. dht11>", "board": "<e.g. esp32>"}',
    "arduino_generate": '{"led_pin": 13, "interval_ms": 1000}',
    "file_organizer": '{"folder": "<absolute path>", "dry_run": true}',
    "code_review": '{"code": "<python source>"}',
    "reverse_inspect": '{"path": "<file path>"}',
    "recall": '{"query": "<what to remember>"}',
    "memory_write": '{"key": "<name>", "value": "<text to store>"}',
    "doc_builder": '{"filename": "<name.md>", "content": "<markdown>"}',
    "project_board": '{"project": "<project name>", "steps": ["<step 1>", "<step 2>"]}',
    "foresee": '{"task": "<task description>"}',
    "evolve": '{"approved": false}',
    "board": '{"votes": {"<agent key>": "<option>"}}',
    "guard_scan": '{"text": "<untrusted text to scan>"}',
    "page_draft": '{"title": "<page title>", "content": "<page text>"}',
    "page_approve": '{"draft_id": "<draft id>"}',
    "page_reject": '{"draft_id": "<draft id>"}',
    "page_read": '{"page_id": "<page id>"}',
    "team_ask": '{"message": "<question for the team>"}',
    "browser_task": '{"steps": [{"goto": "https://..."}], "headless": true}',
    "ask_hermes": '{"task": "<what Hermes should do>"}',
    "hermes_result": '{"count": 1}',
    "desktop_open_app": '{"app": "<application name, e.g. Calculator>"}',
    "mac_info": '{"what": "volume|apps|frontmost|battery|all"}',
    "assistant_pause": '{"minutes": 10}',
    "clipboard_read": "{}",
    "reminders_list": '{"limit": 20}',
    "files_find": '{"query": "<name words>", "limit": 20}',
    "mac_volume": '{"level": 50, "mute": false}',
    "mac_brightness": '{"direction": "up|down", "steps": 2}',
    "clipboard_write": '{"text": "<text to copy>"}',
    "window_control": '{"action": "minimize|maximize|left_half|right_half|center"}',
    "music_control": '{"action": "play|pause|playpause|next|previous|now_playing", "app": "Music|Spotify", "query": ""}',
    "reminder_add": '{"title": "<what to remember>", "due": "2026-10-07 09:00", "list": ""}',
    "reminder_delete": '{"title": "<exact reminder name>", "all": false}',
    "reminder_complete": '{"title": "<exact reminder name>", "all": false}',
    "type_text": '{"text": "<text to type, max 2000 chars>"}',
    "key_shortcut": '{"keys": "cmd+shift+4"}',
    "file_op": '{"op": "move|copy|rename|mkdir", "src": "<path in home>", "dst": "<path in home>"}',
    "app_quit": '{"app": "Safari", "force": false}',
    "file_trash": '{"path": "<path in home>"}',
    "dream": "{}", "failure_report": "{}", "team_status": "{}", "undo": "{}", "page_list": "{}",
    "doc_create": '{"kind": "docx|xlsx|pptx|pdf", "name": "report", "text": "# Heading\\n- bullet", "slides": [{"title": "", "bullets": [""]}], "rows": [[""]], "show": true}',
    "doc_read": '{"path": "<file: docx|xlsx|pptx|pdf|txt|md|csv, relative to Documents/7 or absolute in home>"}',
    "doc_edit": '{"path": "<file>", "action": "append|replace|set_cell|append_row|add_slide", "text": "", "find": "", "cell": "B2", "value": "", "row": [""], "slide": {"title": "", "bullets": [""]}, "show": true}',
    "doc_to_pdf": '{"path": "<docx|pptx|md|txt file>", "show": true}',
    "calendar_list": '{"days": 1, "start": "2026-10-07"}',
    "calendar_add": '{"title": "<event>", "start": "2026-10-07 15:00", "end": "", "calendar": "", "location": "", "notes": ""}',
    "daily_briefing": '{"city": ""}',
    "news_show": '{"feed": "world|india", "limit": 8}',
    "news_live": '{"channel": "Al Jazeera English|DW News|Sky News|France 24 English"}',
    "news_close": "{}",
}


class PlanParseError(Exception):
    pass


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Tolerant extraction: raw JSON, fenced JSON, or JSON embedded in prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise PlanParseError("no JSON array found in model output")
        candidate = text[start:end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(s, dict) for s in data):
        raise PlanParseError("plan must be a JSON array of step objects")
    return data


class LLMPlanner:
    """
    Turns a trigger into a typed plan via the hybrid router. The model's output
    is treated as UNTRUSTED INPUT: parsed tolerantly, then passed through the
    same PlanValidator (registry + schema) that guards every plan. If the local
    model produces garbage, one escalation retry happens via the router's
    remote provider — after that, the plan fails loudly, never silently.
    """

    def __init__(self, router: HybridRouter, registry: ToolRegistry,
                 broker: SecretsBroker, audit: Optional[AuditLog] = None):
        self.router = router
        self.registry = registry
        self.broker = broker
        self.audit = audit
        self._validator = PlanValidator(registry)

    def _describe_registry(self) -> str:
        # The planner used to see only tool names, so it emitted empty payloads
        # (e.g. weather_report {} with no city). Show the payload shape too.
        lines = []
        for name, spec in self.registry._tools.items():
            hint = TOOL_HINTS.get(name)
            lines.append(f"- {name}: risk={spec.risk.value}" + (f"  payload: {hint}" if hint else ""))
        return "\n".join(lines) or "(no tools registered)"

    async def plan(self, trigger: str, context: List[str]) -> List[ActionIntent]:
        prompt = (
            f"Tool registry:\n{self._describe_registry()}\n\n"
            f"Recent context: {json.dumps(context[-4:], default=str)}\n\n"
            f"Trigger: {trigger}\n\nPlan (JSON array only):"
        )
        last_error: Optional[Exception] = None
        for attempt in (1, 2):
            try:
                resp = await self.router.complete(
                    PLANNER_SYSTEM, prompt,
                    trigger=trigger, n_tools=len(self.registry._tools))
                intents = self._to_intents(extract_json_array(resp.text))
                return self._validator.validate(intents)  # raises on bad plan
            except (PlanParseError, ProviderUnavailable, ValueError, KeyError) as exc:
                last_error = exc
                log.warning("plan attempt %d failed (%s); %s", attempt, exc,
                            "escalating" if attempt == 1 else "giving up")
        raise PlanParseError(f"planner failed after escalation: {last_error}")

    def _to_intents(self, steps: List[Dict[str, Any]]) -> List[ActionIntent]:
        intents: List[ActionIntent] = []
        for i, step in enumerate(steps):
            try:
                caps = Capability.NONE
                for c in step.get("capabilities", []):
                    caps |= _CAPS[c]
                intents.append(ActionIntent(
                    tool=step["tool"],
                    payload=step.get("payload", {}),
                    risk=_RISKS[step.get("risk", "low")],
                    capabilities=caps,
                    requires_approval=bool(step.get("requires_approval", False)),
                ))
            except KeyError as exc:
                raise PlanParseError(f"step {i}: unknown field value {exc}") from exc
        if not intents:
            raise PlanParseError("empty plan")
        return intents


# --------------------------------------------------------------------------- #
# LAYERS 1+4+5+6+7. Agent 7 — thin assembly over the sentinel core            #
# --------------------------------------------------------------------------- #

class Agent7:
    """
    Assembly point. Inherits the full security/resilience stack from the
    sentinel Orchestrator (GUARD/PERMIT/ACT/REMEMBER) and adds the hybrid
    brain (SENSE/THINK/ROUTE). SENSE = run(trigger) entry point; schedules can
    call it from cron/systemd — always-on without a cloud bill.
    """

    def __init__(self,
                 registry: ToolRegistry,
                 tools_sandbox,                       # Sandbox impl
                 approval_callback,                   # out-of-band approver
                 state_dir: Path,
                 *,
                 local_model: str = "qwen3:8b",
                 remote_model: str = "gpt-4o-mini",
                 daily_budget_usd: float = 1.00,
                 local_complexity_ceiling: int = 40,
                 local_provider: Optional[LLMProvider] = None,
                 remote_provider: Optional[LLMProvider] = None,
                 ):
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.broker = SecretsBroker(vault={"ci_token": os.environ.get("CI_TOKEN", "")})
        audit = AuditLog(state_dir / "agent7_audit.jsonl")
        ledger = CostLedger(daily_budget_usd=daily_budget_usd)
        self.router = HybridRouter(
            local=local_provider or OllamaProvider(model=local_model),
            remote=remote_provider if remote_provider is not None else OpenAIProvider(model=remote_model),
            ledger=ledger,
            local_complexity_ceiling=local_complexity_ceiling,
        )
        planner = LLMPlanner(self.router, registry, self.broker, audit)
        self.core = Orchestrator(
            name="Agent-7",
            planner=planner,
            registry=registry,
            sandbox=tools_sandbox,
            approval_gate=ApprovalGate(approval_callback, timeout_s=300.0),
            audit=audit,
            checkpoints=CheckpointStore(state_dir / "agent7_checkpoints.json"),
            policy=PolicyEngine(),
        )
        self.ledger = ledger

    async def run(self, trigger: str) -> Dict[str, Any]:
        summary = await self.core.run(trigger)
        summary["brain"] = {
            "daily_budget_usd": self.ledger.daily_budget_usd,
            "spent_today_usd": round(self.ledger.spent_today, 6),
            "mode": "local+api" if self.ledger.can_spend() else "local-only",
        }
        return summary
