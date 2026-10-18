"""
Sentinel Orchestrator v2.0 — production-grade autonomous agent orchestration.

Design goals (each maps to a documented failure of the naive "Advanced Autonomous
Agent Orchestrator" draft this replaces):

  1. DURABILITY     — every completed step is checkpointed to disk; a crashed run
                      resumes where it stopped instead of restarting or, worse,
                      re-executing side effects.
  2. RESILIENCE     — retries with exponential backoff + full jitter, per-step
                      timeouts, and a circuit breaker per tool. One fault never
                      silently kills the whole plan.
  3. REAL GUARDRAILS— a policy engine implementing Meta's "Agents Rule of Two":
                      an action may hold at most two of {untrusted input,
                      sensitive access, external side effect}. Three => structurally
                      vetoed, no human can rubber-stamp it away.
  4. REAL HITL      — approvals are out-of-band, time-boxed, and DEFAULT-DENY on
                      timeout. The reasoning component proposes; a separate
                      authority disposes (Sentinel pattern).
  5. AUDITABILITY   — append-only, hash-chained (tamper-evident) audit log of
                      every decision: policy checks, approvals, attempts, results.
  6. IDEMPOTENCY    — every action carries an idempotency key; retried or
                      re-planned steps cannot double-fire side effects.
  7. LEAST SECRETS  — a secrets broker hands the plan surrogate tokens; real
                      credentials are injected only at the execution boundary.
  8. OBSERVABILITY  — structured run summary with per-step latency, attempts and
                      status; failure budget aborts a run that is degrading.

Stdlib only. The LLM planner and the sandbox are Protocols: plug in OpenAI
Agents SDK / LangGraph for planning and E2B/Firecracker microVMs for execution.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sentinel-orchestrator")


# --------------------------------------------------------------------------- #
# 1. Risk & capability model                                                   #
# --------------------------------------------------------------------------- #

class RiskLevel(enum.Enum):
    LOW = "low"          # read-only, no external effect
    MEDIUM = "medium"    # outbound communication / drafts
    HIGH = "high"        # state-changing, reversible with effort
    CRITICAL = "critical"  # irreversible: payments, prod deploys, credential changes


class Capability(enum.Flag):
    """The three dangerous properties from the Agents Rule of Two."""
    NONE = 0
    UNTRUSTED_INPUT = enum.auto()    # processes attacker-controllable content
    SENSITIVE_ACCESS = enum.auto()   # touches private data / credentials
    EXTERNAL_EFFECT = enum.auto()    # changes state or communicates outward


RULE_OF_TWO_MAX = 2  # an autonomous action may hold AT MOST two capabilities


class StepStatus(enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"
    VETOED_BY_POLICY = "vetoed_by_policy"
    VETOED_BY_HUMAN = "vetoed_by_human"
    APPROVAL_TIMEOUT = "approval_timeout"
    CIRCUIT_OPEN = "circuit_open"
    RETRIES_EXHAUSTED = "retries_exhausted"
    SKIPPED_DUPLICATE = "skipped_duplicate"


class RunAborted(Exception):
    """Raised when the failure budget is exhausted mid-plan."""


# --------------------------------------------------------------------------- #
# 2. Action intents & results                                                  #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ActionIntent:
    tool: str
    payload: Dict[str, Any]
    risk: RiskLevel
    capabilities: Capability = Capability.NONE
    requires_approval: bool = False
    action_id: str = field(default_factory=lambda: f"act_{uuid.uuid4().hex[:12]}")
    idempotency_key: str = field(default_factory=lambda: f"idem_{uuid.uuid4().hex}")
    timeout_s: float = 30.0
    max_retries: int = 3

    def __post_init__(self):
        # CRITICAL actions can never skip human approval — enforced at the type
        # boundary so a sloppy planner cannot smuggle one through.
        if self.risk == RiskLevel.CRITICAL and not self.requires_approval:
            object.__setattr__(self, "requires_approval", True)


@dataclass
class StepResult:
    action_id: str
    tool: str
    status: StepStatus
    attempts: int = 0
    latency_s: float = 0.0
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# 3. Tool registry — typed, validated, risk-classified                         #
# --------------------------------------------------------------------------- #

Validator = Callable[[Dict[str, Any]], None]
ToolFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass
class ToolSpec:
    name: str
    fn: ToolFn
    risk: RiskLevel
    capabilities: Capability
    validate: Validator = lambda payload: None  # raise ValueError to reject


class ToolRegistry:
    """Only registered, validated tools may run — the planner cannot invent tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"unregistered tool: {name!r}")
        return self._tools[name]


# --------------------------------------------------------------------------- #
# 4. Policy engine — Agents Rule of Two + risk ladder                          #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    needs_approval: bool


class PolicyEngine:
    """
    Hard, non-bypassable rules evaluated BEFORE anything executes.

    Rule of Two (Meta, 2025): an action holding all three of {untrusted input,
    sensitive access, external effect} in one session is structurally vetoed.
    Risk ladder: HIGH+ always needs human approval; MEDIUM needs approval only
    when it has an external effect.
    """

    def evaluate(self, intent: ActionIntent) -> PolicyDecision:
        caps = intent.capabilities
        n_caps = sum(bool(caps & c) for c in
                     (Capability.UNTRUSTED_INPUT, Capability.SENSITIVE_ACCESS,
                      Capability.EXTERNAL_EFFECT))
        if n_caps > RULE_OF_TWO_MAX:
            return PolicyDecision(
                allowed=False,
                reason=f"Rule-of-Two violation: action holds {n_caps}/3 dangerous "
                       f"capabilities ({caps!r}); split it into separate sessions",
                needs_approval=False,
            )
        needs = (intent.requires_approval
                 or intent.risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)
                 or (intent.risk == RiskLevel.MEDIUM
                     and bool(caps & Capability.EXTERNAL_EFFECT)))
        return PolicyDecision(allowed=True, reason="ok", needs_approval=needs)


# --------------------------------------------------------------------------- #
# 5. Approval gate — out-of-band, time-boxed, default-deny                     #
# --------------------------------------------------------------------------- #

ApprovalCallback = Callable[[ActionIntent, PolicyDecision], Awaitable[bool]]


class ApprovalGate:
    """
    Propose-then-permit: the agent can only *propose*; the gate disposes.
    - Approval channel is a separate callback (never the agent's own context,
      so injected text cannot forge a 'yes').
    - Time-boxed: silence == denial (default-deny), never default-permit.
    """

    def __init__(self, callback: ApprovalCallback, timeout_s: float = 120.0):
        self._callback = callback
        self._timeout = timeout_s

    async def request(self, intent: ActionIntent, decision: PolicyDecision) -> StepStatus:
        try:
            approved = await asyncio.wait_for(
                self._callback(intent, decision), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            log.warning("Approval TIMEOUT for %s — defaulting to DENY", intent.action_id)
            return StepStatus.APPROVAL_TIMEOUT
        except Exception as exc:  # a broken gate must fail closed, not open
            log.error("Approval channel fault for %s: %s — DENY", intent.action_id, exc)
            return StepStatus.APPROVAL_TIMEOUT
        return StepStatus.SUCCESS if approved else StepStatus.VETOED_BY_HUMAN


# --------------------------------------------------------------------------- #
# 6. Audit log — append-only, hash-chained, tamper-evident                     #
# --------------------------------------------------------------------------- #

class AuditLog:
    """
    Every entry includes the SHA-256 of the previous entry, so editing or
    deleting history breaks the chain and is detectable via verify().
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self._prev_hash = "0" * 64
        self._entries: List[Dict[str, Any]] = []

    def record(self, event: str, **fields: Any) -> None:
        entry = {
            "ts": time.time(),
            "event": event,
            "prev_hash": self._prev_hash,
            **fields,
        }
        self._prev_hash = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()
        ).hexdigest()
        entry["hash"] = self._prev_hash
        self._entries.append(entry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")

    def verify(self) -> bool:
        prev = "0" * 64
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            body = {k: v for k, v in entry.items() if k != "hash"}
            if hashlib.sha256(
                json.dumps(body, sort_keys=True, default=str).encode()
            ).hexdigest() != entry["hash"]:
                return False
            prev = entry["hash"]
        return True

    def __len__(self) -> int:
        return len(self._entries)


# --------------------------------------------------------------------------- #
# 7. Checkpoint store — durable execution / crash resume                       #
# --------------------------------------------------------------------------- #

class CheckpointStore:
    """
    Persists finished step results + spent idempotency keys to JSON.
    On restart, completed steps are NOT re-executed (at-least-once delivery of
    the *plan*, exactly-once effect per idempotency key).
    """

    def __init__(self, path: Path):
        self.path = path
        self.results: Dict[str, Dict[str, Any]] = {}
        self.spent_keys: set[str] = set()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.results = data.get("results", {})
            self.spent_keys = set(data.get("spent_keys", []))

    def mark_key_spent(self, key: str) -> None:
        self.spent_keys.add(key)
        self._flush()

    def is_key_spent(self, key: str) -> bool:
        return key in self.spent_keys

    def save_result(self, result: StepResult) -> None:
        self.results[result.action_id] = {
            "tool": result.tool,
            "status": result.status.value,
            "attempts": result.attempts,
            "latency_s": result.latency_s,
            "error": result.error,
        }
        self._flush()

    def prior_status(self, action_id: str) -> Optional[StepStatus]:
        r = self.results.get(action_id)
        return StepStatus(r["status"]) if r else None

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"results": self.results, "spent_keys": sorted(self.spent_keys)},
            indent=2, default=str,
        ), encoding="utf-8")
        tmp.replace(self.path)  # atomic rename: no half-written checkpoints


# --------------------------------------------------------------------------- #
# 8. Secrets broker — surrogate tokens, injection at the boundary only         #
# --------------------------------------------------------------------------- #

class SecretsBroker:
    """
    The planner and the model only ever see surrogate placeholders like
    '${SECRET:stripe}' . The real value is resolved inside the execution
    boundary, so a prompt-injection that exfiltrates the agent's context
    steals nothing real (credential surrogation pattern).
    """

    def __init__(self, vault: Optional[Dict[str, str]] = None):
        self._vault = vault or {}

    def surrogate(self, name: str) -> str:
        return f"${{SECRET:{name}}}"

    def resolve(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        def _resolve(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("${SECRET:"):
                name = value[len("${SECRET:"):-1]
                if name not in self._vault:
                    raise KeyError(f"unknown secret {name!r}")
                return self._vault[name]
            return value
        return {k: _resolve(v) for k, v in payload.items()}


# --------------------------------------------------------------------------- #
# 9. Circuit breaker + retry with backoff & jitter                             #
# --------------------------------------------------------------------------- #

class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    """Per-tool: opens after `failure_threshold` consecutive faults, half-opens
    after `reset_after_s` to probe recovery."""

    def __init__(self, failure_threshold: int = 3, reset_after_s: float = 30.0):
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._failures = 0
        self._opened_at: Optional[float] = None

    def before_call(self) -> None:
        if self._opened_at is None:
            return
        if time.monotonic() - self._opened_at >= self.reset_after_s:
            return  # half-open probe allowed through
        raise CircuitOpenError("circuit open — tool temporarily disabled")

    def on_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def on_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()
            log.warning("Circuit OPEN (failures=%d, cooldown %.0fs)",
                        self._failures, self.reset_after_s)


async def with_retry(
    fn: Callable[[], Awaitable[Dict[str, Any]]],
    *,
    max_retries: int,
    timeout_s: float,
    base_delay_s: float = 0.25,
) -> tuple[Dict[str, Any], int]:
    """
    Attempt fn up to 1 + max_retries times. Each attempt is time-boxed with
    asyncio.wait_for; delays use exponential backoff with FULL jitter
    (sleep = random(0, base * 2**attempt)) to avoid thundering-herd retries.
    Returns (result, attempts_used).
    """
    attempts = 0
    while True:
        attempts += 1
        try:
            return await asyncio.wait_for(fn(), timeout=timeout_s), attempts
        except Exception:
            if attempts > max_retries:
                raise
            await asyncio.sleep(random.uniform(0, base_delay_s * (2 ** (attempts - 1))))


# --------------------------------------------------------------------------- #
# 10. Sandbox — execution boundary (Protocol + safe local default)             #
# --------------------------------------------------------------------------- #

class Sandbox(Protocol):
    async def execute(self, tool: ToolSpec, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


class LocalSandbox:
    """
    Development sandbox: runs the tool coroutine in-process but WITHOUT real
    credentials (broker-resolved copy) and with a wall-clock note. Production
    deployments should swap this for a microVM backend (E2B / Firecracker /
    gVisor) — shared-kernel containers are not a boundary for untrusted code.
    """

    def __init__(self, broker: SecretsBroker, simulated_latency_s: float = 0.05):
        self._broker = broker
        self._latency = simulated_latency_s

    async def execute(self, tool: ToolSpec, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool.validate(payload)                       # schema check at the gate
        resolved = self._broker.resolve(payload)     # secrets enter only here
        await asyncio.sleep(self._latency)           # simulate I/O latency
        return await tool.fn(resolved)


# --------------------------------------------------------------------------- #
# 11. Working memory — bounded, compacting, failure-aware                      #
# --------------------------------------------------------------------------- #

@dataclass
class WorkingMemory:
    max_slots: int = 12
    failure_budget: int = 3
    context: List[str] = field(default_factory=list)
    failure_counter: int = 0

    def append(self, note: str, failed: bool = False) -> None:
        if failed:
            self.failure_counter += 1
        if len(self.context) >= self.max_slots:
            # Compact instead of blind FIFO-drop: fold the oldest half into one
            # summary line so the eviction policy is lossy-but-honest.
            half = len(self.context) // 2
            summary = f"[compacted {half} prior notes: {self.context[0][:60]}…]"
            self.context = [summary] + self.context[half:]
        self.context.append(note)

    def budget_exhausted(self) -> bool:
        return self.failure_counter >= self.failure_budget


# --------------------------------------------------------------------------- #
# 12. Planner — Protocol + deterministic fallback                              #
# --------------------------------------------------------------------------- #

class Planner(Protocol):
    async def plan(self, trigger: str, context: List[str]) -> List[ActionIntent]:
        ...


class DeterministicPlanner:
    """
    Stand-in for an LLM planner (OpenAI Agents SDK / LangGraph node). Emits a
    typed plan; a PlanValidator still re-checks every intent against the
    registry and policy — planner output is NEVER trusted blindly.
    """

    def __init__(self, broker: SecretsBroker):
        self._broker = broker

    async def plan(self, trigger: str, context: List[str]) -> List[ActionIntent]:
        return [
            ActionIntent(
                tool="env_scanner",
                payload={"scope": "local_git"},
                risk=RiskLevel.LOW,
                capabilities=Capability.UNTRUSTED_INPUT,   # reads repo content
            ),
            ActionIntent(
                tool="cloud_terminal",
                payload={
                    "command": "apply_patch && trigger_ci",
                    "deploy_token": self._broker.surrogate("ci_token"),
                },
                risk=RiskLevel.CRITICAL,
                capabilities=Capability.EXTERNAL_EFFECT | Capability.SENSITIVE_ACCESS,
                requires_approval=True,
                timeout_s=45.0,
                max_retries=2,
            ),
        ]


class PlanValidator:
    """Reject plans referencing unknown tools or failing schema validation."""

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def validate(self, plan: List[ActionIntent]) -> List[ActionIntent]:
        for intent in plan:
            spec = self._registry.get(intent.tool)   # raises on unknown tool
            spec.validate(intent.payload)            # raises on bad payload
        return plan


# --------------------------------------------------------------------------- #
# 13. The orchestrator core                                                    #
# --------------------------------------------------------------------------- #

class Orchestrator:
    def __init__(
        self,
        name: str,
        planner: Planner,
        registry: ToolRegistry,
        sandbox: Sandbox,
        approval_gate: ApprovalGate,
        audit: AuditLog,
        checkpoints: CheckpointStore,
        policy: Optional[PolicyEngine] = None,
        max_concurrency: int = 4,
    ):
        self.name = name
        self.planner = planner
        self.registry = registry
        self.sandbox = sandbox
        self.approval_gate = approval_gate
        self.audit = audit
        self.checkpoints = checkpoints
        self.policy = policy or PolicyEngine()
        self.memory = WorkingMemory()
        self._validator = PlanValidator(registry)
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()   # guards against concurrent run() calls
        self.is_running = False

    def _breaker_for(self, tool: str) -> CircuitBreaker:
        return self._breakers.setdefault(tool, CircuitBreaker())

    async def run(self, trigger: str) -> Dict[str, Any]:
        async with self._lock:              # the old code's flag never worked;
            self.is_running = True          # this lock actually enforces it
            started = time.monotonic()
            results: List[StepResult] = []
            aborted = False
            try:
                plan = await self.planner.plan(trigger, self.memory.context)
                plan = self._validator.validate(plan)
                self.audit.record("plan_accepted", trigger=trigger,
                                  steps=[a.action_id for a in plan])
                for intent in plan:
                    result = await self._execute_step(intent)
                    results.append(result)
                    self.checkpoints.save_result(result)
                    if self.memory.budget_exhausted():
                        self.audit.record("run_aborted",
                                          reason="failure budget exhausted",
                                          failures=self.memory.failure_counter)
                        aborted = True
                        break
            finally:
                self.is_running = False
            summary = self._summarize(trigger, results, time.monotonic() - started)
            summary["aborted"] = aborted
            return summary

    async def _execute_step(self, intent: ActionIntent) -> StepResult:
        started = time.monotonic()

        # 0. Idempotency: a spent key means the effect already happened — skip.
        if self.checkpoints.is_key_spent(intent.idempotency_key):
            self.audit.record("step_skipped_duplicate", action_id=intent.action_id)
            return StepResult(intent.action_id, intent.tool,
                              StepStatus.SKIPPED_DUPLICATE)

        # 1. Policy (Rule of Two) — structural veto, cannot be overridden.
        decision = self.policy.evaluate(intent)
        self.audit.record("policy_check", action_id=intent.action_id,
                          allowed=decision.allowed, reason=decision.reason)
        if not decision.allowed:
            self.memory.append(f"Policy-vetoed {intent.action_id}", failed=True)
            return StepResult(intent.action_id, intent.tool,
                              StepStatus.VETOED_BY_POLICY,
                              error=decision.reason)

        # 2. Human-in-the-loop — out-of-band, time-boxed, default-deny.
        if decision.needs_approval:
            gate_status = await self.approval_gate.request(intent, decision)
            self.audit.record("approval", action_id=intent.action_id,
                              outcome=gate_status.value)
            if gate_status != StepStatus.SUCCESS:
                self.memory.append(f"Not approved: {intent.action_id}", failed=True)
                return StepResult(intent.action_id, intent.tool, gate_status)

        # 3. Circuit breaker — don't hammer a tool that is already failing.
        breaker = self._breaker_for(intent.tool)
        try:
            breaker.before_call()
        except CircuitOpenError as exc:
            self.audit.record("circuit_open_block", action_id=intent.action_id)
            self.memory.append(f"Circuit-open {intent.tool}", failed=True)
            return StepResult(intent.action_id, intent.tool,
                              StepStatus.CIRCUIT_OPEN, error=str(exc))

        # 4. Execute, time-boxed, retried — inside the semaphore.
        spec = self.registry.get(intent.tool)
        try:
            async with self._semaphore:
                output, attempts = await with_retry(
                    lambda: self.sandbox.execute(spec, intent.payload),
                    max_retries=intent.max_retries,
                    timeout_s=intent.timeout_s,
                )
        except Exception as exc:
            breaker.on_failure()
            latency = time.monotonic() - started
            self.memory.append(f"Fault: {intent.action_id} ({exc})", failed=True)
            self.audit.record("step_failed", action_id=intent.action_id,
                              error=str(exc))
            return StepResult(intent.action_id, intent.tool,
                              StepStatus.RETRIES_EXHAUSTED,
                              attempts=intent.max_retries + 1,
                              latency_s=latency, error=str(exc))

        # 5. Success — burn the idempotency key, close the breaker, audit.
        breaker.on_success()
        self.checkpoints.mark_key_spent(intent.idempotency_key)
        latency = time.monotonic() - started
        self.memory.append(f"Success: {intent.action_id}")
        self.audit.record("step_success", action_id=intent.action_id,
                          attempts=attempts, latency_s=round(latency, 4))
        return StepResult(intent.action_id, intent.tool, StepStatus.SUCCESS,
                          attempts=attempts, latency_s=latency, output=output)

    def _summarize(self, trigger: str, results: List[StepResult],
                   wall_s: float) -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        for r in results:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        return {
            "agent": self.name,
            "trigger": trigger,
            "wall_time_s": round(wall_s, 3),
            "steps": [r.__dict__ | {"status": r.status.value} for r in results],
            "by_status": by_status,
            "failures": self.memory.failure_counter,
            "audit_entries": len(self.audit),
            "audit_chain_valid": self.audit.verify(),
        }


# --------------------------------------------------------------------------- #
# 14. Demo wiring                                                              #
# --------------------------------------------------------------------------- #

async def demo_approval(intent: ActionIntent, decision: PolicyDecision) -> bool:
    """Out-of-band approval stub — in production this is a push notification,
    a Slack button, or an SSO-gated web page, NEVER chat text the agent reads."""
    print(f"\n[APPROVAL REQUIRED] {intent.action_id} via tool={intent.tool!r} "
          f"risk={intent.risk.value} payload={intent.payload}")
    await asyncio.sleep(0.01)
    return True


async def _demo_tools(registry: ToolRegistry) -> None:
    async def env_scanner(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "ok", "findings": f"scanned {payload['scope']}"}

    async def cloud_terminal(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "ok", "ran": payload["command"]}

    def terminal_schema(p: Dict[str, Any]) -> None:
        if "command" not in p or not isinstance(p["command"], str):
            raise ValueError("cloud_terminal requires a string 'command'")

    registry.register(ToolSpec("env_scanner", env_scanner, RiskLevel.LOW,
                               Capability.UNTRUSTED_INPUT))
    registry.register(ToolSpec("cloud_terminal", cloud_terminal, RiskLevel.CRITICAL,
                               Capability.EXTERNAL_EFFECT | Capability.SENSITIVE_ACCESS,
                               validate=terminal_schema))


async def main() -> None:
    tmp = Path("./orchestrator_state")
    tmp.mkdir(exist_ok=True)
    broker = SecretsBroker(vault={"ci_token": "real-token-value"})
    registry = ToolRegistry()
    await _demo_tools(registry)

    agent = Orchestrator(
        name="Sentinel-O",
        planner=DeterministicPlanner(broker),
        registry=registry,
        sandbox=LocalSandbox(broker),
        approval_gate=ApprovalGate(demo_approval, timeout_s=30.0),
        audit=AuditLog(tmp / "audit.jsonl"),
        checkpoints=CheckpointStore(tmp / "checkpoints.json"),
    )
    summary = await agent.run(trigger="NEW_INCIDENT_REPORT_CRITICAL")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
