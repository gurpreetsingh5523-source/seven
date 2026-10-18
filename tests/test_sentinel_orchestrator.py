"""
Test suite for sentinel_orchestrator.py — covers the 10 properties that the
original draft lacked. Run:  python -m pytest test_sentinel_orchestrator.py -v
(plain pytest, no pytest-asyncio needed: each test drives its own loop).
"""

import asyncio
import json
from pathlib import Path

import pytest

from sentinel_orchestrator import (
    ActionIntent, ApprovalGate, AuditLog, Capability, CheckpointStore,
    CircuitBreaker, CircuitOpenError, DeterministicPlanner, LocalSandbox,
    Orchestrator, PolicyEngine, RiskLevel, RunAborted, SecretsBroker,
    StepStatus, ToolRegistry, ToolSpec, WorkingMemory, with_retry,
)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- fixtures #

def make_registry(flaky_tool=None):
    reg = ToolRegistry()

    async def noop(payload):
        return {"ok": True}

    reg.register(ToolSpec("noop", noop, RiskLevel.LOW, Capability.NONE))
    if flaky_tool:
        reg.register(flaky_tool)
    return reg


def make_agent(tmp_path, *, approval=None, planner=None, extra_tools=None,
               failure_budget=3):
    broker = SecretsBroker(vault={"ci_token": "x"})
    reg = make_registry()
    for t in (extra_tools or []):
        reg.register(t)
    audit = AuditLog(tmp_path / "audit.jsonl")
    cp = CheckpointStore(tmp_path / "cp.json")

    async def auto_approve(intent, decision):
        return True

    agent = Orchestrator(
        name="test-agent",
        planner=planner or DeterministicPlanner(broker),
        registry=reg,
        sandbox=LocalSandbox(broker, simulated_latency_s=0.0),
        approval_gate=ApprovalGate(approval or auto_approve, timeout_s=0.3),
        audit=audit,
        checkpoints=cp,
    )
    agent.memory.failure_budget = failure_budget
    return agent, audit, cp


class StaticPlanner:
    def __init__(self, intents):
        self._intents = intents

    async def plan(self, trigger, context):
        return self._intents


# ------------------------------------------------------------------- tests #

def test_rule_of_two_structural_veto(tmp_path):
    """An action holding all 3 dangerous capabilities is vetoed even if a
    human WOULD approve — no approval request is even sent."""
    bad = ActionIntent(
        tool="noop", payload={}, risk=RiskLevel.MEDIUM,
        capabilities=(Capability.UNTRUSTED_INPUT | Capability.SENSITIVE_ACCESS
                      | Capability.EXTERNAL_EFFECT),
        requires_approval=True,
    )
    asked = []

    async def spy_approval(intent, decision):
        asked.append(intent.action_id)
        return True

    agent, audit, _ = make_agent(tmp_path, approval=spy_approval,
                                 planner=StaticPlanner([bad]))
    summary = run(agent.run("t"))
    assert summary["steps"][0]["status"] == StepStatus.VETOED_BY_POLICY.value
    assert asked == [], "approval must not be requested for policy-vetoed steps"


def test_high_risk_requires_human_and_denial_is_honored(tmp_path):
    intent = ActionIntent(tool="noop", payload={}, risk=RiskLevel.HIGH)

    async def deny(intent_, decision):
        return False

    agent, audit, _ = make_agent(tmp_path, approval=deny,
                                 planner=StaticPlanner([intent]))
    summary = run(agent.run("t"))
    assert summary["steps"][0]["status"] == StepStatus.VETOED_BY_HUMAN.value
    outcomes = [e for e in audit._entries if e["event"] == "approval"]
    assert outcomes and outcomes[0]["outcome"] == StepStatus.VETOED_BY_HUMAN.value


def test_critical_forces_approval_even_if_planner_forgets(tmp_path):
    intent = ActionIntent(tool="noop", payload={}, risk=RiskLevel.CRITICAL,
                          requires_approval=False)
    assert intent.requires_approval is True  # fixed at construction


def test_approval_timeout_defaults_to_deny(tmp_path):
    intent = ActionIntent(tool="noop", payload={}, risk=RiskLevel.HIGH)

    async def hanging(intent_, decision):
        await asyncio.sleep(10)  # never answers; gate timeout is 0.3s
        return True

    agent, audit, _ = make_agent(tmp_path, approval=hanging,
                                 planner=StaticPlanner([intent]))
    summary = run(agent.run("t"))
    assert summary["steps"][0]["status"] == StepStatus.APPROVAL_TIMEOUT.value


def test_retry_recovers_from_transient_faults(tmp_path):
    calls = {"n": 0}

    async def flaky(payload):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient 503")
        return {"ok": True}

    spec = ToolSpec("flaky", flaky, RiskLevel.LOW, Capability.NONE)
    agent, _, _ = make_agent(tmp_path, extra_tools=[spec],
                             planner=StaticPlanner([ActionIntent(
                                 tool="flaky", payload={}, risk=RiskLevel.LOW,
                                 max_retries=3)]))
    summary = run(agent.run("t"))
    step = summary["steps"][0]
    assert step["status"] == StepStatus.SUCCESS.value
    assert step["attempts"] == 3 and calls["n"] == 3


def test_retries_exhausted_and_circuit_breaker_opens(tmp_path):
    async def always_fails(payload):
        raise RuntimeError("boom")

    spec = ToolSpec("bad", always_fails, RiskLevel.LOW, Capability.NONE)
    intents = [ActionIntent(tool="bad", payload={}, risk=RiskLevel.LOW,
                            max_retries=1) for _ in range(4)]
    agent, audit, _ = make_agent(tmp_path, extra_tools=[spec],
                                 planner=StaticPlanner(intents),
                                 failure_budget=99)
    summary = run(agent.run("t"))
    statuses = [s["status"] for s in summary["steps"]]
    assert statuses[0] == StepStatus.RETRIES_EXHAUSTED.value
    # breaker threshold = 3 consecutive faults: intent1 (2 attempts) + intent2
    # (2 attempts) open it, so later intents are blocked without executing.
    assert StepStatus.CIRCUIT_OPEN.value in statuses[2:]


def test_step_timeout_is_enforced(tmp_path):
    async def sleepy(payload):
        await asyncio.sleep(10)
        return {"ok": True}

    spec = ToolSpec("sleepy", sleepy, RiskLevel.LOW, Capability.NONE)
    intent = ActionIntent(tool="sleepy", payload={}, risk=RiskLevel.LOW,
                          timeout_s=0.2, max_retries=1)
    agent, _, _ = make_agent(tmp_path, extra_tools=[spec],
                             planner=StaticPlanner([intent]))
    summary = run(agent.run("t"))
    assert summary["steps"][0]["status"] == StepStatus.RETRIES_EXHAUSTED.value
    assert summary["wall_time_s"] < 5  # proves wait_for actually fired


def test_idempotency_key_prevents_double_execution(tmp_path):
    calls = {"n": 0}

    async def counted(payload):
        calls["n"] += 1
        return {"ok": True}

    spec = ToolSpec("counted", counted, RiskLevel.LOW, Capability.NONE)
    key = "idem_FIXED"
    intents = [ActionIntent(tool="counted", payload={}, risk=RiskLevel.LOW,
                            idempotency_key=key) for _ in range(2)]
    agent, _, _ = make_agent(tmp_path, extra_tools=[spec],
                             planner=StaticPlanner(intents))
    summary = run(agent.run("t"))
    statuses = [s["status"] for s in summary["steps"]]
    assert statuses == [StepStatus.SUCCESS.value,
                        StepStatus.SKIPPED_DUPLICATE.value]
    assert calls["n"] == 1


def test_checkpoint_resume_skips_completed_steps(tmp_path):
    broker = SecretsBroker(vault={"ci_token": "x"})
    reg = make_registry()
    calls = {"n": 0}

    async def counted(payload):
        calls["n"] += 1
        return {"ok": True}

    reg.register(ToolSpec("counted", counted, RiskLevel.LOW, Capability.NONE))
    intent = ActionIntent(tool="counted", payload={}, risk=RiskLevel.LOW)
    cp_path = tmp_path / "cp.json"

    async def ok(i, d):
        return True

    a1 = Orchestrator("a1", StaticPlanner([intent]), reg, LocalSandbox(broker, 0),
                      ApprovalGate(ok, 1), AuditLog(), CheckpointStore(cp_path))
    run(a1.run("t"))
    assert calls["n"] == 1

    # Simulate restart with the SAME checkpoint file and same idempotency key.
    a2 = Orchestrator("a2", StaticPlanner([intent]), reg, LocalSandbox(broker, 0),
                      ApprovalGate(ok, 1), AuditLog(), CheckpointStore(cp_path))
    summary = run(a2.run("t"))
    assert calls["n"] == 1, "resume must not re-execute a completed side effect"
    assert summary["steps"][0]["status"] == StepStatus.SKIPPED_DUPLICATE.value


def test_audit_chain_is_tamper_evident(tmp_path):
    audit = AuditLog()
    audit.record("a", x=1)
    audit.record("b", x=2)
    audit.record("c", x=3)
    assert audit.verify() is True
    audit._entries[1]["x"] = 999  # attacker edits history
    assert audit.verify() is False


def test_audit_chain_detects_deleted_entry(tmp_path):
    audit = AuditLog()
    for i in range(4):
        audit.record("evt", i=i)
    del audit._entries[2]
    assert audit.verify() is False


def test_secrets_never_reach_the_planner(tmp_path):
    broker = SecretsBroker(vault={"ci_token": "SUPERSECRET"})
    planner = DeterministicPlanner(broker)
    plan = run(planner.plan("t", []))
    blob = json.dumps([p.payload for p in plan])
    assert "SUPERSECRET" not in blob
    assert "${SECRET:ci_token}" in blob
    # and resolution happens only at the boundary:
    assert broker.resolve(plan[1].payload)["deploy_token"] == "SUPERSECRET"


def test_unknown_tool_in_plan_is_rejected(tmp_path):
    intent = ActionIntent(tool="does_not_exist", payload={}, risk=RiskLevel.LOW)
    agent, _, _ = make_agent(tmp_path, planner=StaticPlanner([intent]))
    with pytest.raises(KeyError):
        run(agent.run("t"))


def test_failure_budget_aborts_run(tmp_path):
    async def fail(payload):
        raise RuntimeError("x")

    spec = ToolSpec("bad", fail, RiskLevel.LOW, Capability.NONE)
    intents = [ActionIntent(tool=f"noop", payload={}, risk=RiskLevel.LOW)
               for _ in range(10)]
    # fail every step by pointing at the failing tool:
    intents = [ActionIntent(tool="bad", payload={}, risk=RiskLevel.LOW,
                            max_retries=0) for _ in range(10)]
    agent, audit, _ = make_agent(tmp_path, extra_tools=[spec],
                                 planner=StaticPlanner(intents),
                                 failure_budget=3)
    summary = run(agent.run("t"))
    # budget 3 -> run aborts after 3 failures, not 10
    assert len(summary["steps"]) == 3
    assert summary["failures"] == 3


def test_working_memory_compaction_keeps_bounds():
    mem = WorkingMemory(max_slots=8)
    for i in range(50):
        mem.append(f"note {i}")
    assert len(mem.context) <= 8
    assert any("compacted" in n for n in mem.context)


def test_concurrent_runs_are_serialized(tmp_path):
    intents = [ActionIntent(tool="noop", payload={}, risk=RiskLevel.LOW)]
    agent, _, _ = make_agent(tmp_path, planner=StaticPlanner(intents))

    async def both():
        return await asyncio.gather(agent.run("t1"), agent.run("t2"))

    s1, s2 = run(both())
    assert s1["audit_chain_valid"] and s2["audit_chain_valid"]
    assert agent.is_running is False


def test_circuit_breaker_half_open_recovery():
    cb = CircuitBreaker(failure_threshold=1, reset_after_s=0.05)
    cb.on_failure()
    with pytest.raises(CircuitOpenError):
        cb.before_call()
    import time; time.sleep(0.06)
    cb.before_call()          # half-open probe allowed
    cb.on_success()
    cb.before_call()          # fully closed again
