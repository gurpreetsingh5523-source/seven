"""
Tests for agent7.py — the hybrid local+API brain and its equity guarantees.
Run: PYTHONPATH=. python -m pytest test_agent7.py -q
"""

import asyncio
import json

import pytest

from agent7 import (
    Agent7, CostLedger, HybridRouter, LLMPlanner, LLMResponse, OllamaProvider,
    PlanParseError, ProviderUnavailable, extract_json_array,
)
from sentinel_orchestrator import (
    ApprovalGate, AuditLog, Capability, CheckpointStore, LocalSandbox,
    RiskLevel, SecretsBroker, StepStatus, ToolRegistry, ToolSpec,
)


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------- fake LLMs #

class FakeProvider:
    """Scriptable provider: serves queued responses, can be made unavailable."""

    def __init__(self, name, local, responses=None, available=True, cost=0.0):
        self.name = name
        self.local = local
        self.cost_per_1k_in = cost
        self.cost_per_1k_out = cost
        self._responses = list(responses or [])
        self._available = available
        self.calls = 0

    async def available(self):
        return self._available

    async def complete(self, system, prompt, max_tokens=2048):
        self.calls += 1
        if not self._responses:
            raise ProviderUnavailable(f"{self.name}: no scripted response")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item, provider=self.name,
                           input_tokens=100, output_tokens=50,
                           cost_usd=0.0 if self.local else 0.001)


GOOD_PLAN = json.dumps([
    {"tool": "noop", "payload": {}, "risk": "low", "capabilities": []},
])
FENCED_PLAN = "Here is the plan:\n```json\n" + GOOD_PLAN + "\n```\nDone."
PROSE_PLAN = "Sure! " + GOOD_PLAN + " — hope that helps."
BAD_PLAN = "I cannot make plans today."


# ---------------------------------------------------------------- fixtures #

def make_registry():
    reg = ToolRegistry()

    async def noop(payload):
        return {"ok": True}

    reg.register(ToolSpec("noop", noop, RiskLevel.LOW, Capability.NONE))
    return reg


def make_router(local, remote, budget=1.0, ceiling=40):
    return HybridRouter(local=local, remote=remote,
                        ledger=CostLedger(daily_budget_usd=budget),
                        local_complexity_ceiling=ceiling)


# ------------------------------------------------------------------- tests #

def test_extract_raw_json():
    assert len(extract_json_array(GOOD_PLAN)) == 1


def test_extract_fenced_and_prose_json():
    assert len(extract_json_array(FENCED_PLAN)) == 1
    assert len(extract_json_array(PARSE_PLAN := PROSE_PLAN)) == 1


def test_extract_rejects_garbage():
    with pytest.raises(PlanParseError):
        extract_json_array(BAD_PLAN)


def test_router_prefers_local_for_simple_tasks():
    local = FakeProvider("local", True, responses=[GOOD_PLAN])
    remote = FakeProvider("api", False, responses=[GOOD_PLAN])
    router = make_router(local, remote)
    resp = run(router.complete("sys", "p", trigger="scan repo", n_tools=2))
    assert resp.provider == "local" and remote.calls == 0
    assert resp.cost_usd == 0.0


def test_router_escalates_complex_tasks_to_api():
    local = FakeProvider("local", True, responses=[GOOD_PLAN])
    remote = FakeProvider("api", False, responses=[GOOD_PLAN])
    router = make_router(local, remote, ceiling=5)
    complex_trigger = ("analyze repo, patch auth module, run tests, "
                       "deploy to prod and email the 42 stakeholders; "
                       "also rotate keys and update docs")
    resp = run(router.complete("sys", "p", trigger=complex_trigger, n_tools=6))
    assert resp.provider == "api" and local.calls == 0


def test_router_falls_back_to_api_when_local_garbage_or_dead():
    # Local is DOWN entirely:
    local = FakeProvider("local", True, available=False)
    remote = FakeProvider("api", False, responses=[GOOD_PLAN])
    router = make_router(local, remote)
    resp = run(router.complete("sys", "p", trigger="scan repo", n_tools=2))
    assert resp.provider == "api"


def test_budget_exhaustion_forces_local_only_mode():
    local = FakeProvider("local", True, responses=[GOOD_PLAN])
    remote = FakeProvider("api", False, responses=[GOOD_PLAN])
    router = make_router(local, remote, budget=0.0, ceiling=5)  # zero budget
    complex_trigger = "a " * 30  # forces the API-first order
    resp = run(router.complete("sys", "p", trigger=complex_trigger, n_tools=6))
    # API skipped due to budget; local serves instead — agent DEGRADES, not dies
    assert resp.provider == "local"
    assert router.ledger.spent_today == 0.0


def test_ledger_records_only_api_spend():
    ledger = CostLedger(daily_budget_usd=1.0)
    ledger.record(0.0)
    assert ledger.spent_today == 0.0
    ledger.record(0.004)
    assert ledger.spent_today == pytest.approx(0.004)


def test_planner_escalates_once_on_local_garbage():
    local = FakeProvider("local", True, responses=[BAD_PLAN])
    remote = FakeProvider("api", False, responses=[GOOD_PLAN])
    router = make_router(local, remote)
    planner = LLMPlanner(router, make_registry(), SecretsBroker())
    intents = run(planner.plan("scan repo", []))
    assert len(intents) == 1 and intents[0].tool == "noop"
    # attempt 1: local garbage -> attempt 2 re-offers local (free-first policy),
    # local is exhausted -> remote serves. Exactly one paid call.
    assert remote.calls == 1


def test_planner_fails_loudly_when_both_brains_garbage():
    local = FakeProvider("local", True, responses=[BAD_PLAN])
    remote = FakeProvider("api", False, responses=[BAD_PLAN])
    router = make_router(local, remote)
    planner = LLMPlanner(router, make_registry(), SecretsBroker())
    with pytest.raises(PlanParseError):
        run(planner.plan("scan repo", []))


def test_planner_rejects_llm_invented_tools():
    evil = json.dumps([{"tool": "rm_rf_root", "payload": {}, "risk": "low"}])
    local = FakeProvider("local", True, responses=[evil, evil])
    remote = FakeProvider("api", False, responses=[evil])
    router = make_router(local, remote)
    planner = LLMPlanner(router, make_registry(), SecretsBroker())
    with pytest.raises(PlanParseError):
        run(planner.plan("scan repo", []))


def test_planner_marks_external_effect_capability():
    plan = json.dumps([{"tool": "noop", "payload": {}, "risk": "medium",
                        "capabilities": ["external_effect"]}])
    local = FakeProvider("local", True, responses=[plan])
    router = make_router(local, None)
    planner = LLMPlanner(router, make_registry(), SecretsBroker())
    intents = run(planner.plan("scan repo", []))
    assert intents[0].capabilities & Capability.EXTERNAL_EFFECT


def test_end_to_end_offline_run(tmp_path):
    """Full Agent 7 run with NO api key and zero budget — the 'ਗਰੀਬ ਮੋਡ':
    everything works, costs $0, audit chain valid."""
    local = FakeProvider("local", True, responses=[
        json.dumps([
            {"tool": "noop", "payload": {"x": 1}, "risk": "low"},
            {"tool": "noop", "payload": {"x": 2}, "risk": "high"},
        ]),
    ])
    remote = FakeProvider("api", False, available=False)

    async def approve(intent, decision):
        return True

    agent = Agent7(
        registry=make_registry(),
        tools_sandbox=LocalSandbox(SecretsBroker(), simulated_latency_s=0.0),
        approval_callback=approve,
        state_dir=tmp_path,
        daily_budget_usd=0.0,
        local_provider=local,
        remote_provider=remote,
    )
    summary = run(agent.run("NEW_INCIDENT_REPORT_CRITICAL"))
    assert summary["by_status"]["success"] == 2
    assert summary["audit_chain_valid"] is True
    assert summary["brain"]["spent_today_usd"] == 0.0
    assert summary["brain"]["mode"] == "local-only"
    assert remote.calls == 0  # paid brain never touched


def test_end_to_end_hybrid_run_spends_within_budget(tmp_path):
    local = FakeProvider("local", True, responses=[GOOD_PLAN])
    remote = FakeProvider("api", False, responses=[GOOD_PLAN])

    async def approve(intent, decision):
        return True

    agent = Agent7(
        registry=make_registry(),
        tools_sandbox=LocalSandbox(SecretsBroker(), simulated_latency_s=0.0),
        approval_callback=approve,
        state_dir=tmp_path,
        daily_budget_usd=0.01,
        local_provider=local,
        remote_provider=remote,
    )
    summary = run(agent.run("scan repo"))
    assert summary["by_status"]["success"] == 1
    assert summary["brain"]["spent_today_usd"] <= 0.01


def test_ollama_provider_unavailable_without_server():
    p = OllamaProvider(host="http://localhost:9")  # nothing listening
    assert run(p.available()) is False
    with pytest.raises(ProviderUnavailable):
        run(p.complete("s", "p"))
