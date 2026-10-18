"""
Tests for skill_forge.py — self-evolution with seatbelts.
Run: PYTHONPATH=. python -m pytest test_skill_forge.py -q
"""

import asyncio
import json

import pytest

from agent7 import CostLedger, HybridRouter, LLMResponse, ProviderUnavailable
from skill_forge import ForgeError, SkillForge, static_check, verify_in_subprocess
from sentinel_orchestrator import (
    AuditLog, Capability, RiskLevel, StepStatus, ToolRegistry, ToolSpec,
)

def run(coro):
    return asyncio.run(coro)


GOOD_CODE = '''
def tool_fn(payload: dict) -> dict:
    words = payload.get("text", "").split()
    return {"word_count": len(words)}
'''

EVIL_OS = 'import os\ndef tool_fn(payload):\n    os.system("rm -rf /")\n    return {}'
EVIL_EVAL = 'def tool_fn(payload):\n    return {"x": eval(payload["expr"])}'
EVIL_DUNDER = 'def tool_fn(payload):\n    return {"x": ().__class__.__base__}'
NO_ENTRY = 'def helper(x):\n    return x'
BROKEN_RUNTIME = 'def tool_fn(payload):\n    return {"x": 1/0}'
NON_DICT = 'def tool_fn(payload):\n    return 42'


class FakeRouter:
    def __init__(self, code, provider="local"):
        self._code = code
        self.provider = provider
        self.calls = 0

    async def complete(self, system, prompt, *, trigger, n_tools, max_tokens=2048):
        self.calls += 1
        return LLMResponse(text=self._code, provider=self.provider)


def make_forge(code, approve=True, audit=None):
    reg = ToolRegistry()

    async def approval(candidate):
        return approve

    return SkillForge(FakeRouter(code), reg, approval, audit), reg


# ------------------------------------------------------------------- tests #

def test_static_check_accepts_pure_function():
    static_check(GOOD_CODE)  # must not raise


def test_static_check_rejects_os_system():
    with pytest.raises(ForgeError, match="denied import"):
        static_check(EVIL_OS)


def test_static_check_rejects_eval():
    with pytest.raises(ForgeError, match="denied call"):
        static_check(EVIL_EVAL)


def test_static_check_rejects_dunder_escape():
    with pytest.raises(ForgeError, match="dunder"):
        static_check(EVIL_DUNDER)


def test_static_check_requires_entry_point():
    with pytest.raises(ForgeError, match="tool_fn"):
        static_check(NO_ENTRY)


def test_subprocess_verify_runs_pure_code():
    out = run(verify_in_subprocess(GOOD_CODE, {"text": "sat sri akal duniya"}))
    assert out == {"word_count": 4}


def test_subprocess_verify_rejects_crashing_code():
    with pytest.raises(ForgeError, match="crashed"):
        run(verify_in_subprocess(BROKEN_RUNTIME, {}))


def test_subprocess_verify_rejects_non_dict_output():
    with pytest.raises(ForgeError, match="crashed|non-JSON"):
        run(verify_in_subprocess(NON_DICT, {}))


def test_forge_end_to_end_registers_working_tool():
    forge, reg = make_forge(GOOD_CODE)
    spec = run(forge.forge("count words in text", sample_payload={"text": "a b c"}))
    assert spec.name in reg._tools
    result = run(reg.get(spec.name).fn({"text": "ik do tin"}))
    assert result == {"word_count": 3}


def test_forge_rejects_malicious_code_before_approval():
    asked = []

    async def spy(candidate):
        asked.append(candidate.name)
        return True

    reg = ToolRegistry()
    forge = SkillForge(FakeRouter(EVIL_OS), reg, spy, AuditLog())
    with pytest.raises(ForgeError, match="denied import"):
        run(forge.forge("wipe the disk"))
    assert asked == []          # approval never even reached
    assert reg._tools == {}     # nothing registered


def test_forge_respects_human_veto():
    forge, reg = make_forge(GOOD_CODE, approve=False)
    with pytest.raises(ForgeError, match="human"):
        run(forge.forge("count words", sample_payload={"text": "x"}))
    assert reg._tools == {}


def test_forge_rejects_name_collision():
    forge, reg = make_forge(GOOD_CODE)
    run(forge.forge("count words in text", sample_payload={"text": "a"}))
    with pytest.raises(ForgeError, match="already registered"):
        run(forge.forge("count words in text", sample_payload={"text": "a"}))


def test_forge_audit_trail_covers_success_and_rejection():
    audit = AuditLog()
    forge, reg = make_forge(GOOD_CODE, audit=audit)
    run(forge.forge("count words in text", sample_payload={"text": "a b"}))
    bad_forge, _ = make_forge(EVIL_EVAL, audit=audit)
    with pytest.raises(ForgeError):
        run(bad_forge.forge("evaluate expressions"))
    events = [e["event"] for e in audit._entries]
    assert "forge_verified" in events and "forge_registered" in events
    assert "forge_rejected" in events
    assert audit.verify() is True


def test_forged_skill_usable_inside_agent7_run(tmp_path):
    """Integration: forge a skill, then a real Agent 7 run plans with it and
    executes it through the full policy/audit/checkpoint pipeline."""
    from agent7 import Agent7
    from agent7 import HybridRouter as HR
    from sentinel_orchestrator import LocalSandbox, SecretsBroker

    async def approve(intent, decision):
        return True

    async def forge_approve(candidate):
        return True

    reg = ToolRegistry()

    class SeqRouter:
        """1st call: skill code (forge). 2nd call: plan using the new skill."""
        def __init__(self):
            self.calls = 0
        async def complete(self, system, prompt, *, trigger, n_tools, max_tokens=2048):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(text=GOOD_CODE, provider="local")
            return LLMResponse(text=json.dumps([{
                "tool": "skill_count_words_in_text", "payload": {"text": "a b c"},
                "risk": "low", "capabilities": []}]), provider="local")

    router = SeqRouter()
    forge = SkillForge(router, reg, forge_approve, AuditLog())
    run(forge.forge("count words in text", sample_payload={"text": "a"}))

    agent = Agent7(registry=reg,
                   tools_sandbox=LocalSandbox(SecretsBroker(), simulated_latency_s=0.0),
                   approval_callback=approve, state_dir=tmp_path,
                   daily_budget_usd=0.0,
                   local_provider=None, remote_provider=None)
    # swap in our scripted router behind the agent's planner
    agent.router = router
    agent.core.planner.router = router
    summary = run(agent.run("count words in my note"))
    assert summary["steps"][0]["status"] == StepStatus.SUCCESS.value
    assert summary["steps"][0]["output"] == {"word_count": 3}
    assert summary["audit_chain_valid"] is True
