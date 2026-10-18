"""Tests for the 2026-10-05 additions: loopback-only local LLM provider,
planner tool-payload hints, and the local STT decoder's error handling."""

import asyncio

import pytest

from agent7 import TOOL_HINTS, LLMPlanner, LocalOpenAICompatProvider
from sentinel_orchestrator import Capability, RiskLevel, ToolRegistry, ToolSpec
from stt7 import STTError, decode_audio


def test_local_provider_refuses_non_loopback_hosts():
    for host in ("http://api.openai.com", "http://192.168.0.10:8095", "http://example.com:8095"):
        with pytest.raises(ValueError):
            LocalOpenAICompatProvider(host=host)
    for host in ("http://127.0.0.1:8095", "http://localhost:8095"):
        LocalOpenAICompatProvider(host=host)


def test_local_provider_is_free_and_local():
    p = LocalOpenAICompatProvider()
    assert p.local is True and p.cost_per_1k_in == 0 and p.cost_per_1k_out == 0


def test_local_provider_unavailable_when_nothing_listens():
    p = LocalOpenAICompatProvider(host="http://127.0.0.1:1")  # nothing on port 1
    assert asyncio.run(p.available()) is False


def test_planner_registry_description_includes_payload_hints():
    async def noop(_):
        return {}

    reg = ToolRegistry()
    reg.register(ToolSpec("weather_report", noop, RiskLevel.LOW, Capability.NONE))
    reg.register(ToolSpec("no_hint_tool", noop, RiskLevel.LOW, Capability.NONE))
    planner = LLMPlanner.__new__(LLMPlanner)
    planner.registry = reg
    text = planner._describe_registry()
    assert '"city"' in text                      # the hint that fixed weather_report {}
    assert "no_hint_tool: risk=low" in text      # tools without hints still listed
    assert "weather_report" in TOOL_HINTS and "chat_reply" in TOOL_HINTS


@pytest.mark.parametrize("data", [b"", b"not audio at all"])
def test_decode_audio_rejects_bad_input(data):
    with pytest.raises(STTError):
        decode_audio(data)


def test_decode_audio_rejects_oversized_input():
    with pytest.raises(STTError):
        decode_audio(b"\0" * (16 * 1024 * 1024))
