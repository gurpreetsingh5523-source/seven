"""Tool bridge: routes every model tool call through Sentinel.

The Live model can only *propose* actions. Every call goes:
    PolicyEngine (Rule of Two veto) -> ApprovalGate (time-boxed, default-deny)
    -> LocalSandbox.execute
and every step is written to the hash-chained AuditLog with event names
starting with "live_tool".

`dispatch` NEVER raises — the model must always get a structured answer.

`declarations()` builds the Gemini function declarations with real
descriptions and a parameter schema recovered from agent7.TOOL_HINTS
(e.g. weather_report -> {"city": ...}); unknown parameter types default
to STRING.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from agent7 import TOOL_HINTS
from sentinel_orchestrator import (
    ActionIntent,
    ApprovalGate,
    AuditLog,
    LocalSandbox,
    PolicyEngine,
    StepStatus,
    ToolRegistry,
)

_GEMINI_TYPES = ("STRING", "NUMBER", "INTEGER", "BOOLEAN", "OBJECT", "ARRAY")


def _precheck_folder(args: Dict[str, Any]) -> Optional[str]:
    """file_organizer: refuse folders that do not resolve before asking the user."""
    from capabilities import resolve_folder  # local import: capabilities imports sentinel too
    name = args.get("folder", "")
    if resolve_folder(name) is None:
        from pathlib import Path
        return (f"folder not found: {name} — the user's home folder is {Path.home()}; "
                "ask the user which folder they mean")
    return None


# Tool name -> fn(args) -> error string | None. Evaluated after the policy
# decision and BEFORE any approval is asked, so a call that can never work
# never produces an approval card.
DEFAULT_PRECHECKS: Dict[str, Any] = {
    "file_organizer": _precheck_folder,
}


def _hint_schema(name: str) -> Dict[str, Any]:
    """Recover a Gemini parameter schema from the agent7.TOOL_HINTS payload.

    Hints are example JSON payloads like '{"city": "<city name>"}' — we use
    each top-level key as a property and infer the type from the example
    value (default STRING). Returns {} when there is no usable hint.
    """
    hint = TOOL_HINTS.get(name)
    if not hint:
        return {}
    match = re.search(r"\{.*\}", hint, re.DOTALL)
    if not match:
        return {}
    try:
        example = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(example, dict):
        return {}
    properties: Dict[str, Any] = {key: _prop_schema(value) for key, value in example.items()}
    return {"type": "OBJECT", "properties": properties}


def _prop_schema(value: Any) -> Dict[str, Any]:
    """Gemini schema for one example value. Gemini rejects an ARRAY without `items`
    (close code 1007: "...properties[steps].items: missing field") and an OBJECT
    needs properties, so nested examples are described recursively."""
    if isinstance(value, bool):
        return {"type": "BOOLEAN"}
    if isinstance(value, int):
        return {"type": "INTEGER"}
    if isinstance(value, float):
        return {"type": "NUMBER"}
    if isinstance(value, dict):
        if not value:
            return {"type": "STRING", "description": "JSON object"}
        return {"type": "OBJECT", "properties": {k: _prop_schema(v) for k, v in value.items()}}
    if isinstance(value, list):
        return {"type": "ARRAY", "items": _prop_schema(value[0]) if value else {"type": "STRING"}}
    prop: Dict[str, Any] = {"type": "STRING"}
    if isinstance(value, str) and value.strip("<>") != value:
        prop["description"] = value.strip("<>")
    return prop


def _result_ok(output: Any) -> bool:
    """The wrapped result's own success flag (defaults True when absent)."""
    if isinstance(output, dict) and "ok" in output:
        return bool(output["ok"])
    return True


def _wrap(output: Any) -> Dict[str, Any]:
    """Tool reply for the model: top-level ok follows the tool's own ok, and
    a failing tool's error is lifted up so the model cannot miss it."""
    ok = _result_ok(output)
    reply: Dict[str, Any] = {"ok": ok, "result": output}
    if not ok and isinstance(output, dict) and output.get("error"):
        reply["error"] = str(output["error"])
    return reply


class ToolBridge:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: PolicyEngine,
        sandbox: LocalSandbox,
        audit: AuditLog,
        approval: ApprovalGate,
        *,
        prechecks: Optional[Dict[str, Any]] = None,
        repeat_window_s: float = 60.0,
        max_gated_calls: int = 8,
        gated_window_s: float = 10.0,
        clock: Any = time.monotonic,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._sandbox = sandbox
        self._audit = audit
        self._approval = approval
        self._prechecks = DEFAULT_PRECHECKS if prechecks is None else dict(prechecks)
        # Approval-storm guards (only calls that need approval are tracked):
        self._repeat_window_s = float(repeat_window_s)
        self._max_gated_calls = int(max_gated_calls)
        self._gated_window_s = float(gated_window_s)
        self._clock = clock
        self._waiting: Dict[str, float] = {}        # key -> time the approval request was sent
        self._approved: Dict[str, tuple] = {}       # key -> (time, result)
        self._refused: Dict[str, float] = {}        # key -> time of refusal/timeout
        self._failed: Dict[str, tuple] = {}         # key -> (time, error) of a failed approved run
        self._gate_times: List[float] = []          # times of approval requests actually sent

    @staticmethod
    def _call_key(name: str, args: Dict[str, Any]) -> str:
        """Canonical identity of a call: tool name + args, key order ignored."""
        try:
            canonical = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            canonical = repr(sorted(args.items(), key=lambda kv: kv[0]))
        return f"{name}\x00{canonical}"

    def declarations(self) -> List[Dict[str, Any]]:
        """Gemini function declarations for every registered tool.

        Descriptions carry the agent7.TOOL_HINTS payload hint when one
        exists, and parameters.properties covers every key named in it.
        """
        decls = []
        for spec in self._registry._tools.values():
            hint = TOOL_HINTS.get(spec.name)
            description = (
                f"Agent 7 tool '{spec.name}' "
                f"(risk: {spec.risk.value}, capabilities: {spec.capabilities!r})."
            )
            if hint:
                description += f" Arguments payload: {hint}"
            decls.append(
                {
                    "name": spec.name,
                    "description": description,
                    "parameters": _hint_schema(spec.name)
                    or {"type": "OBJECT", "properties": {}},
                }
            )
        return decls

    async def dispatch(self, name: str, args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run one model tool call through Sentinel. Never raises."""
        args = dict(args or {})
        try:
            spec = self._registry.get(name)
        except KeyError:
            self._audit.record("live_tool_unknown", tool=name)
            return {"ok": False, "error": f"unknown tool: {name!r}"}

        intent = ActionIntent(
            tool=name,
            payload=args,
            risk=spec.risk,
            capabilities=spec.capabilities,
        )
        decision = self._policy.evaluate(intent)
        self._audit.record(
            "live_tool_policy",
            tool=name,
            allowed=decision.allowed,
            reason=decision.reason,
        )
        if not decision.allowed:  # Rule-of-Two veto: cannot be appealed
            return {"ok": False, "error": decision.reason, "vetoed": True}

        precheck = self._prechecks.get(name)
        if precheck is not None:
            try:
                error = precheck(args)
            except Exception as exc:
                error = str(exc)
            if error:
                self._audit.record("live_tool_precheck_failed", tool=name, error=error)
                return {"ok": False, "precheck": True, "error": error}

        # M19c: the registered spec's own validate() also runs BEFORE any
        # approval is asked — a call that can never work (e.g. an app name
        # that does not resolve on this Mac) must not produce an approval
        # card. LocalSandbox.execute runs validate() again at the gate.
        if decision.needs_approval:
            try:
                spec.validate(args)
            except Exception as exc:
                self._audit.record("live_tool_precheck_failed", tool=name, error=str(exc))
                return {"ok": False, "precheck": True, "error": str(exc)}

        if decision.needs_approval:
            # Approval-storm guards: the model may re-issue the exact same
            # call many times a second (seen live 2026-10-08 — 233 app_open
            # calls / 211 approval cards in 3 minutes). Answer repeats from
            # the ledger instead of pestering the user again.
            key = self._call_key(name, args)
            now = self._clock()

            if key in self._waiting:
                self._audit.record("live_tool_duplicate", tool=name, state="waiting")
                return {
                    "ok": False,
                    "duplicate": True,
                    "error": "this action is already waiting for the user's "
                             "approval — do not ask again",
                }

            approved_at = self._approved.get(key)
            if approved_at is not None and now - approved_at[0] < self._repeat_window_s:
                self._audit.record("live_tool_duplicate", tool=name, state="approved")
                output = approved_at[1]
                ok = _result_ok(output)
                # Honest repeat: a repeat of a FAILED action is itself a
                # failure — never report top-level ok=True for it (seen live
                # 2026-10-10: the model believed a failed file_trash worked).
                if not ok:
                    reason = (output.get("error") if isinstance(output, dict) else None) or output
                    return {
                        "ok": False,
                        "duplicate": True,
                        "result": output,
                        "error": f"the earlier approved attempt failed ({reason}) "
                                 "— do not retry it with the same arguments; fix "
                                 "them or ask the user",
                    }
                return {"ok": True, "duplicate": True, "result": output}
            self._approved.pop(key, None)

            refused_at = self._refused.get(key)
            if refused_at is not None and now - refused_at < self._repeat_window_s:
                self._audit.record("live_tool_repeat_refused", tool=name)
                return {
                    "ok": False,
                    "approved": False,
                    "error": "the user said no to this exact action moments ago "
                             "— do not retry it; ask the user what they want instead",
                }
            self._refused.pop(key, None)

            failed = self._failed.get(key)
            if failed is not None and now - failed[0] < self._repeat_window_s:
                self._audit.record("live_tool_repeat_failed", tool=name)
                return {
                    "ok": False,
                    "error": f"this exact action just failed ({failed[1]}) — do not "
                             "retry it with the same arguments; fix them or ask the user",
                }
            self._failed.pop(key, None)

            # The rate limit counts approval requests actually sent to the
            # user (the ledger shortcuts above never reach the gate).
            self._gate_times[:] = [t for t in self._gate_times
                                   if now - t < self._gated_window_s]
            if len(self._gate_times) >= self._max_gated_calls:
                self._audit.record("live_tool_rate_limited", tool=name,
                                   max_calls=self._max_gated_calls,
                                   window_s=self._gated_window_s)
                return {
                    "ok": False,
                    "error": "too many approval requests in a few seconds — "
                             "refusing without asking the user again",
                }

            self._waiting[key] = now
            self._gate_times.append(now)
            try:
                # Keep the key in _waiting until the call has fully finished
                # (approved + executed, refused, or errored) — an identical
                # call arriving while the approved action still executes must
                # get the "already waiting" duplicate answer, not a new card.
                status = await self._approval.request(intent, decision)
                self._audit.record("live_tool_approval", tool=name, outcome=status.value)
                if status is not StepStatus.SUCCESS:
                    self._refused[key] = self._clock()
                    return {
                        "ok": False,
                        "error": f"action not approved ({status.value})",
                        "approved": False,
                    }
                try:
                    output = await self._sandbox.execute(spec, args)
                except Exception as exc:
                    self._failed[key] = (self._clock(), str(exc))
                    self._audit.record("live_tool_error", tool=name, error=str(exc))
                    return {"ok": False, "error": str(exc)}
            finally:
                self._waiting.pop(key, None)

            self._approved[key] = (self._clock(), output)
            self._audit.record("live_tool_result", tool=name, ok=_result_ok(output))
            return _wrap(output)

        try:
            output = await self._sandbox.execute(spec, args)
        except Exception as exc:
            self._audit.record("live_tool_error", tool=name, error=str(exc))
            return {"ok": False, "error": str(exc)}

        self._audit.record("live_tool_result", tool=name, ok=_result_ok(output))
        return _wrap(output)
