"""
SkillForge — self-evolution for Agent 7, done the safe way.

Inspired by the self-evolving-agents literature (Voyager's skill library,
CREATOR's tool synthesis, SkillWeaver) plus one hard rule: never run agent-written code with the user's full
privileges. SkillForge enforces a strict pipeline:

    REQUEST  -> DRAFT (LLM writes a pure function, free-local-brain first)
             -> STATIC CHECK (AST scan: denylisted imports/calls rejected)
             -> VERIFY (run in a SUBPROCESS with sample input, timeout, JSON out)
             -> APPROVAL (out-of-band human gate, deny-by-default)
             -> REGISTER (becomes a ToolSpec; from now on it passes through the
                          same PolicyEngine / PlanValidator / audit as every
                          built-in tool)

Design invariants:
  * The task agent NEVER executes fresh model output directly. Only the Forge
    (a separate component — the "meta-agent") may create tools, and only after
    verification + human approval.
  * Generated skills run as pure functions: payload dict in (JSON on stdin),
    result dict out (JSON on stdout), no imports beyond a small allowlist,
    no network, no filesystem outside their temp dir, hard timeout.
  * Every forge attempt — success or rejection — is written to the audit log.

Stdlib only, consistent with the rest of the project.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from agent7 import HybridRouter, PlanParseError
from sentinel_orchestrator import (
    AuditLog, Capability, RiskLevel, ToolRegistry, ToolSpec,
)

log = logging.getLogger("skill-forge")


class ForgeError(Exception):
    """Any stage of the forge pipeline rejecting a candidate skill."""


# --------------------------------------------------------------------------- #
# What a generated skill looks like                                           #
# --------------------------------------------------------------------------- #

@dataclass
class SkillCandidate:
    name: str
    description: str
    code: str                      # must define: def tool_fn(payload: dict) -> dict
    risk: RiskLevel = RiskLevel.LOW
    capabilities: Capability = Capability.NONE
    sample_payload: Optional[Dict[str, Any]] = None


DRAFTER_SYSTEM = """You write a SINGLE self-contained Python function for an agent's tool.
Hard requirements:
- Output ONLY Python code. No markdown fences, no prose.
- Define exactly one entry point:  def tool_fn(payload: dict) -> dict
- The function must be pure: deterministic, no network, no files outside the
  current directory, no os/subprocess/sys/socket/shutil imports.
- Allowed imports ONLY: math, statistics, json, re, datetime, collections,
  itertools, functools, textwrap, hashlib, base64.
- Return a JSON-serializable dict.
- Keep it under 60 lines."""

# Modules/names a generated skill may never touch. Deliberately conservative:
# anything that reaches the OS, the network, or dynamic code execution is out.
DENY_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests",
    "urllib", "http", "ftplib", "asyncio", "threading", "multiprocessing",
    "ctypes", "importlib", "builtins", "pickle", "marshal", "pty", "signal",
    "webbrowser", "tkinter", "sqlite3", "glob",
}
DENY_CALLS = {"exec", "eval", "compile", "__import__", "open", "input",
              "globals", "locals", "getattr", "setattr", "delattr", "exit", "quit"}


def static_check(code: str) -> None:
    """Parse and AST-scan candidate code. Raise ForgeError on any violation."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ForgeError(f"syntax error in generated skill: {exc}") from exc

    has_entry = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = ({a.name.split(".")[0] for a in node.names}
                    if isinstance(node, ast.Import)
                    else {(node.module or "").split(".")[0]})
            bad = mods & DENY_MODULES
            if bad:
                raise ForgeError(f"denied import(s): {sorted(bad)}")
        elif isinstance(node, ast.Name) and node.id in DENY_CALLS:
            raise ForgeError(f"denied call: {node.id}()")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ForgeError(f"dunder access denied: {node.attr}")
        elif isinstance(node, ast.FunctionDef) and node.name == "tool_fn":
            has_entry = True
    if not has_entry:
        raise ForgeError("no 'def tool_fn(payload: dict)' entry point found")


_RUNNER = r"""
import json, sys
sys.path.insert(0, {dir!r})
import {mod} as m
payload = json.loads(sys.stdin.read())
result = m.tool_fn(payload)
if not isinstance(result, dict):
    raise TypeError("tool_fn must return a dict")
print(json.dumps(result))
"""


async def verify_in_subprocess(code: str, sample_payload: Dict[str, Any],
                               timeout_s: float = 10.0) -> Dict[str, Any]:
    """
    Run the candidate in a SEPARATE Python process with a sample payload.
    A crash, timeout, non-dict return, or non-JSON output = ForgeError.
    (Prototype-grade isolation; production should swap in the microVM sandbox.)
    """
    with tempfile.TemporaryDirectory() as td:
        mod_path = Path(td) / "candidate_skill.py"
        mod_path.write_text(code, encoding="utf-8")
        runner = _RUNNER.format(dir=td, mod="candidate_skill")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", runner,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=td,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(input=json.dumps(sample_payload).encode()),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ForgeError(f"verification timed out after {timeout_s}s")
        if proc.returncode != 0:
            raise ForgeError(f"verification crashed: {err.decode()[:300]}")
        try:
            return json.loads(out.decode())
        except json.JSONDecodeError as exc:
            raise ForgeError(f"non-JSON output: {exc}") from exc


# --------------------------------------------------------------------------- #
# The Forge — meta-agent that turns "I need X" into a registered, tested tool #
# --------------------------------------------------------------------------- #

ApprovalFn = Callable[[SkillCandidate], Awaitable[bool]]


class SkillForge:
    def __init__(self, router: HybridRouter, registry: ToolRegistry,
                 approval: ApprovalFn, audit: Optional[AuditLog] = None):
        self.router = router
        self.registry = registry
        self.approval = approval
        self.audit = audit

    def _log(self, event: str, **fields: Any) -> None:
        if self.audit is not None:  # NOTE: empty AuditLog is falsy via __len__
            self.audit.record(event, **fields)

    async def forge(self, request: str, sample_payload: Optional[Dict[str, Any]] = None,
                    risk: RiskLevel = RiskLevel.LOW,
                    capabilities: Capability = Capability.NONE) -> ToolSpec:
        """Full pipeline. Raises ForgeError at the first failed stage."""
        # 1. DRAFT — the hybrid router decides local-vs-API like everything else
        resp = await self.router.complete(
            DRAFTER_SYSTEM,
            f"Capability request: {request}\n\nPython code only:",
            trigger=request, n_tools=len(self.registry._tools),
        )
        code = resp.text.strip()
        fence = code.find("```")
        if fence != -1:  # tolerate fenced output despite instructions
            inner = code[fence:].split("```")
            code = inner[1] if len(inner) > 1 else code
            code = code.removeprefix("python").strip()

        name = "skill_" + "".join(c if c.isalnum() else "_" for c in request.lower())[:28]
        candidate = SkillCandidate(name=name, description=request, code=code,
                                   risk=risk, capabilities=capabilities,
                                   sample_payload=sample_payload or {})

        # 2. Name collision — never silently overwrite an existing tool
        if name in self.registry._tools:
            self._log("forge_rejected", name=name, reason="name collision")
            raise ForgeError(f"tool name {name!r} already registered")

        # 3. STATIC CHECK
        try:
            static_check(candidate.code)
        except ForgeError as exc:
            self._log("forge_rejected", name=name, reason=str(exc), stage="static")
            raise

        # 4. VERIFY in subprocess
        try:
            result = await verify_in_subprocess(candidate.code, candidate.sample_payload)
        except ForgeError as exc:
            self._log("forge_rejected", name=name, reason=str(exc), stage="verify")
            raise
        self._log("forge_verified", name=name, sample_result=result,
                  drafter=resp.provider)

        # 5. APPROVAL — out-of-band human gate, deny-by-default
        if not await self.approval(candidate):
            self._log("forge_rejected", name=name, reason="human veto", stage="approval")
            raise ForgeError("human rejected the forged skill")

        # 6. REGISTER — executions still run via subprocess, never in-process
        code_snapshot = candidate.code

        async def skill_fn(payload: Dict[str, Any]) -> Dict[str, Any]:
            return await verify_in_subprocess(code_snapshot, payload)

        spec = ToolSpec(name=name, fn=skill_fn, risk=risk,
                        capabilities=capabilities, validate=lambda p: None)
        self.registry.register(spec)
        self._log("forge_registered", name=name)
        log.info("Forged and registered new skill: %s (via %s)", name, resp.provider)
        return spec
