"""Hermes bridge: delegate real work to the user's own Hermes agent (SPEC E6, §5, §7.3).

`ask_hermes(task)` starts a background job running EXACTLY
    [hermes_bin, "chat", "-q", task, "-Q"]
— Hermes' own guardrails stay on; we never pass "-z", "--oneshot" or "--yolo".
The default runner is an asyncio subprocess with cwd = the user's home
directory, killed on timeout. Each job is one JSON file in jobs_dir whose
"status" moves running -> done | error | timeout, with "reply" = stdout minus
any "session_id:" lines. When a job finishes, the on_done callback is awaited
exactly once (the server wires this to LiveRelay.announce_hermes so 7 speaks
the result without being asked). `hermes_result` reads the latest job files,
newest first.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sentinel_orchestrator import Capability, RiskLevel, ToolRegistry, ToolSpec

log = logging.getLogger("live7.hermes")

Runner = Callable[[List[str], float], Awaitable[Tuple[int, str, str]]]
OnDone = Callable[[Dict[str, Any]], Awaitable[None]]


async def _default_runner(argv: List[str], timeout_s: float) -> Tuple[int, str, str]:
    """Run the Hermes CLI as a subprocess from the home directory.

    Killed (not just abandoned) if it overruns timeout_s.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(Path.home()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        raise
    return (
        proc.returncode if proc.returncode is not None else -1,
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


def _clean_reply(stdout: str) -> str:
    """stdout without the CLI's "session_id:" bookkeeping lines, stripped."""
    lines = [
        line for line in stdout.splitlines()
        if not line.strip().lower().startswith("session_id:")
    ]
    return "\n".join(lines).strip()


class HermesBridge:
    """Starts Hermes jobs in the background and tracks them as JSON files."""

    def __init__(
        self,
        jobs_dir: Any,
        runner: Optional[Runner] = None,
        hermes_bin: str = "hermes",
        timeout_s: float = 600,
        on_done: Optional[OnDone] = None,
        on_start: Optional[OnDone] = None,
    ) -> None:
        self._jobs_dir = Path(jobs_dir)
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._runner: Runner = runner or _default_runner
        self._hermes_bin = hermes_bin
        self._timeout_s = timeout_s
        self._on_done = on_done
        self._on_start = on_start
        self._tasks: set = set()

    # ------------------------------------------------------------------ #

    async def ask(self, task: str) -> Dict[str, Any]:
        """Start a Hermes job; returns in well under a second."""
        task = (task or "").strip()
        if not task:
            return {"ok": False, "error": "empty task"}
        job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job: Dict[str, Any] = {
            "job_id": job_id,
            "task": task,
            "status": "running",
            "reply": "",
            "started_at": time.time(),
            "finished_at": None,
        }
        self._write(job)
        if self._on_start is not None:
            # Timeline (M12): the page shows the job as running right away.
            try:
                await self._on_start(dict(job))
            except Exception:
                log.exception("on_start failed for job %s", job["job_id"])
        task_obj = asyncio.create_task(self._run(job))
        self._tasks.add(task_obj)
        task_obj.add_done_callback(self._tasks.discard)
        return {"ok": True, "status": "started", "job_id": job_id}

    def result(self, count: int = 1) -> Dict[str, Any]:
        """Latest job files, newest first (finished jobs before running)."""
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = 1
        count = max(1, count)
        jobs: List[Dict[str, Any]] = []
        for path in self._jobs_dir.glob("*.json"):
            try:
                jobs.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        jobs.sort(
            key=lambda j: (j.get("finished_at") or 0.0, j.get("started_at") or 0.0),
            reverse=True,
        )
        return {"ok": True, "jobs": jobs[:count]}

    async def wait_idle(self) -> None:
        """Await every running job (tests / shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ------------------------------------------------------------------ #

    def _path(self, job_id: str) -> Path:
        return self._jobs_dir / f"{job_id}.json"

    def _write(self, job: Dict[str, Any]) -> None:
        try:
            self._path(job["job_id"]).write_text(
                json.dumps(job, ensure_ascii=False, indent=2)
            )
        except OSError:
            log.exception("could not write job file %s", job.get("job_id"))

    async def _run(self, job: Dict[str, Any]) -> None:
        # Hermes' guardrails stay on: chat -q TASK -Q, and nothing else.
        argv = [self._hermes_bin, "chat", "-q", job["task"], "-Q"]
        job["argv"] = argv
        try:
            code, stdout, stderr = await self._runner(argv, self._timeout_s)
            if code == 0:
                job["status"] = "done"
                job["reply"] = _clean_reply(stdout)
            else:
                job["status"] = "error"
                job["reply"] = _clean_reply(stdout)
                job["error"] = (stderr or "").strip() or f"exit code {code}"
        except asyncio.TimeoutError:
            job["status"] = "timeout"
            job["error"] = f"exceeded timeout of {self._timeout_s}s"
        except Exception as exc:  # runner itself blew up (e.g. hermes not installed)
            job["status"] = "error"
            job["error"] = str(exc)
        job["finished_at"] = time.time()
        self._write(job)
        if self._on_done is not None:
            try:
                await self._on_done(dict(job))
            except Exception:
                log.exception("on_done failed for job %s", job["job_id"])


def register_hermes_tools(registry: ToolRegistry, bridge: HermesBridge) -> None:
    """Expose ask_hermes / hermes_result as LOW-risk tools in the registry."""

    async def ask_fn(payload: Dict[str, Any]) -> Dict[str, Any]:
        return await bridge.ask(str(payload.get("task", "")))

    async def result_fn(payload: Dict[str, Any]) -> Dict[str, Any]:
        return bridge.result(payload.get("count", 1))

    registry.register(ToolSpec(
        name="ask_hermes", fn=ask_fn, risk=RiskLevel.LOW,
        capabilities=Capability.NONE,
    ))
    registry.register(ToolSpec(
        name="hermes_result", fn=result_fn, risk=RiskLevel.LOW,
        capabilities=Capability.NONE,
    ))
