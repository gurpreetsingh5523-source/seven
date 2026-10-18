"""
undo7.py — "7" ਦਾ undo stack (undo-first)।

ਸਿਧਾਂਤ:
  * ਜੋ ਕੰਮ ਉਲਟਾਇਆ ਜਾ ਸਕਦਾ ਹੈ → ਤੁਰੰਤ ਕਰੋ + inverse action stack 'ਤੇ ਰੱਖੋ।
    "undo" ਬੋਲੋ ਤਾਂ ਵਾਪਸ। ਸਵਾਲ ਪੁੱਛਣ ਤੋਂ ਤੇਜ਼।
  * ਜੋ ਨਹੀਂ ਉਲਟਾਇਆ ਜਾ ਸਕਦਾ → ਉਹੀ ਮਨੁੱਖੀ ਮਨਜ਼ੂਰੀ (orchestrator ਦਾ gate)।

ਸਾਡਾ ਜੋੜ: ਹਰ undo entry audit log ਵਿੱਚ ਜਾਂਦੀ ਹੈ (undo ਕਰਨਾ ਵੀ audited ਹੈ)।
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class UndoEntry:
    label: str
    undo_fn: Callable[[], Awaitable[Dict[str, Any]]]
    ts: float = field(default_factory=time.time)


class UndoStack:
    """single-user local stack — ਆਖ਼ਰੀ N ਕੰਮ ਵਾਪਸ ਕਰਨ ਯੋਗ।"""

    def __init__(self, capacity: int = 50, audit=None):
        self._stack: List[UndoEntry] = []
        self.capacity = capacity
        self.audit = audit  # sentinel AuditLog (optional)

    def push(self, label: str, undo_fn) -> None:
        self._stack.append(UndoEntry(label, undo_fn))
        if len(self._stack) > self.capacity:
            self._stack.pop(0)

    async def undo_last(self) -> Dict[str, Any]:
        if not self._stack:
            return {"error": "undo ਲਈ ਕੁਝ ਨਹੀਂ — stack ਖਾਲੀ"}
        entry = self._stack.pop()
        try:
            result = await entry.undo_fn()
            if self.audit is not None:
                self.audit.record("undo_performed", label=entry.label)
            return {"undone": entry.label, "result": result}
        except Exception as exc:
            if self.audit is not None:
                self.audit.record("undo_failed", label=entry.label,
                                  error=str(exc))
            return {"error": f"undo ਫੇਲ ({entry.label}): {exc}"}

    def pending(self) -> List[str]:
        return [e.label for e in self._stack]


# --------------------------------------------------------------------------- #
# ਤਿਆਰ-ਬਣੇ inverse actions                                                     #
# --------------------------------------------------------------------------- #

def make_move_undo(src: Path, dst: Path):
    """ਫਾਈਲ move ਦਾ ਉਲਟਾ — ਵਾਪਸ ਆਪਣੀ ਥਾਂ।"""
    async def _undo() -> Dict[str, Any]:
        if not dst.exists():
            raise FileNotFoundError(f"{dst} ਹੁਣ ਮੌਜੂਦ ਨਹੀਂ")
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst), str(src))
        return {"restored": str(src)}
    return _undo


def make_write_undo(path: Path, previous_content: Optional[str]):
    """ਫਾਈਲ ਲਿਖਣ ਦਾ ਉਲਟਾ — ਪੁਰਾਣਾ content ਵਾਪਸ (ਜਾਂ ਡਿਲੀਟ ਜੇ ਪਹਿਲਾਂ ਨਹੀਂ ਸੀ)।"""
    async def _undo() -> Dict[str, Any]:
        if previous_content is None:
            if path.exists():
                path.unlink()
            return {"deleted": str(path)}
        path.write_text(previous_content, encoding="utf-8")
        return {"restored_content": str(path)}
    return _undo
