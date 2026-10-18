"""M20: a local text log of every 7 Live conversation.

Until now 7 kept no text of what was said, so failures (a reminder that
could not be deleted, a reply in the wrong language) could not be checked
afterwards. ChatLog appends one JSON line per event to
~/.seven/chats/chat-YYYY-MM-DD.jsonl (local date):

    {"ts": 1791619103.0, "kind": "user"|"seven"|"typed"|"tool", ...}

Privacy: files stay on this Mac — the directory is mode 0700, each file
0600, and nothing is ever sent anywhere. Days older than keep_days are
removed on construction. write() never raises: a logging problem must
never break a conversation.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

__all__ = ["ChatLog"]


class ChatLog:
    def __init__(self, directory: Path, clock: Callable[[], float] = time.time,
                 keep_days: int = 14) -> None:
        self._dir = Path(directory)
        self._clock = clock
        self._keep_days = int(keep_days)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self._dir, 0o700)
        except OSError:
            pass
        self._prune()

    def _prune(self) -> None:
        """Remove chat-YYYY-MM-DD.jsonl files older than keep_days (by name)."""
        try:
            cutoff = datetime.date.fromtimestamp(self._clock()) - datetime.timedelta(
                days=self._keep_days)
            for f in self._dir.glob("chat-*.jsonl"):
                try:
                    day = datetime.date.fromisoformat(f.name[len("chat-"):-len(".jsonl")])
                except ValueError:
                    continue
                if day < cutoff:
                    try:
                        f.unlink()
                    except OSError:
                        pass
        except Exception:
            pass

    def _path_for(self, ts: float) -> Path:
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        return self._dir / f"chat-{day}.jsonl"

    def write(self, kind: str, **fields: Any) -> Optional[Path]:
        """Append one JSON line; returns the file path or None. Never raises."""
        try:
            ts = float(self._clock())
            path = self._path_for(ts)
            record = {"ts": ts, "kind": str(kind), **fields}
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
            return path
        except Exception:
            return None
