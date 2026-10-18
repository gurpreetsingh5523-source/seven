"""Bring what 7 made or opened to the front of the screen.

When 7 creates a document, the user should SEE it, like a person handing it over:
`show_front(path)` opens it in its default app (`open` activates that app). Switched off
with SEVEN_SHOW_RESULTS=0, or per call with "show": false in the tool payload.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

log = logging.getLogger("live7.front")


def enabled() -> bool:
    return (os.environ.get("SEVEN_SHOW_RESULTS") or "1").strip().lower() not in ("0", "false", "no", "off")


async def show_front(path) -> bool:
    """Open `path` in its default app at the front. Never raises."""
    if not enabled():
        return False
    try:
        p = Path(path)
        if not p.exists():
            return False
        proc = await asyncio.create_subprocess_exec(
            "open", str(p), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        return await asyncio.wait_for(proc.wait(), 15) == 0
    except Exception:
        log.exception("show_front failed for %s", path)
        return False
