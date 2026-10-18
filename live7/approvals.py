"""Voice + screen approvals for MEDIUM/HIGH actions (SPEC E7, §7.1-7.2).

Security rule (non-negotiable): the MODEL can never approve. An approval is
resolved only by
  (a) the user's own speech — input transcription text ("transcript_in"), or
  (b) a click in the page ({"type":"approval","action_id":...,"approved":bool}).
The model's own output ("Yes, approved") is never fed here, so a prompt-
injected page that makes the model say "yes" cannot authorise anything.

VoiceApprovals plugs into sentinel_orchestrator.ApprovalGate as its callback:
pending action -> approval_needed event to the browser + a spoken question
(via `announce`) -> wait for a voice answer or a click -> approval_closed.
Silence (timeout) is a denial. Every outcome is visible in the event stream
and (via ToolBridge) in the hash-chained audit log.
"""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, List, Optional

SendEvent = Callable[[Dict[str, Any]], Awaitable[None]]
Announce = Callable[[str], Awaitable[None]]

# M19b: common ways of saying yes heard in the real voice test — safe anywhere.
YES_WORDS = frozenset({
    "yes", "yeah", "haan", "han", "haanji", "ok", "okay",
    "kar do", "kholo", "haan ji", "ji haan", "please do", "do it",
    "yes please", "yep", "yup", "bilkul", "zaroor", "sure", "go ahead",
    "go for it", "sounds good", "theek hai", "kar de",
    "ਹਾਂ", "ਹਾਂਜੀ", "ਠੀਕ ਹੈ",
})
NO_WORDS = frozenset({
    "no", "nope", "nahi", "nahin", "na", "cancel", "stop", "rehn de", "wait", "hold on",
    # Negations must count as NO even when they contain a yes phrase.
    "don't", "dont", "do not", "doesn't", "doesnt", "didn't", "didnt",
    "mat", "na karo", "mat karo", "rehn do", "rehne do", "rehne de", "na kar",
    "ਨਹੀਂ", "ਨਾ",
})
# Gurmukhi "ਨਾ" is one Unicode word; romanised "na" needs word boundaries.
_MULTI_YES = [w for w in YES_WORDS if " " in w]
_MULTI_NO = [w for w in NO_WORDS if " " in w]
_SINGLE_YES = [w for w in YES_WORDS if " " not in w]
_SINGLE_NO = [w for w in NO_WORDS if " " not in w]


def _verdict(text: str) -> Optional[bool]:
    """True=yes, False=no, None=unrelated speech. 'no' wins over 'yes'."""
    # Normalise curly apostrophes so "Don’t do it" matches "don't".
    lowered = text.lower().replace("’", "'").replace("‘", "'").replace("`", "'")

    def _has(words) -> bool:
        return any(w in lowered for w in words if " " in w) or any(
            re.search(rf"(?<!\w){re.escape(w)}(?!\w)", lowered)
            for w in words if " " not in w
        )

    said_no = _has(NO_WORDS)
    said_yes = _has(YES_WORDS)
    if said_no:
        return False
    if said_yes:
        return True
    return None


def _question(intent: Any) -> str:
    """Plain-language spoken question ending 'Yes or no?' — no raw tool names."""
    tool = str(getattr(intent, "tool", "") or "")
    payload = dict(getattr(intent, "payload", None) or {})
    name = payload.get("app") or payload.get("name")
    if tool in ("desktop_open_app", "app_open") and name:
        return f"Should I open {name}? Yes or no?"
    if tool == "site_open" and payload.get("url"):
        return f"Should I open {payload['url']}? Yes or no?"
    if tool == "app_quit" and name:
        return f"Should I quit {name}? Yes or no?"
    if tool == "file_trash" and payload.get("path"):
        return f"Should I move {payload['path']} to the Trash? Yes or no?"
    if tool == "reminder_delete" and payload.get("title"):
        if payload.get("all"):
            return (f"Should I delete every reminder called "
                    f"\"{payload['title']}\"? Yes or no?")
        return f"Should I delete the reminder \"{payload['title']}\"? Yes or no?"
    spoken = tool.replace("_", " ").strip() or "this action"
    vals = [str(v) for v in list(payload.values())[:2]]
    detail = f" ({', '.join(vals)})" if vals else ""
    return f"Should I {spoken}{detail}? Yes or no?"


class _Pending:
    def __init__(self, intent: Any) -> None:
        self.intent = intent
        self.future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self.reason: Optional[str] = None


class VoiceApprovals:
    def __init__(
        self,
        send_event: SendEvent,
        announce: Optional[Announce] = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._send = send_event
        self._announce = announce
        self._timeout = timeout_s
        self._pending: "OrderedDict[str, _Pending]" = OrderedDict()
        # One utterance settles at most ONE pending approval: Gemini splits a
        # single spoken "haan" into several input-transcription fragments, and
        # each fragment used to resolve another card. The flag is re-armed by
        # end_utterance() (the relay calls it at turn_complete / interrupted).
        self._utterance_settled = False

    # ---- resolution channels (user only, never the model) ----------------- #

    def resolve(self, action_id: str, approved: bool) -> bool:
        """Click path: the page's Yes/No button. False if no such pending."""
        pend = self._pending.get(action_id)
        if pend is None:
            return False
        pend.reason = "click"
        if not pend.future.done():
            pend.future.set_result(bool(approved))
        return True

    def on_user_transcript(self, text: str) -> bool:
        """Voice path: the user's own speech resolves the OLDEST pending action.

        Returns True when the text settled an approval, False when ignored
        (unrelated speech, nothing pending, or this utterance already settled
        one approval — see end_utterance).
        """
        verdict = _verdict(text or "")
        if verdict is None or not self._pending or self._utterance_settled:
            return False
        _aid, pend = next(iter(self._pending.items()))
        pend.reason = "voice"
        if not pend.future.done():
            pend.future.set_result(verdict)
        self._utterance_settled = True
        return True

    def end_utterance(self) -> None:
        """Start a new utterance: the NEXT voice answer may settle one approval."""
        self._utterance_settled = False

    def pending(self) -> List[str]:
        """Pending action_ids, oldest first."""
        return list(self._pending.keys())

    # ---- sentinel ApprovalGate callback ------------------------------------ #

    async def callback(self, intent: Any, decision: Any) -> bool:
        pend = _Pending(intent)
        self._pending[intent.action_id] = pend
        approved = False
        try:
            await self._send({
                "type": "approval_needed",
                "action_id": intent.action_id,
                "tool": intent.tool,
                "risk": intent.risk.value,
                "payload": dict(intent.payload or {}),
            })
            if self._announce is not None:
                await self._announce(_question(intent))
            try:
                approved = await asyncio.wait_for(pend.future, timeout=self._timeout)
            except asyncio.TimeoutError:
                pend.reason = "timeout"
                approved = False
            return approved
        finally:
            self._pending.pop(intent.action_id, None)
            await self._send({
                "type": "approval_closed",
                "action_id": intent.action_id,
                "approved": approved,
                "reason": pend.reason or "timeout",
            })
