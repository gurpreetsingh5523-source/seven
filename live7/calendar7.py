"""M14: macOS Calendar for 7 Live — EventKit with an honest access model.

The backend interface is async so tests (and a future Google backend) plug in:
    list_events(start, end) -> [dict(title, start, end, calendar, location)]
    add_event(title, start, end, calendar, location, notes) -> event_id
    delete_event(event_id) -> bool
    access() -> "full" | "write_only" | "denied" | "not_determined" | "restricted"

Honesty rule: calendar_list must FAIL when access is not "full" — an empty
list would mean "you're free", which may be false. On this Mac 7 currently has
write-only access: adding events works, reading does not, and the tool says so
(pointing at System Settings > Privacy & Security > Calendars).
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sentinel_orchestrator import Capability, RiskLevel, ToolRegistry, ToolSpec

log = logging.getLogger("live7.calendar")

_START_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ACCESS_STATES = ("full", "write_only", "denied", "not_determined", "restricted")

_SETTINGS_HINT = ("System Settings > Privacy & Security > Calendars — 7 needs "
                  "Full Calendar Access to READ your events")


def _err(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg}


# --------------------------------------------------------------------------- #
# the real backend                                                             #
# --------------------------------------------------------------------------- #

class EventKitBackend:
    """EventKit via pyobjc. Blocking EventKit calls run in a thread so the
    live loop never stalls. Recurring events are expanded by
    predicateForEventsWithStartDate_endDate_calendars_."""

    def __init__(self) -> None:
        self._store = None

    def _get_store(self):
        if self._store is None:
            import EventKit
            self._store = EventKit.EKEventStore.alloc().init()
        return self._store

    async def access(self) -> str:
        def _status() -> str:
            try:
                import EventKit
                status = EventKit.EKEventStore.authorizationStatusForEntityType_(
                    EventKit.EKEntityTypeEvent)
                full = getattr(EventKit, "EKAuthorizationStatusFullAccess", None)
                write_only = getattr(EventKit,
                                     "EKAuthorizationStatusWriteOnly", None)
                if full is not None and status == full:
                    return "full"
                if write_only is not None and status == write_only:
                    return "write_only"
                if status == EventKit.EKAuthorizationStatusAuthorized:
                    return "full"          # pre-macOS-14 style full access
                if status == EventKit.EKAuthorizationStatusNotDetermined:
                    return "not_determined"
                if status == EventKit.EKAuthorizationStatusRestricted:
                    return "restricted"
                return "denied"
            except Exception as exc:
                log.warning("EventKit access check failed: %s", exc)
                return "denied"
        return await asyncio.to_thread(_status)

    async def list_events(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        def _list() -> List[Dict[str, Any]]:
            from Foundation import NSDate
            store = self._get_store()
            s = NSDate.dateWithTimeIntervalSince1970_(start.timestamp())
            e = NSDate.dateWithTimeIntervalSince1970_(end.timestamp())
            pred = store.predicateForEventsWithStartDate_endDate_calendars_(
                s, e, None)
            events = store.eventsMatchingPredicate_(pred) or []
            out: List[Dict[str, Any]] = []
            for ev in events:
                sid = ev.eventIdentifier()
                out.append({
                    "id": str(sid) if sid else "",
                    "title": str(ev.title() or ""),
                    "start": datetime.fromtimestamp(ev.startDate().timeIntervalSince1970()),
                    "end": datetime.fromtimestamp(ev.endDate().timeIntervalSince1970()),
                    "calendar": str(ev.calendar().title()) if ev.calendar() else "",
                    "location": str(ev.location() or ""),
                })
            return out
        return await asyncio.to_thread(_list)

    async def add_event(self, title: str, start: datetime, end: datetime,
                        calendar: str = "", location: str = "",
                        notes: str = "") -> str:
        def _add() -> str:
            import EventKit
            from Foundation import NSDate
            store = self._get_store()
            ev = EventKit.EKEvent.eventWithEventStore_(store)
            ev.setTitle_(title)
            ev.setStartDate_(NSDate.dateWithTimeIntervalSince1970_(start.timestamp()))
            ev.setEndDate_(NSDate.dateWithTimeIntervalSince1970_(end.timestamp()))
            if location:
                ev.setLocation_(location)
            if notes:
                ev.setNotes_(notes)
            target = None
            if calendar:
                for cal in store.calendarsForEntityType_(EventKit.EKEntityTypeEvent):
                    if str(cal.title()).lower() == calendar.lower():
                        target = cal
                        break
                if target is None:
                    raise ValueError(f"no calendar named {calendar!r}")
            if target is None:
                target = store.defaultCalendarForNewEvents()
            if target is None:
                raise RuntimeError("no writable calendar available")
            ev.setCalendar_(target)
            ok, error = store.saveEvent_span_error_(ev, EventKit.EKSpanThisEvent, None)
            if not ok:
                raise RuntimeError(f"EventKit save failed: {error}")
            return str(ev.eventIdentifier())
        return await asyncio.to_thread(_add)

    async def delete_event(self, event_id: str) -> bool:
        def _delete() -> bool:
            import EventKit
            store = self._get_store()
            ev = store.eventWithIdentifier_(event_id)
            if ev is None:
                return False
            ok, _error = store.removeEvent_span_error_(
                ev, EventKit.EKSpanThisEvent, None)
            return bool(ok)
        return await asyncio.to_thread(_delete)


# --------------------------------------------------------------------------- #
# tools                                                                        #
# --------------------------------------------------------------------------- #

def register_calendar_tools(registry: ToolRegistry, undo=None,
                            backend=None) -> None:
    if backend is None:
        backend = EventKitBackend()

    def push_undo(label: str, undo_fn) -> None:
        if undo is not None:
            undo.push(label, undo_fn)

    async def calendar_list(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                payload = {}
            raw_days = payload.get("days", 1)
            try:
                days = int(raw_days)
            except (TypeError, ValueError):
                days = 1
            days = max(1, min(31, days))
            start_raw = payload.get("start")
            if start_raw in (None, ""):
                start_day = datetime.now().replace(hour=0, minute=0,
                                                   second=0, microsecond=0)
            elif isinstance(start_raw, str) and _DATE_RE.match(start_raw.strip()):
                try:
                    start_day = datetime.strptime(start_raw.strip(), "%Y-%m-%d")
                except ValueError:
                    return _err("start must be a real date like 2026-10-07")
            else:
                return _err("start must look like YYYY-MM-DD")
            end_day = start_day + timedelta(days=days)

            state = await backend.access()
            if state != "full":
                # Never return an empty success: it would mean "you're free",
                # which may be false.
                return _err(f"7 can't READ the calendar (access: {state}). "
                            f"Allow it in {_SETTINGS_HINT}.")
            raw_events = await backend.list_events(start_day, end_day)
            events: List[Dict[str, Any]] = []
            for ev in raw_events or []:
                if not isinstance(ev, dict):
                    continue
                item: Dict[str, Any] = {"title": str(ev.get("title") or "")}
                for key in ("start", "end"):
                    value = ev.get(key)
                    if isinstance(value, datetime):
                        item[key] = value.strftime("%Y-%m-%d %H:%M")
                    elif value:
                        item[key] = str(value)
                for key in ("calendar", "location", "id"):
                    if ev.get(key):
                        item[key] = str(ev[key])
                events.append(item)
            events.sort(key=lambda e: e.get("start", ""))
            return {"ok": True, "events": events,
                    "start": start_day.strftime("%Y-%m-%d"), "days": days}
        except Exception as exc:
            return _err(str(exc))

    async def calendar_add(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                return _err("payload must be an object")
            title = payload.get("title")
            if not isinstance(title, str) or not title.strip():
                return _err("title must be a non-empty string")
            start_raw = payload.get("start")
            if not isinstance(start_raw, str) or not _START_RE.match(start_raw.strip()):
                return _err("start must look like YYYY-MM-DD HH:MM")
            try:
                start = datetime.strptime(start_raw.strip(), "%Y-%m-%d %H:%M")
            except ValueError:
                return _err("start must be a real date/time like "
                            "2026-10-07 15:00")
            end_raw = payload.get("end")
            if end_raw in (None, ""):
                end = start + timedelta(minutes=60)
            elif isinstance(end_raw, str) and _START_RE.match(end_raw.strip()):
                try:
                    end = datetime.strptime(end_raw.strip(), "%Y-%m-%d %H:%M")
                except ValueError:
                    return _err("end must be a real date/time")
            else:
                return _err("end must look like YYYY-MM-DD HH:MM")
            if end <= start:
                return _err("end must be after start")
            calendar = payload.get("calendar")
            location = payload.get("location")
            notes = payload.get("notes")
            event_id = await backend.add_event(
                title.strip(), start, end,
                calendar.strip() if isinstance(calendar, str) else "",
                location.strip() if isinstance(location, str) else "",
                notes if isinstance(notes, str) else "")
            event_id = str(event_id)

            async def _undo_add(eid=event_id, t=title.strip()):
                gone = await backend.delete_event(eid)
                return {"deleted_event": eid, "ok": bool(gone)}
            push_undo(f"calendar_add {title.strip()}", _undo_add)
            return {"ok": True, "id": event_id}
        except Exception as exc:
            return _err(str(exc))

    registry.register(ToolSpec("calendar_list", calendar_list, RiskLevel.LOW,
                               Capability.SENSITIVE_ACCESS))
    registry.register(ToolSpec("calendar_add", calendar_add, RiskLevel.MEDIUM,
                               Capability.NONE))
