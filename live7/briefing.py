"""M14: a real daily briefing — built ONLY from what the sources returned.

NEVER INVENT: the spoken text contains the date, then each section from real
items only. A source that is missing or raised is named as unavailable; an
empty calendar from a working source says "no events today"; never more than
5 news headlines; nothing is filled in from memory.
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sentinel_orchestrator import Capability, RiskLevel, ToolRegistry, ToolSpec

log = logging.getLogger("live7.briefing")

NEWS_FEEDS = {
    "world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "india": "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
}

MAX_HEADLINES = 5
_FETCH_TIMEOUT_S = 10.0


# --------------------------------------------------------------------------- #
# news                                                                         #
# --------------------------------------------------------------------------- #

async def _default_fetch(url: str) -> str:
    import httpx
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


async def fetch_news(limit: int = 5, feed: str = "world",
                     fetch: Optional[Callable[[str], Any]] = None
                     ) -> List[Dict[str, str]]:
    """Fixed BBC feeds only (no arbitrary URLs). The RSS document is refused
    if it carries a DOCTYPE or ENTITY declaration (XXE class)."""
    if feed not in NEWS_FEEDS:
        raise ValueError(f"unknown news feed {feed!r}; known: {sorted(NEWS_FEEDS)}")
    fetch = fetch or _default_fetch
    raw = await fetch(NEWS_FEEDS[feed])
    if not isinstance(raw, str):
        raise ValueError("news fetch did not return text")
    if "<!DOCTYPE" in raw or "<!ENTITY" in raw:
        raise ValueError("refusing RSS with DOCTYPE/ENTITY declarations")
    root = ET.fromstring(raw)
    channel_title = root.findtext("channel/title") or "BBC News"
    source = channel_title.strip() or "BBC News"
    items: List[Dict[str, str]] = []
    for item in root.findall("channel/item")[:max(0, int(limit))]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if title:
            items.append({"title": title, "source": source, "link": link})
    return items


# --------------------------------------------------------------------------- #
# briefing                                                                     #
# --------------------------------------------------------------------------- #

async def _call_source(fn: Callable, *args) -> Any:
    return await fn(*args)


async def build_briefing(now: datetime, sources: Dict[str, Callable]) -> Dict[str, Any]:
    """sources: async callables, ANY may be missing or raise:
        "calendar"(now)   -> list of events (dicts with "title", maybe "start")
        "reminders"()     -> list of {"name", "due"}
        "weather"()       -> dict with "summary" (str)
        "news"()          -> list of {"title", "source"}
    """
    if not isinstance(sources, dict):
        sources = {}
    date_str = ""
    try:
        date_str = now.strftime("%A %-d %B %Y")
    except (ValueError, AttributeError):
        date_str = now.strftime("%A %d %B %Y").replace(" 0", " ", 1)

    sections: Dict[str, Dict[str, Any]] = {}
    lines: List[str] = [f"Here is your day for {date_str}."]

    async def calendar_items() -> List[Dict[str, Any]]:
        fn = sources.get("calendar")
        if fn is None:
            raise KeyError("no calendar source")
        events = await _call_source(fn, now)
        if events is None:
            return []
        if not isinstance(events, list):
            raise TypeError("calendar source did not return a list")
        return events

    async def reminder_items() -> List[Dict[str, Any]]:
        fn = sources.get("reminders")
        if fn is None:
            raise KeyError("no reminders source")
        items = await _call_source(fn)
        if items is None:
            return []
        if not isinstance(items, list):
            raise TypeError("reminders source did not return a list")
        return items

    async def weather_summary() -> str:
        fn = sources.get("weather")
        if fn is None:
            raise KeyError("no weather source")
        data = await _call_source(fn)
        if not isinstance(data, dict) or not isinstance(data.get("summary"), str) \
                or not data["summary"].strip():
            raise TypeError("weather source returned no summary")
        return data["summary"].strip()

    async def news_items() -> List[Dict[str, Any]]:
        fn = sources.get("news")
        if fn is None:
            raise KeyError("no news source")
        items = await _call_source(fn)
        if items is None:
            return []
        if not isinstance(items, list):
            raise TypeError("news source did not return a list")
        return items[:MAX_HEADLINES]

    results = await asyncio.gather(
        calendar_items(), reminder_items(), weather_summary(), news_items(),
        return_exceptions=True)
    (events, reminders, weather, headlines) = results

    # ---- calendar ----
    if isinstance(events, BaseException):
        reason = ("no calendar source" if isinstance(events, KeyError)
                  else str(events))
        sections["calendar"] = {"ok": False, "error": reason}
        lines.append(f"Calendar: unavailable — {reason}.")
    else:
        sections["calendar"] = {"ok": True, "items": events}
        if not events:
            lines.append("Calendar: no events today.")
        else:
            bits = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                title = str(ev.get("title") or "").strip()
                if not title:
                    continue
                start = ev.get("start")
                when = ""
                if isinstance(start, datetime):
                    when = start.strftime(" at %H:%M")
                elif isinstance(start, str) and len(start) >= 16:
                    when = f" at {start[-5:]}"
                bits.append(f"{title}{when}")
            if bits:
                lines.append("Calendar: " + "; ".join(bits) + ".")
            else:
                lines.append("Calendar: no events today.")

    # ---- reminders ----
    if isinstance(reminders, BaseException):
        reason = ("no reminders source" if isinstance(reminders, KeyError)
                  else str(reminders))
        sections["reminders"] = {"ok": False, "error": reason}
        lines.append(f"Reminders: unavailable — {reason}.")
    else:
        sections["reminders"] = {"ok": True, "items": reminders}
        names = [str(r.get("name") or "").strip()
                 for r in reminders if isinstance(r, dict)]
        names = [n for n in names if n]
        if names:
            lines.append("Reminders: " + "; ".join(names[:MAX_HEADLINES]) + ".")
        else:
            lines.append("Reminders: nothing open.")

    # ---- weather ----
    if isinstance(weather, BaseException):
        reason = ("no weather source" if isinstance(weather, KeyError)
                  else str(weather))
        sections["weather"] = {"ok": False, "error": reason}
        lines.append(f"Weather: unavailable — {reason}.")
    else:
        sections["weather"] = {"ok": True, "items": [weather]}
        lines.append(f"Weather: {weather}.")

    # ---- news ----
    if isinstance(headlines, BaseException):
        reason = ("no news source" if isinstance(headlines, KeyError)
                  else str(headlines))
        sections["news"] = {"ok": False, "error": reason}
        lines.append(f"News: unavailable — {reason}.")
    else:
        sections["news"] = {"ok": True, "items": headlines}
        titles = [str(h.get("title") or "").strip()
                  for h in headlines if isinstance(h, dict)]
        titles = [t for t in titles if t]
        if titles:
            lines.append("News: " + "; ".join(titles) + ".")
        else:
            lines.append("News: no headlines right now.")

    return {"ok": True, "date": date_str, "sections": sections,
            "text": "\n".join(lines)}


def register_briefing_tool(registry: ToolRegistry,
                           sources: Dict[str, Callable]) -> None:
    async def daily_briefing(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                payload = {}
            city = payload.get("city")
            if isinstance(city, str) and city.strip() and callable(
                    sources.get("weather")):
                # The weather source decides whether it accepts a city.
                weather_fn = sources["weather"]

                async def weather_with_city():
                    return await weather_fn(city.strip())
                used = dict(sources)
                used["weather"] = weather_with_city
            else:
                used = sources
            return await build_briefing(datetime.now(), used)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    registry.register(ToolSpec("daily_briefing", daily_briefing, RiskLevel.LOW,
                               Capability.SENSITIVE_ACCESS))
