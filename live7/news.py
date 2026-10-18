"""M16: news on 7's own screen — headlines and live TV, never a browser.

Why: asking 7 for the news used to open Safari via web_search. Now the model
has three fast LOW-risk tools:

    news_show  {"feed": "world", "limit": 8}   headlines rendered in the Work panel
    news_live  {"channel": "DW News"}          live TV embedded in the Work panel
    news_close {}                              hide both

Safe by design:
  * headlines come ONLY from live7.briefing.fetch_news / NEWS_FEEDS (fixed BBC
    feeds — no arbitrary URLs), the same source the daily briefing uses;
  * live TV comes ONLY from the LIVE_CHANNELS allow-list (verified YouTube
    live-stream embeds). A "url" argument from the model is IGNORED — the page
    also refuses anything that is not a youtube.com/embed/ URL;
  * no tool ever raises: bad input or a fetch failure returns
    {"ok": False, "error": ...} and emits nothing.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sentinel_orchestrator import Capability, RiskLevel, ToolRegistry, ToolSpec

from live7.briefing import NEWS_FEEDS, fetch_news

_EMBED = "https://www.youtube.com/embed/live_stream?channel={}&autoplay=1"

LIVE_CHANNELS: Dict[str, str] = {
    "Al Jazeera English": _EMBED.format("UCNye-wNBqNL5ZzHSJj3l8Bg"),
    "DW News": _EMBED.format("UCknLrEdhRCp1aegoMqRaCZg"),
    "Sky News": _EMBED.format("UCoMdktPbSTixAyNGwb-UYkQ"),
    "France 24 English": _EMBED.format("UCQfwfsi5VrQ8yKZ-UWmAEFg"),
}

Emitter = Callable[[Dict[str, Any]], Awaitable[None]]
FetchNews = Callable[..., Awaitable[List[Dict[str, str]]]]


def _err(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg}


def live_url(name: Any) -> Optional[Tuple[str, str]]:
    """(display_name, embed_url) for a channel name; None when unknown.

    Case-insensitive; also matches a UNIQUE case-insensitive substring of one
    display name ("dw", "al jazeera", "sky" work; "news" matches several -> None).
    """
    if not isinstance(name, str):
        return None
    text = name.strip().lower()
    if not text:
        return None
    for display, url in LIVE_CHANNELS.items():
        if display.lower() == text:
            return display, url
    matches = [(display, url) for display, url in LIVE_CHANNELS.items()
               if text in display.lower()]
    if len(matches) == 1:
        return matches[0]
    return None


def _clamp_limit(raw: Any) -> int:
    try:
        value = int(float(raw))
    except (TypeError, ValueError, OverflowError):
        value = 8
    return max(1, min(15, value))


def register_news_tools(registry: ToolRegistry, emit: Emitter,
                        fetch_news: FetchNews = fetch_news) -> None:
    """Three LOW-risk, no-capability tools; every event goes through `emit`."""

    async def _emit(event: Dict[str, Any]) -> None:
        if emit is not None:
            await emit(event)

    async def news_show(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                payload = {}
            feed = str(payload.get("feed") or "world").strip().lower()
            if feed not in NEWS_FEEDS:
                return _err(f"unknown news feed {feed!r}; known: {sorted(NEWS_FEEDS)}")
            limit = _clamp_limit(payload.get("limit"))
            items = await fetch_news(limit=limit, feed=feed)
            if not isinstance(items, list):
                return _err("news fetch did not return a list")
            await _emit({"type": "news", "feed": feed, "items": items})
            return {"ok": True, "feed": feed, "items": items}
        except Exception as exc:
            return _err(str(exc))

    async def news_live(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                payload = {}
            # Any "url" from the caller is IGNORED: only allow-listed channels.
            found = live_url(payload.get("channel"))
            if found is None:
                return {"ok": False,
                        "error": f"unknown channel {payload.get('channel')!r}",
                        "channels": list(LIVE_CHANNELS)}
            channel, url = found
            await _emit({"type": "news_live", "channel": channel, "url": url})
            return {"ok": True, "channel": channel}
        except Exception as exc:
            return _err(str(exc))

    async def news_close(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            await _emit({"type": "news_close"})
            return {"ok": True}
        except Exception as exc:
            return _err(str(exc))

    for spec in (
        ToolSpec("news_show", news_show, RiskLevel.LOW, Capability.NONE),
        ToolSpec("news_live", news_live, RiskLevel.LOW, Capability.NONE),
        ToolSpec("news_close", news_close, RiskLevel.LOW, Capability.NONE),
    ):
        registry.register(spec)
