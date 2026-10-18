"""M9: the full Agent 7 tool set for 7 Live, built on seven_ui.SevenService.

build_live_registry(state_dir, hermes_bridge=None) -> (registry, service)

The registry is every tool of SevenService.registry EXCEPT "chat_reply" —
7 Live speaks for itself, it never needs a tool to form a reply — with the
SAME ToolSpec objects (same fn, risk, capabilities; the service keeps the
shared state: undo stack, episodic memory, pages, team, prolepsis, DNA),
plus the live-only tools:
    "ask_hermes" / "hermes_result" (live7.hermes_bridge.register_hermes_tools,
                                    only when a bridge is given)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from sentinel_orchestrator import ToolRegistry

from live7.briefing import fetch_news, register_briefing_tool
from live7.calendar7 import register_calendar_tools
from live7.hermes_bridge import HermesBridge, register_hermes_tools
from live7.mac import register_mac_tools
from live7.front import show_front
from live7.news import register_news_tools
from live7.office import register_office_tools


def build_live_registry(
    state_dir,
    hermes_bridge: Optional[HermesBridge] = None,
    on_pause=None,
    on_event=None,
) -> Tuple[ToolRegistry, object]:
    """Build the 7 Live registry on top of the SevenService state."""
    from seven_ui import SevenService  # deferred: seven_ui pulls in stt7 & friends

    service = SevenService(Path(state_dir))
    registry = ToolRegistry()
    for name, spec in service.registry._tools.items():
        if name == "chat_reply":  # 7 Live speaks for itself
            continue
        registry.register(spec)  # same ToolSpec objects, unchanged

    register_mac_tools(registry, undo=service.undo, on_pause=on_pause)

    # M19c: on macOS, opening apps really opens them.
    #
    # * desktop_open_app: the capabilities.py spec only knows Windows/Linux
    #   binaries and its "bad app" validation used to run AFTER the user had
    #   already approved. Swap in live7.mac.open_app (open -a <resolved name>,
    #   never a shell) + validate_open_app (resolves aliases and installed
    #   .app names, so a bad name fails as a precheck BEFORE the question).
    #   Same tool name, same HIGH risk; capabilities.py stays unchanged.
    # * app_open: the approval IS the user's consent, so a call without an
    #   explicit "dry_run" really opens the app here (apps7.app_open keeps
    #   dry_run=True as its own default for the old Agent 7 UI).
    import platform
    if platform.system() == "Darwin":
        from apps7 import app_open as _apps7_app_open

        from live7.mac import open_app as _mac_open_app, validate_open_app

        # Mutate the live spec in place: the M9 exam requires the 7 Live
        # registry to hold the SAME ToolSpec objects as SevenService.
        registry.get("desktop_open_app").fn = _mac_open_app
        registry.get("desktop_open_app").validate = validate_open_app

        app_spec = registry.get("app_open")

        async def _live_app_open(payload, _fn=_apps7_app_open):
            args = dict(payload or {})
            args.setdefault("dry_run", False)
            return await _fn(args)

        app_spec.fn = _live_app_open

    # M16: news on 7's own screen — the tools emit news/news_live/news_close
    # events which __main__ forwards to the active relay (the browser page).
    async def _no_event(event):
        return None
    register_news_tools(registry, on_event or _no_event)

    # M14: office documents + calendar + a real daily briefing, all on the
    # shared undo stack.
    register_office_tools(registry, undo=service.undo, show=show_front)
    register_calendar_tools(registry, undo=service.undo)

    async def _calendar_source(now):
        result = await registry.get("calendar_list").fn(
            {"start": now.strftime("%Y-%m-%d"), "days": 1})
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "calendar unavailable"))
        return result.get("events", [])

    async def _reminders_source():
        result = await registry.get("reminders_list").fn({})
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "reminders unavailable"))
        return result.get("reminders", [])

    async def _weather_source(city=None):
        import os
        city = city or os.environ.get("SEVEN_HOME_CITY", "").strip()
        if not city:
            raise RuntimeError("no city given and SEVEN_HOME_CITY is not set")
        result = await registry.get("weather_report").fn({"city": city})
        if not result.get("ok", True) or "temp_c" not in result:
            raise RuntimeError(result.get("error", "weather unavailable"))
        return {"summary": (
            f"{result.get('city')}: {result.get('temp_c')}°C, "
            f"feels like {result.get('feels_c')}°C, "
            f"{result.get('desc')}, humidity {result.get('humidity')}%, "
            f"wind {result.get('wind_kmph')} km/h")}

    async def _news_source():
        return await fetch_news()

    register_briefing_tool(registry, {
        "calendar": _calendar_source,
        "reminders": _reminders_source,
        "weather": _weather_source,
        "news": _news_source,
    })

    if hermes_bridge is not None:
        register_hermes_tools(registry, hermes_bridge)
    return registry, service
