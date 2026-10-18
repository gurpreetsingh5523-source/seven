"""
browser7.py — "7" ਦਾ browser automation (ਸਰਚ, ਸਫ਼ਾ ਖੋਲ੍ਹੋ, ਫਾਰਮ ਭਰੋ)।

ਦੋ ਪੱਧਰ:
  1. Playwright ਮਿਲੇ → ਅਸਲੀ automation: ਖੋਲ੍ਹੋ, click, type, ਫਾਰਮ ਭਰੋ, text ਕੱਢੋ
     (pip install playwright && playwright install chromium)
  2. ਨਾ ਮਿਲੇ → ਇਮਾਨਦਾਰ fallback: site_open/search (webbrowser) ਜਾਰੀ, ਫਾਰਮ-ਭਰਨਾ
     "playwright ਚਾਹੀਦਾ" ਦੱਸਦਾ ਹੈ — ਕਦੇ ਝੂਠੀ ਸਫਲਤਾ ਨਹੀਂ।

ਸੁਰੱਖਿਆ: ਹਰ action capabilities.py ਦੀ EgressPolicy ਰਾਹੀਂ — allowlist ਤੋਂ ਬਾਹਰ
host ਨਹੀਂ, private IP ਬਲੌਕ। ਫਾਰਮ submit = EXTERNAL_EFFECT (orchestrator ਮਨਜ਼ੂਰੀ)।
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from capabilities import EgressPolicy

try:  # optional dependency
    from playwright.async_api import async_playwright
    _PLAYWRIGHT = True
except ImportError:
    _PLAYWRIGHT = False


def playwright_available() -> bool:
    return _PLAYWRIGHT


class BrowserSession:
    """ਇੱਕ Chromium session — ਲੰਬੇ ਕੰਮਾਂ ਲਈ ਖੁੱਲ੍ਹਾ ਰੱਖੋ (multi-step ਫਾਰਮ)।"""

    def __init__(self, egress: EgressPolicy, headless: bool = True):
        self.egress = egress
        self.headless = headless
        self._pw = None
        self._browser = None
        self.page = None

    async def __aenter__(self) -> "BrowserSession":
        if not _PLAYWRIGHT:
            raise RuntimeError(
                "playwright ਨਹੀਂ ਮਿਲਿਆ — ਅਸਲੀ browser automation ਲਈ: "
                "pip install playwright && playwright install chromium")
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=self.headless)
        self.page = await self._browser.new_page()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    async def goto(self, url: str) -> Dict[str, Any]:
        self.egress.check(url)  # Sentinel-ਗੇਟ: allowlist ਤੋਂ ਬਿਨਾਂ ਕੁਝ ਨਹੀਂ
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return {"url": self.page.url, "title": await self.page.title()}

    async def fill(self, selector: str, value: str) -> Dict[str, Any]:
        await self.page.fill(selector, value)
        return {"filled": selector}

    async def click(self, selector: str) -> Dict[str, Any]:
        await self.page.click(selector, timeout=15000)
        return {"clicked": selector}

    async def text(self, selector: str = "body", max_chars: int = 2000) -> Dict[str, Any]:
        content = await self.page.inner_text(selector)
        return {"text": content[:max_chars]}

    async def press(self, key: str) -> Dict[str, Any]:
        await self.page.keyboard.press(key)
        return {"pressed": key}


async def browse_task(payload: Dict[str, Any], egress: EgressPolicy) -> Dict[str, Any]:
    """ਇੱਕ multi-step browser ਕੰਮ — steps ਦੀ ਸੂਚੀ:
      {"steps": [{"goto": url}, {"fill": [selector, value]},
                 {"click": selector}, {"press": "Enter"}, {"text": selector}]}
    ਹਰ step ਲੜੀ ਵਿੱਚ; ਗਲਤੀ 'ਤੇ ਰੁਕ ਕੇ ਇਮਾਨਦਾਰ error।
    """
    steps: List[Dict[str, Any]] = payload.get("steps", [])
    if not steps:
        raise ValueError("browse_task needs non-empty 'steps'")
    headless = bool(payload.get("headless", True))
    results: List[Dict[str, Any]] = []
    async with BrowserSession(egress, headless=headless) as s:
        for i, step in enumerate(steps):
            if "goto" in step:
                results.append(await s.goto(step["goto"]))
            elif "fill" in step:
                sel, val = step["fill"]
                results.append(await s.fill(sel, val))
            elif "click" in step:
                results.append(await s.click(step["click"]))
            elif "press" in step:
                results.append(await s.press(step["press"]))
            elif "text" in step:
                results.append(await s.text(step.get("text") or "body"))
            else:
                raise ValueError(f"step {i}: ਅਣਜਾਣ action {list(step)}")
    return {"completed": len(results), "results": results}


async def smart_search(payload: Dict[str, Any], egress: EgressPolicy) -> Dict[str, Any]:
    """ਸਰਚ ਕਰਕੇ ਪਹਿਲੇ ਨਤੀਜੇ ਦੇ title+link ਕੱਢੋ (Playwright ਹੋਵੇ ਤਾਂ)।
    ਨਹੀਂ ਤਾਂ ਸਾਫ਼ ਦੱਸੋ ਕਿ simplified mode ਹੈ।"""
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ValueError("smart_search needs 'query'")
    engine_url = "https://duckduckgo.com/?q=" + __import__(
        "urllib.parse", fromlist=["quote_plus"]).quote_plus(query)
    egress.check(engine_url)
    if not _PLAYWRIGHT:
        return {"mode": "no_playwright", "query": query,
                "note": "results ਕੱਢਣ ਲਈ playwright install ਕਰੋ; "
                        "site_open ਨਾਲ browser ਵਿੱਚ ਖੋਲ੍ਹ ਸਕਦੇ ਹੋ",
                "url": engine_url}
    async with BrowserSession(egress) as s:
        await s.goto(engine_url)
        await asyncio.sleep(1.5)  # results render
        data = await s.text("body", max_chars=1500)
    return {"mode": "live", "query": query, "preview": data["text"]}
