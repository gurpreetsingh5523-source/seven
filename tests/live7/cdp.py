"""Tiny Chrome DevTools Protocol helper for UI exams: open a page in real headless Chrome at a given
viewport, run JavaScript, take screenshots. Test helper only (not product code)."""

import asyncio
import base64
import json
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path

CHROME = next((p for p in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                           shutil.which("google-chrome") or "", shutil.which("chromium") or "")
               if p and Path(p).exists()), None)


class Page:
    def __init__(self, ws):
        self.ws, self._id = ws, 0

    async def send(self, method, **params):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def eval(self, expr):
        r = await self.send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("text", "JS error") + ": " +
                               str(r["exceptionDetails"].get("exception", {}).get("description", "")))
        return r["result"].get("value")

    async def screenshot(self, path):
        r = await self.send("Page.captureScreenshot", format="png")
        Path(path).write_bytes(base64.b64decode(r["data"]))


async def open_page(url, width=1280, height=800, settle_s=1.5):
    """async context: yields a Page with the url loaded at width x height."""
    from websockets.asyncio.client import connect

    class _Ctx:
        async def __aenter__(self):
            self.prof = tempfile.TemporaryDirectory()
            self.proc = await asyncio.create_subprocess_exec(
                CHROME, "--headless=new", "--disable-gpu", "--remote-debugging-port=0",
                f"--user-data-dir={self.prof.name}", "--no-first-run", "--autoplay-policy=no-user-gesture-required",
                "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "about:blank",
                stderr=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL)
            port = None
            while port is None:
                line = (await asyncio.wait_for(self.proc.stderr.readline(), 20)).decode()
                m = re.search(r"ws://127\.0\.0\.1:(\d+)/", line)
                if m:
                    port = int(m.group(1))
            targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10).read())
            page_ws = next(t["webSocketDebuggerUrl"] for t in targets if t.get("type") == "page")
            self.conn = await connect(page_ws, max_size=None)
            page = Page(self.conn)
            await page.send("Page.enable")
            await page.send("Runtime.enable")
            await page.send("Emulation.setDeviceMetricsOverride", width=width, height=height,
                            deviceScaleFactor=1, mobile=width < 600)
            await page.send("Page.navigate", url=url)
            await asyncio.sleep(settle_s)
            return page

        async def __aexit__(self, *exc):
            await self.conn.close()
            self.proc.kill()
            await self.proc.wait()
            self.prof.cleanup()

    return _Ctx()


# JS: WCAG contrast of every visible text element against its effective background colour.
CONTRAST_JS = r"""
(() => {
  const parse = c => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null;
    const p = m[1].split(',').map(x => parseFloat(x)); return {r:p[0], g:p[1], b:p[2], a: p.length > 3 ? p[3] : 1}; };
  const lum = ({r,g,b}) => { const f = v => { v /= 255; return v <= 0.03928 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4); };
    return 0.2126*f(r) + 0.7152*f(g) + 0.0722*f(b); };
  const bgOf = el => { for (let e = el; e; e = e.parentElement) { const c = parse(getComputedStyle(e).backgroundColor);
      if (c && c.a > 0.5) return c; } return parse(getComputedStyle(document.body).backgroundColor) || {r:0,g:0,b:0,a:1}; };
  const out = [];
  for (const el of document.querySelectorAll('body *')) {
    const own = [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim().length > 1);
    if (!own) continue;
    const st = getComputedStyle(el), r = el.getBoundingClientRect();
    if (st.visibility === 'hidden' || st.display === 'none' || r.width === 0 || r.height === 0 || parseFloat(st.opacity) === 0) continue;
    if (el.closest('[hidden],[aria-hidden="true"]')) continue;
    const fg = parse(st.color); if (!fg) continue;
    let op = 1; for (let e = el; e; e = e.parentElement) op *= parseFloat(getComputedStyle(e).opacity);
    const bg = bgOf(el), a = fg.a * op;
    const mix = {r: fg.r*a + bg.r*(1-a), g: fg.g*a + bg.g*(1-a), b: fg.b*a + bg.b*(1-a)};
    const L1 = lum(mix), L2 = lum(bg), ratio = (Math.max(L1,L2)+0.05)/(Math.min(L1,L2)+0.05);
    const size = parseFloat(st.fontSize), bold = parseInt(st.fontWeight) >= 700;
    const need = (size >= 24 || (size >= 18.66 && bold)) ? 3 : 4.5;
    out.push({tag: el.tagName, id: el.id, cls: el.className && String(el.className).slice(0,40),
              text: el.textContent.trim().slice(0, 30), ratio: Math.round(ratio*100)/100, need});
  }
  return out;
})()
"""
