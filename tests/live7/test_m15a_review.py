"""Claude's visual review of M15a (2026-10-07, real screenshots): the new UI passes the exam but
  1. an EMPTY captions box (640x172, solid background) sits right over the orb's glowing core,
  2. before Start a large opaque card covers the orb and hides half of the "READY" label,
  3. the state label is drawn on top of the orb core,
  4. on a phone the dock wraps into three rows (166 px tall) and eats the screen.
These checks keep the orb visible: nothing opaque may sit on the stage centre (where the canvas draws
the core), captions live BELOW the core and are invisible while empty, the dock stays compact."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdp import CHROME, open_page  # noqa: E402
from test_m15a_ui import served  # noqa: E402

pytestmark = pytest.mark.skipif(CHROME is None, reason="Chrome not installed")

GEOM = r"""
(() => {
  const st = document.getElementById('stage').getBoundingClientRect();
  const cx = st.left + st.width / 2, cy = st.top + st.height / 2;
  const opaque = el => { const c = getComputedStyle(el).backgroundColor.match(/rgba?\(([^)]+)\)/);
    if (!c) return false; const p = c[1].split(',').map(Number); return (p.length < 4 ? 1 : p[3]) > 0.15; };
  const allowed = new Set(['HTML', 'BODY']);
  const blockers = [];
  for (const [dx, dy] of [[0,0],[-50,0],[50,0],[0,-50],[0,50]]) {
    for (const el of document.elementsFromPoint(cx + dx, cy + dy)) {
      if (allowed.has(el.tagName) || el.id === 'stage' || el.id === 'orb') continue;
      if (el.id === 'start' || el.closest('#start')) continue;           // the call to action may sit on the orb
      if (opaque(el) || getComputedStyle(el).borderTopWidth !== '0px' && getComputedStyle(el).borderTopStyle !== 'none')
        blockers.push((el.id || el.className || el.tagName).toString().slice(0, 40));
    }
  }
  const cap = document.getElementById('captions'), cr = cap.getBoundingClientRect();
  const lab = document.getElementById('state').getBoundingClientRect();
  const capEmpty = cap.textContent.trim() === '';
  const capVisibleBox = cr.height > 0 && (opaque(cap) || getComputedStyle(cap).borderTopStyle !== 'none' && getComputedStyle(cap).borderTopWidth !== '0px');
  const r = Math.min(st.width, st.height);
  return {blockers: [...new Set(blockers)], capEmpty, capVisibleBox,
          capTopBelowCore: cr.height === 0 || cr.top >= cy + 0.12 * r,
          labelOffCore: lab.bottom <= cy - 0.12 * r || lab.top >= cy + 0.12 * r,
          dockH: document.getElementById('dock').getBoundingClientRect().height};
})()
"""


def geom(width, height, start, captions_text=None):
    async def go():
        async with served() as url:
            async with await open_page(url, width=width, height=height) as p:
                if start:
                    await p.eval("document.getElementById('start').click()")
                    await asyncio.sleep(1.2)
                if captions_text:
                    await p.eval("(() => { const c = document.getElementById('captions'); const d = document.createElement('div');"
                                 f" d.textContent = {captions_text!r}; c.appendChild(d); }})()")
                    await asyncio.sleep(0.2)
                return await p.eval(GEOM)
    return asyncio.run(go())


@pytest.mark.parametrize("start", [False, True])
def test_nothing_opaque_covers_the_orb_core_desktop(start):
    g = geom(1280, 800, start)
    assert g["blockers"] == [], g
    assert g["labelOffCore"], g
    assert not (g["capEmpty"] and g["capVisibleBox"]), g     # an empty captions box must not show


def test_captions_with_text_sit_below_the_core():
    g = geom(1280, 800, True, "Hello, how are you doing today?")
    assert g["capTopBelowCore"] and g["blockers"] == [], g


def test_phone_orb_clear_and_dock_compact():
    g = geom(390, 844, True)
    assert g["blockers"] == [] and g["labelOffCore"], g
    assert g["dockH"] <= 120, g
