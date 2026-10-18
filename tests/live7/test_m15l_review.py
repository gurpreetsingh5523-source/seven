"""Claude's visual review of the M15L boot animation (2026-10-07, frame captures): it works but is far
too small and weak next to the reference launch — wordmark letters ~30 px tall, the ੭ mark small, the
light sweep drawn as a grey bar BESIDE the letters, small ring. The reference fills the screen centre
with big, bold, glowing letters. These checks hold the boot at a real "launch moment" scale."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdp import CHROME, open_page  # noqa: E402
from test_m15a_ui import served  # noqa: E402

pytestmark = pytest.mark.skipif(CHROME is None, reason="Chrome not installed")

MEASURE = r"""(() => {
  const R = id => document.getElementById(id).getBoundingClientRect();
  const w = R('word'), m = R('mark'), t = R('tag');
  const st = getComputedStyle(document.querySelector('#word span'));
  return {vw: innerWidth, vh: innerHeight, word: [w.left, w.top, w.width, w.height], mark: [m.left, m.top, m.width, m.height],
          tag: [t.left, t.top, t.width, t.height], weight: parseInt(st.fontWeight), size: parseFloat(st.fontSize),
          sweepOutside: [...document.querySelectorAll('body *')].some(e => {
             const s = getComputedStyle(e), r = e.getBoundingClientRect();
             return /sweep|shine|shimmer/i.test(e.className + ' ' + e.id) && e.closest('#word') === null && r.width > 0 && s.opacity !== '0'; })};
})()"""


@pytest.mark.parametrize("size", [(1440, 900), (1280, 720)])
def test_boot_is_big_bold_and_centred(size):
    w, h = size

    async def go():
        async with served() as url:
            async with await open_page(url + "boot.html?silent=1", width=w, height=h, settle_s=2.4) as p:
                return await p.eval(MEASURE)
    g = asyncio.run(go())
    vw, vh = g["vw"], g["vh"]
    wl, wt, ww, wh = g["word"]
    ml, mt, mw, mh = g["mark"]
    assert g["size"] >= 0.08 * vh and g["weight"] >= 600, g          # big, bold letters
    assert ww >= 0.40 * vw, g                                          # wordmark spans the centre
    assert mh >= 0.16 * vh, g                                          # the ੭ core is a real hero
    assert abs((wl + ww / 2) - vw / 2) <= 0.03 * vw, g                 # centred horizontally
    group_mid = (mt + (wt + wh)) / 2
    assert abs(group_mid - vh / 2) <= 0.12 * vh, g                     # mark+word centred vertically
    assert not g["sweepOutside"], g                                    # sweep passes THROUGH the letters


def test_launcher_opens_window_via_on_done_and_without_overlay(monkeypatch):
    """Claude review: PyObjC stopEventLoop() after runEventLoop() TERMINATES the process, so code
    after show_overlay() never ran and the 7 window never opened. The window must open from on_done."""
    from live7 import launcher
    opened = []
    monkeypatch.setattr(launcher, "ensure_server", lambda **kw: True)
    monkeypatch.setattr(launcher, "run_app", lambda base, **kw: False)   # no WebKit: Chrome fallback path
    monkeypatch.setattr(launcher, "boot_stamp", lambda: 0)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda cmd, **kw: opened.append(cmd))

    def overlay_that_exits(url, timeout_s=6, on_done=None):
        on_done()
        raise SystemExit(0)              # what PyObjC does after stopEventLoop
    monkeypatch.setattr(launcher, "show_overlay", overlay_that_exits)
    with pytest.raises(SystemExit):
        launcher.main([])
    assert len(opened) == 1 and any("--app=" in a for a in opened[0])
    opened.clear()
    monkeypatch.setattr(launcher, "show_overlay", lambda url, timeout_s=6, on_done=None: False)
    assert launcher.main([]) == 0 and len(opened) == 1
