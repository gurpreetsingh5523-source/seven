#!/usr/bin/env python3
"""
seven.py — "7" ਲਾਂਚਰ: ਇੱਕੋ ਕਮਾਂਡ ਨਾਲ ਸਭ ਚਾਲੂ।

  python seven.py         → ਮੁੱਖ ਫੇਸ (orb UI) http://127.0.0.1:7878
  python seven.py --demo  → ਬਿਨਾਂ UI, ਕੰਸੋਲ ਡੈਮੋ (ਸਾਰੀਆਂ capabilities)

ਅਸਲੀ ਦਿਮਾਗ਼ ਲਈ (ਵਿਕਲਪਿਕ, ਮੁਫ਼ਤ):
  ollama serve &          → ਲੋਕਲ ਮਾਡਲ ਆਪਣੇ-ਆਪ ਵਰਤਿਆ ਜਾਵੇਗਾ
  OPENAI_API_KEY=...      → paid escalation
  pip install playwright && playwright install chromium
                          → ਅਸਲੀ browser automation (ਫਾਰਮ ਭਰਨਾ ਆਦਿ)
  pip install psutil      → ਸਾਰੇ OS 'ਤੇ system monitor
"""

import sys


def demo() -> None:
    """UI ਬਿਨਾਂ — capabilities ਦਾ ਜਲਦੀ ਪ੍ਰਦਰਸ਼ਨ (ਸਭ dry-run / ਸੁਰੱਖਿਅਤ)।"""
    import asyncio
    from apps7 import app_open, site_open, web_search_open, known_apps
    from skills7 import system_monitor, globe_distance, circuit_schematic
    from undo7 import UndoStack

    async def go():
        print("═" * 52)
        print("  7 — capability demo (ਸਭ ਸੁਰੱਖਿਅਤ/dry-run)")
        print("═" * 52)

        print("\n[1] ਐਪ ਲਾਂਚਰ (dry-run):")
        print("   ", await app_open({"name": "chrome", "dry_run": True}))

        print("\n[2] ਵੈਬਸਾਈਟ (dry-run):")
        print("   ", await site_open({"url": "youtube.com", "dry_run": True}))

        print("\n[3] ਸਰਚ (dry-run):")
        print("   ", await web_search_open({"query": "ਪੰਜਾਬ ਖ਼ਬਰਾਂ", "dry_run": True}))

        print("\n[4] ਸਿਸਟਮ ਮਾਨੀਟਰ (ਅਸਲੀ, ਸਿਰਫ਼-ਪੜ੍ਹਨ):")
        print("   ", await system_monitor({}))

        print("\n[5] ਗਲੋਬ ਦੂਰੀ:")
        print("   ", await globe_distance({"from_city": "amritsar",
                                            "to_city": "london"}))

        print("\n[6] ਸਰਕਟ ਸਕੀਮੈਟਿਕ:")
        print((await circuit_schematic({"sensor": "dht11",
                                        "board": "esp32"}))["schematic"])

        print("\n[7] Undo stack:")
        u = UndoStack()
        print("    ਖਾਲੀ stack undo →", await u.undo_last())

        print("\n[8] ਜਾਣੀਆਂ ਐਪਾਂ:", ", ".join(known_apps()[:10]), "…")
        print("\nਮੁੱਖ ਫੇਸ ਲਈ:  python seven.py   (ਬਿਨਾਂ arguments)")

    asyncio.run(go())


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        import seven_ui
        seven_ui.main()
