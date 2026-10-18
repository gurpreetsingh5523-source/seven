"""
skills7.py — "7" ਦੇ ਵਾਧੂ skills (desktop-assistant ਕਾਬਲੀਅਤਾਂ,
ਸਾਡਾ ਆਪਣਾ ਕੋਡ, stdlib-first):

  * system_monitor  — CPU/RAM/disk (psutil ਹੋਵੇ ਤਾਂ, ਨਹੀਂ ਤਾਂ Linux /proc)
  * globe_distance  — ਦੋ ਸ਼ਹਿਰਾਂ ਵਿਚਕਾਰ ਹਵਾਈ ਦੂਰੀ (haversine) + ਉਡਾਣ ਸਮਾਂ
  * circuit_schematic — ਆਮ sensor↔board wiring ਡਾਇਆਗ੍ਰਾਮ (ASCII/text)
  * weather_report  — wttr.in ਤੋਂ ਮੌਸਮ (egress allowlist ਰਾਹੀਂ)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Tuple

from capabilities import EgressPolicy


# --------------------------------------------------------------------------- #
# System monitor (ਵੀਡੀਓ ਦਾ ਪਹਿਲਾ forged skill — ਸਾਡਾ built-in)                 #
# --------------------------------------------------------------------------- #

def _read_proc_stat() -> Tuple[int, int]:
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = [int(x) for x in parts]
    idle = vals[3] + vals[4]  # idle + iowait
    total = sum(vals)
    return idle, total


def _cpu_percent_linux(sample_s: float = 0.25) -> float:
    i1, t1 = _read_proc_stat()
    import time
    time.sleep(sample_s)
    i2, t2 = _read_proc_stat()
    di, dt = i2 - i1, t2 - t1
    return round(100.0 * (1 - di / dt), 1) if dt else 0.0


def _mem_linux() -> Dict[str, float]:
    info: Dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k.strip()] = int(v.strip().split()[0])  # kB
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", 0)
    used_pct = round(100.0 * (1 - avail / total), 1) if total else 0.0
    return {"ram_total_mb": total // 1024, "ram_used_pct": used_pct}


async def system_monitor(payload: Dict[str, Any]) -> Dict[str, Any]:
    """CPU/RAM/disk — psutil (ਸਾਰੇ OS) ਜਾਂ Linux /proc fallback।"""
    try:
        import psutil  # type: ignore
        cpu = psutil.cpu_percent(interval=0.25)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {"cpu_pct": cpu, "ram_used_pct": mem.percent,
                "ram_total_mb": mem.total // (1024 * 1024),
                "disk_used_pct": disk.percent, "source": "psutil"}
    except ImportError:
        if os.name != "posix" or not os.path.exists("/proc/stat"):
            return {"error": "psutil ਨਹੀਂ ਅਤੇ /proc ਵੀ ਨਹੀਂ — "
                             "pip install psutil ਕਰੋ"}
        cpu = await asyncio.to_thread(_cpu_percent_linux)
        mem = await asyncio.to_thread(_mem_linux)
        st = os.statvfs("/")
        disk_pct = round(100.0 * (1 - st.f_bavail / st.f_blocks), 1)
        return {"cpu_pct": cpu, "source": "/proc",
                "disk_used_pct": disk_pct, **mem}


# --------------------------------------------------------------------------- #
# Globe distance — haversine (globe ਦੀ core ਗਣਿਤ)                     #
# --------------------------------------------------------------------------- #

CITIES: Dict[str, Tuple[float, float]] = {
    # Punjab + India
    "amritsar": (31.6340, 74.8723), "ludhiana": (30.9010, 75.8573),
    "chandigarh": (30.7333, 76.7794), "jalandhar": (31.3260, 75.5762),
    "patiala": (30.3398, 76.3869), "bathinda": (30.2110, 74.9455),
    "delhi": (28.7041, 77.1025), "mumbai": (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946), "kolkata": (22.5726, 88.3639),
    "srinagar": (34.0837, 74.7973), "lahore": (31.5204, 74.3587),
    # World
    "london": (51.5074, -0.1278), "new york": (40.7128, -74.0060),
    "toronto": (43.6532, -79.3832), "vancouver": (49.2827, -123.1207),
    "dubai": (25.2048, 55.2708), "singapore": (1.3521, 103.8198),
    "sydney": (-33.8688, 151.2093), "tokyo": (35.6762, 139.6503),
    "paris": (48.8566, 2.3522), "berlin": (52.5200, 13.4050),
    "moscow": (55.7558, 37.6173), "beijing": (39.9042, 116.4074),
    "san francisco": (37.7749, -122.4194), "chicago": (41.8781, -87.6298),
    "mozambique": (-18.6657, 35.5296), "nairobi": (-1.2921, 36.8219),
    "auckland": (-36.8485, 174.7633), "los angeles": (34.0522, -118.2437),
}


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


async def globe_distance(payload: Dict[str, Any]) -> Dict[str, Any]:
    """ਦੋ ਸ਼ਹਿਰਾਂ ਵਿਚਕਾਰ ਦੂਰੀ + ਅੰਦਾਜ਼ਨ ਉਡਾਣ ਸਮਾਂ (cruise ~880 km/h + 40 ਮਿੰਟ overhead)।"""
    def _city(value: Any) -> str:
        # Voice/models often add ", country/state" — keep only the first part.
        return " ".join(str(value).split(",")[0].split()).lower()

    f = _city(payload.get("from_city", ""))
    t = _city(payload.get("to_city", ""))
    if f not in CITIES or t not in CITIES:
        missing = [c for c in (f, t) if c not in CITIES]
        return {"error": f"ਨਾਪਛਾਣਤਾ ਸ਼ਹਿਰ: {missing}",
                "hint": "ਜਾਣੇ ਸ਼ਹਿਰਾਂ ਦੀ ਸੂਚੀ 'known_cities' ਵਿੱਚ",
                "known_cities": sorted(CITIES)}
    km = round(haversine_km(CITIES[f], CITIES[t]), 1)
    flight_h = round(km / 880 + 0.66, 1) if km > 300 else None
    return {"from": f, "to": t, "distance_km": km,
            "flight_hours_est": flight_h,
            "note": "ਹਵਾਈ (great-circle) ਦੂਰੀ; ਅਸਲ ਰੂਟ ਵੱਖ ਹੋ ਸਕਦਾ"}


# --------------------------------------------------------------------------- #
# Circuit schematic — ਆਮ sensors ↔ boards (text ਡਾਇਆਗ੍ਰਾਮ)                     #
# --------------------------------------------------------------------------- #

SCHEMATICS: Dict[str, str] = {
    "dht11_esp32": """DHT11 → ESP32
  DHT11 VCC  ── 3V3
  DHT11 GND  ── GND
  DHT11 DATA ── GPIO4  (+10kΩ pull-up ਤੋਂ 3V3)
ਨੋਟ: Adafruit DHT library; pinout ਆਪਣੇ module ਨਾਲ verify ਕਰੋ।""",
    "dht11_arduino": """DHT11 → Arduino Uno
  DHT11 VCC  ── 5V
  DHT11 GND  ── GND
  DHT11 DATA ── D2  (+10kΩ pull-up ਤੋਂ 5V)""",
    "soil_esp32": """Soil Moisture (capacitive) → ESP32
  VCC ── 3V3   GND ── GND   AOUT ── GPIO34 (ADC1; ADC2 WiFi ਨਾਲ ਟਕਰਾਉਂਦਾ)""",
    "soil_arduino": """Soil Moisture → Arduino Uno
  VCC ── 5V   GND ── GND   AOUT ── A0""",
    "hcsr04_arduino": """HC-SR04 Ultrasonic → Arduino Uno
  VCC ── 5V  GND ── GND  TRIG ── D9  ECHO ── D10 (voltage divider ਨਾਲ)""",
    "led_arduino": """LED → Arduino Uno
  D13 ── 220Ω resistor ── LED+ (lambi taang)   LED− ── GND""",
}


async def circuit_schematic(payload: Dict[str, Any]) -> Dict[str, Any]:
    """sensor + board → wiring ਡਾਇਆਗ੍ਰਾਮ। ਜਾਣੇ combos: """
    sensor = str(payload.get("sensor", "")).strip().lower().replace("-", "").replace(" ", "")
    board = str(payload.get("board", "arduino")).strip().lower().replace(" ", "")
    key = f"{sensor}_{board}"
    if key in SCHEMATICS:
        return {"schematic": SCHEMATICS[key], "combo": key,
                "warning": "pinout ਆਪਣੇ module ਦੇ datasheet ਨਾਲ verify ਕਰੋ"}
    return {"error": f"'{key}' ਲਈ ਤਿਆਰ ਡਾਇਆਗ੍ਰਾਮ ਨਹੀਂ",
            "known": sorted(SCHEMATICS)}


# --------------------------------------------------------------------------- #
# Weather — wttr.in (egress allowlist ਨਾਲ)                                     #
# --------------------------------------------------------------------------- #

async def weather_report(payload: Dict[str, Any],
                         egress: EgressPolicy) -> Dict[str, Any]:
    """ਸ਼ਹਿਰ ਦਾ ਮੌਸਮ — wttr.in format j1 (JSON)।"""
    city = str(payload.get("city", "")).strip()
    if not city:
        raise ValueError("weather_report needs 'city'")
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
    egress.check(url)  # allowlist ਵਿੱਚ wttr.in ਹੋਣਾ ਲਾਜ਼ਮੀ

    def _get() -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "7-agent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read(200_000).decode("utf-8", errors="replace")

    raw = await asyncio.to_thread(_get)
    data = json.loads(raw)
    cur = data["current_condition"][0]
    return {"city": city,
            "temp_c": cur.get("temp_C"),
            "feels_c": cur.get("FeelsLikeC"),
            "desc": (cur.get("weatherDesc") or [{}])[0].get("value"),
            "humidity": cur.get("humidity"),
            "wind_kmph": cur.get("windspeedKmph")}
