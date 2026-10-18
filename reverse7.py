"""reverse7 — ਸੁਰੱਖਿਤ ਰਿਵਰਸ-ਇੰਜੀਨੀਅਰਿੰਗ ਕਿੱਟ (read-only)।

ਉਲਟ-ਇੰਜੀਨੀਅਰ (ulat) ਦੇ ਟੂਲ। ਸਖ਼ਤ ਨਿਯਮ:
  - ਨਮੂਨਾ *ਕਦੇ ਨਹੀਂ* ਚਲਾਇਆ ਜਾਂਦਾ — ਸਿਰਫ਼ bytes ਪੜ੍ਹੇ ਜਾਂਦੇ ਹਨ
  - ਕੋਈ ਨੈੱਟਵਰਕ ਨਹੀਂ, ਕੋਈ ਲਿਖਤ ਨਹੀਂ — pure stdlib
  - ਹਰ ਦਾਅਵੇ ਨਾਲ ਸਬੂਤ (offset/section)

ਅਜੇ ਹਦਾਇਤਾਂ: capstone/objdump ਤੋਂ ਬਿਨਾਂ ਗੂੜ੍ਹਾ disassembly ਨਹੀਂ —
ਜੋ ਹੈ ਉਹ ਇਮਾਨਦਾਰੀ ਨਾਲ: format, headers, sections, entropy (packing),
strings, APK permissions। ਅੱਗੇ: capstone plug-in।
"""
from __future__ import annotations

import math
import re
import struct
import zipfile
from pathlib import Path
from typing import Optional

MAX_BYTES = 16 * 1024 * 1024  # 16MB ਤੋਂ ਵੱਡਾ ਨਮੂਨਾ ਨਹੀਂ ਖੋਲ੍ਹਾਂਗੇ

_MAGICS = [
    (b"\x7fELF", "ELF (Linux/Unix executable)"),
    (b"MZ", "PE/Windows executable (MZ)"),
    (b"PK\x03\x04", "ZIP/JAR/APK archive"),
    (b"dex\n", "Android DEX bytecode"),
    (b"\xca\xfe\xba\xbe", "Java class / Mach-O fat"),
    (b"\xfe\xed\xfa\xce", "Mach-O 32-bit"),
    (b"\xfe\xed\xfa\xcf", "Mach-O 64-bit"),
    (b"\xcf\xfa\xed\xfe", "Mach-O 64-bit (LE)"),
    (b"\x89PNG", "PNG image"),
    (b"%PDF", "PDF document"),
]

_PE_MACHINE = {0x014c: "x86", 0x8664: "x86-64", 0x01c0: "ARM",
               0xaa64: "ARM64"}
_ELF_MACHINE = {0x03: "x86", 0x3e: "x86-64", 0x28: "ARM", 0xb7: "ARM64",
                0x08: "MIPS"}


def classify(data: bytes) -> str:
    for magic, name in _MAGICS:
        if data.startswith(magic):
            return name
    return "unknown"


def entropy(data: bytes) -> float:
    """Shannon entropy (0-8) — 7+ ਤੇ ਪੈਕਿੰਗ/ਇੰਕ੍ਰਿਪਸ਼ਨ ਸ਼ੱਕ।"""
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in freq if c)


_ASCII_RE = re.compile(rb"[\x20-\x7e]{4,}")
_UTF16_RE = re.compile(rb"(?:[\x20-\x7e]\x00){4,}")


def extract_strings(data: bytes, limit: int = 200) -> list:
    """ASCII + UTF-16LE strings — ਮੈਲਵੇਅਰ-ਟ੍ਰਾਇਏਜ਼ ਦੀ ਪਹਿਲੀ ਝਾਤ।"""
    out = [m.group().decode("ascii", "replace")
           for m in _ASCII_RE.finditer(data)]
    out += [m.group().decode("utf-16-le", "replace")
            for m in _UTF16_RE.finditer(data)]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:limit]


def elf_info(data: bytes) -> dict:
    if len(data) < 64 or not data.startswith(b"\x7fELF"):
        return {}
    is64 = data[4] == 2
    little = data[5] == 1
    en = "<" if little else ">"
    e_type = struct.unpack_from(en + "H", data, 16)[0]
    machine = struct.unpack_from(en + "H", data, 18)[0]
    entry = struct.unpack_from(en + ("Q" if is64 else "I"), data, 24)[0]
    return {"bits": 64 if is64 else 32,
            "endian": "little" if little else "big",
            "type": {1: "relocatable", 2: "executable", 3: "shared"}.get(
                e_type, str(e_type)),
            "machine": _ELF_MACHINE.get(machine, hex(machine)),
            "entry": hex(entry)}


def pe_info(data: bytes) -> dict:
    if len(data) < 0x40 or not data.startswith(b"MZ"):
        return {}
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_off + 24 > len(data) or data[pe_off:pe_off + 4] != b"PE\0\0":
        return {"note": "MZ ਹੈ ਪਰ PE header ਨਹੀਂ (DOS-only?)"}
    machine, nsec = struct.unpack_from("<HH", data, pe_off + 4)
    ts = struct.unpack_from("<I", data, pe_off + 8)[0]
    opt_size = struct.unpack_from("<H", data, pe_off + 20)[0]
    secs = []
    soff = pe_off + 24 + opt_size
    for i in range(min(nsec, 16)):
        base = soff + i * 40
        if base + 40 > len(data):
            break
        name = data[base:base + 8].rstrip(b"\0").decode("ascii", "replace")
        vsize = struct.unpack_from("<I", data, base + 8)[0]
        raw_off = struct.unpack_from("<I", data, base + 20)[0]
        raw_size = struct.unpack_from("<I", data, base + 16)[0]
        blob = data[raw_off:raw_off + min(raw_size, 1 << 20)]
        secs.append({"name": name, "vsize": vsize,
                     "entropy": round(entropy(blob), 2)})
    return {"machine": _PE_MACHINE.get(machine, hex(machine)),
            "sections": nsec, "timestamp": ts,
            "section_list": secs,
            "packed_suspect": any(s["entropy"] > 7.0 for s in secs)}


def apk_info(path: Path) -> dict:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    except zipfile.BadZipFile:
        return {}
    dex = [n for n in names if n.endswith(".dex")]
    perms: list = []
    if "AndroidManifest.xml" in names:
        blob = Path(path).read_bytes()[:MAX_BYTES]
        perms = sorted({m.group().decode() for m in re.finditer(
            rb"android\.permission\.[A-Z_]+", blob)})
    dangerous = [p for p in perms if p.rsplit(".", 1)[-1] in (
        "READ_SMS", "SEND_SMS", "RECORD_AUDIO", "CAMERA",
        "ACCESS_FINE_LOCATION", "READ_CONTACTS", "SYSTEM_ALERT_WINDOW",
        "REQUEST_INSTALL_PACKAGES", "BIND_ACCESSIBILITY_SERVICE")]
    return {"files": len(names), "dex_files": dex,
            "native_libs": [n for n in names if n.endswith(".so")][:10],
            "permissions": perms, "dangerous_permissions": dangerous,
            "signed": any(n.startswith("META-INF/") for n in names)}


def inspect_file(path: str, want_strings: int = 60) -> dict:
    """ਨਮੂਨੇ ਦੀ ਪੂਰੀ ਸਥਿਰ ਜਾਂਚ — read-only, ਬਿਨਾਂ ਚਲਾਏ।"""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"ਨਹੀਂ ਮਿਲੀ: {p}")
    if not p.is_file():
        raise ValueError("ਇਹ ਫਾਈਲ ਨਹੀਂ (ਫੋਲਡਰ?)")
    size = p.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"{size / 1e6:.0f}MB — ਹੱਦ {MAX_BYTES // 1e6}MB")
    data = p.read_bytes()

    fmt = classify(data)
    report = {"file": str(p), "size": size, "format": fmt,
              "entropy": round(entropy(data), 2),
              "packed_suspect": entropy(data) > 7.2}
    if fmt.startswith("ELF"):
        report["elf"] = elf_info(data)
    elif fmt.startswith("PE"):
        report["pe"] = pe_info(data)
    if fmt.startswith("ZIP") and p.suffix.lower() == ".apk":
        report["apk"] = apk_info(p)
    strings = extract_strings(data, want_strings)
    report["strings_sample"] = strings[:want_strings]
    report["strings_count_sampled"] = len(strings)
    iocs = [s for s in strings if re.search(
        r"https?://|\b\d{1,3}(?:\.\d{1,3}){3}\b|\.onion|powershell|cmd\.exe",
        s)]
    report["iocs"] = iocs[:20]
    report["note"] = ("ਸਥਿਰ ਜਾਂਚ ਸਿਰਫ਼ — ਨਮੂਨਾ ਨਹੀਂ ਚਲਾਇਆ ਗਿਆ। "
                      "ਗੂੜ੍ਹੇ disassembly ਲਈ capstone ਲਗਾਓ।")
    return report
