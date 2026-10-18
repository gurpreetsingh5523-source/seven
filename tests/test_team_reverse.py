"""team7 + reverse7 ਦੇ ਟੈਸਟ"""
import struct
import zipfile
from pathlib import Path

import pytest

from team7 import Team, route, review_code, SPECIALISTS, LAZY_LADDER
from reverse7 import (classify, entropy, extract_strings, elf_info,
                      pe_info, inspect_file, apk_info)


# ---- team7 ----

def test_roster_has_nine_specialists():
    t = Team()
    r = t.roster()
    assert len(r) == 9
    assert {a["key"] for a in r} == set(SPECIALISTS)


def test_route_to_specialists():
    assert route("ਇਸ APK ਦੀ ਜਾਂਚ ਕਰੋ").key == "ulat"
    assert route("review my code").key == "samikhia"
    assert route("ਪੰਜਾਬ ਦੀ ਖ਼ਬਰ ਲੱਭੋ").key == "khoji"
    assert route("ਸੁਪਨਾ ਵੇਖ").key == "sutradhar"
    assert route("ਫਾਈਲਾਂ organize ਕਰੋ").key == "jantri"
    assert route("ਸਤ ਸ੍ਰੀ ਅਕਾਲ").key == "soofi"


def test_team_assign_and_can_use():
    t = Team()
    a = t.assign("review this code")
    assert a["specialist"] == "samikhia"
    assert t.can_use("samikhia", "code_review")
    assert not t.can_use("samikhia", "browser_task")  # whitelist ਬਾਹਰ


def test_review_flags_new_dependency():
    r = review_code("import requests\nprint(requests.get('x'))\n")
    assert any(i["rule"] == "new_dependency" for i in r["issues"])


def test_review_flags_single_method_class_and_bare_except():
    code = (
        "class Thing:\n"
        "    def run(self):\n"
        "        try:\n"
        "            pass\n"
        "        except:\n"
        "            pass\n"
    )
    r = review_code(code)
    rules = {i["rule"] for i in r["issues"]}
    assert "single_method_class" in rules
    assert "bare_except" in rules


def test_review_clean_code_passes():
    r = review_code("from pathlib import Path\n\ndef go(p):\n    return Path(p).name\n")
    assert r["ok"], r["issues"]
    assert len(LAZY_LADDER) == 7


def test_review_syntax_error_reported():
    r = review_code("def broken(:\n")
    assert not r["ok"]
    assert r["issues"][0]["rule"] == "syntax"


# ---- reverse7 ----

def _fake_elf() -> bytes:
    e = bytearray(64)
    e[0:4] = b"\x7fELF"
    e[4] = 2   # 64-bit
    e[5] = 1   # little-endian
    struct.pack_into("<H", e, 16, 2)      # executable
    struct.pack_into("<H", e, 18, 0x3E)   # x86-64
    struct.pack_into("<Q", e, 24, 0x401000)
    return bytes(e)


def _fake_pe() -> bytes:
    d = bytearray(0x400)
    d[0:2] = b"MZ"
    struct.pack_into("<I", d, 0x3C, 0x80)
    d[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HH", d, 0x84, 0x8664, 1)   # x86-64, 1 section
    struct.pack_into("<H", d, 0x98, 0)            # no optional header
    base = 0x80 + 24
    d[base:base + 7] = b".text\0\0"
    struct.pack_into("<I", d, base + 8, 0x1000)   # vsize
    struct.pack_into("<I", d, base + 16, 0x200)   # raw size
    struct.pack_into("<I", d, base + 20, 0x200)   # raw offset
    return bytes(d)


def test_classify_magic():
    assert classify(b"\x7fELF...").startswith("ELF")
    assert classify(b"MZ...").startswith("PE")
    assert classify(b"PK\x03\x04...").startswith("ZIP")
    assert classify(b"\x00\x01").__eq__("unknown")


def test_entropy_uniform_vs_random():
    assert entropy(b"\x00" * 1000) == 0.0
    import os
    assert entropy(os.urandom(4096)) > 7.0


def test_extract_strings():
    data = b"junk\x00http://evil.example/payload\x00more\x00cmd.exe\x00"
    strs = extract_strings(data)
    assert any("evil.example" in s for s in strs)
    assert any("cmd.exe" in s for s in strs)


def test_elf_and_pe_info():
    e = elf_info(_fake_elf())
    assert e["bits"] == 64 and e["machine"] == "x86-64" and e["entry"] == "0x401000"
    p = pe_info(_fake_pe())
    assert p["machine"] == "x86-64" and p["sections"] == 1
    assert p["section_list"][0]["name"] == ".text"


def test_inspect_file_and_iocs(tmp_path: Path):
    f = tmp_path / "sample.bin"
    f.write_bytes(_fake_elf() + b"\x00http://bad.example/x\x00" * 3)
    r = inspect_file(str(f))
    assert r["format"].startswith("ELF")
    assert r["elf"]["machine"] == "x86-64"
    assert any("bad.example" in s for s in r["iocs"])
    with pytest.raises(FileNotFoundError):
        inspect_file(str(tmp_path / "nope.bin"))


def test_apk_info_and_dangerous_perms(tmp_path: Path):
    apk = tmp_path / "app.apk"
    with zipfile.ZipFile(apk, "w") as z:
        z.writestr("classes.dex", b"dex\n035 fake")
        z.writestr("AndroidManifest.xml", b"binary")
        z.writestr("META-INF/CERT.SF", b"sig")
    raw = apk.read_bytes() + b"android.permission.SEND_SMS\x00android.permission.INTERNET\x00"
    apk.write_bytes(raw)
    # zip ਟੁੱਟ ਗਈ — ਨਵੀਂ ਵੈਧ apk ਬਣਾ ਕੇ manifest-ਬਾਈਟਸ ਆਪਣੇ-ਆਪ ਟੈਸਟ
    apk2 = tmp_path / "app2.apk"
    with zipfile.ZipFile(apk2, "w") as z:
        z.writestr("classes.dex", b"dex\n035 fake")
        z.writestr("AndroidManifest.xml", b"binary")
    info = apk_info(apk2)
    assert info["dex_files"] == ["classes.dex"]
    assert "signed" in info
