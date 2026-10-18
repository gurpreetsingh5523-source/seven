"""Drag-and-drop file intake for 7 Live (M12).

The page sends {"type":"file","name","mime","data_b64"}; LiveRelay.handle_file
decodes the bytes and calls ingest_file(), which:

- rejects anything over 10 MB,
- sanitises the name (no path separators, no leading dots),
- saves it under the uploads dir (~/.seven/uploads, or SEVEN_UPLOADS),
- classifies it: "image" (image/* mime), "text" (.txt .md .csv .json .py as
  UTF-8, plus .pdf/.docx when pypdf / python-docx are importable), or
  "unsupported".

Everything stays on this machine; the uploads dir is the only place written.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

MAX_BYTES = 10 * 1024 * 1024  # 10 MB, enforced both in the page and here

TEXT_EXTS = frozenset({".txt", ".md", ".csv", ".json", ".py"})
TEXT_MIMES = frozenset(
    {"text/plain", "text/markdown", "text/csv", "application/json"}
)


def default_uploads_dir() -> Path:
    """~/.seven/uploads, overridable with SEVEN_UPLOADS (tests use tmp_path)."""
    env = os.environ.get("SEVEN_UPLOADS")
    return Path(env) if env else Path.home() / ".seven" / "uploads"


def _sanitise_name(name: str) -> str:
    """Just a bare filename: no directories, no leading dots, never empty."""
    base = str(name or "").replace("\\", "/").split("/")[-1]
    base = base.strip().lstrip(".").strip()
    return base or "upload"


def _unique_path(directory: Path, name: str) -> Path:
    """directory/name, or name_2/name_3... when that file already exists."""
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    for n in range(2, 1000):
        candidate = directory / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
    return directory / f"{stem}_{os.getpid()}{suffix}"


def _read_pdf(data: bytes) -> Optional[str]:
    """Text of a PDF via pypdf, or None when pypdf is not installed/fails."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception:
        return None


def _read_docx(data: bytes) -> Optional[str]:
    """Text of a .docx via python-docx, or None when unavailable/fails."""
    try:
        import docx
    except ImportError:
        return None
    try:
        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs).strip()
    except Exception:
        return None


def ingest_file(
    name: str,
    mime: str,
    data: bytes,
    uploads_dir: Any = None,
) -> Dict[str, Any]:
    """Save one uploaded file under uploads_dir and classify it.

    Returns {"ok", "path", "kind": "image"|"text"|"unsupported", "text"?};
    over-size files return {"ok": False, "error": ...} and nothing is written.
    """
    data = bytes(data or b"")
    if len(data) > MAX_BYTES:
        return {"ok": False, "error": f"file too large (max {MAX_BYTES} bytes)"}
    uploads = Path(uploads_dir) if uploads_dir is not None else default_uploads_dir()
    uploads.mkdir(parents=True, exist_ok=True)
    path = _unique_path(uploads, _sanitise_name(name))
    path.write_bytes(data)

    mime = (mime or "").strip().lower()
    ext = path.suffix.lower()
    kind = "unsupported"
    text: Optional[str] = None
    if ext in TEXT_EXTS or mime in TEXT_MIMES:
        kind = "text"
        text = data.decode("utf-8", "replace")
    elif ext == ".pdf":
        text = _read_pdf(data)
        kind = "text" if text is not None else "unsupported"
    elif ext == ".docx":
        text = _read_docx(data)
        kind = "text" if text is not None else "unsupported"
    elif mime.startswith("image/"):
        kind = "image"

    if kind == "unsupported":
        path.unlink(missing_ok=True)   # never keep files we will not use (e.g. dropped .exe)
    result: Dict[str, Any] = {"ok": True, "path": str(path), "kind": kind}
    if text is not None:
        result["text"] = text
    return result
