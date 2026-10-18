"""pages7 — 7 ਦੇ ਪੇਜ: ਲਿਖਤ ਤੋਂ ਪਹਿਲਾਂ ਡਰਾਫ਼ਟ, ਹਰ ਸੇਵ ਤੋਂ ਪਹਿਲਾਂ ਸਮੀਖਿਆ।

OpenDots ਤੋਂ ਲਏ ਤਿੰਨ ਵਿਚਾਰ (MIT; ਆਪਣਾ Python-ਕੋਡ):
  1. Page = revisioned ਦਸਤਾਵੇਜ਼ — ਹਰ ਸੇਵ ਨਾਲ revision ਵਧਦਾ
  2. expectedRevision — ਆਸ਼ਾਵਾਦੀ ਤਾਲਾ: ਦੋ ਪਾਸੇ ਇਕੱਠੇ ਸੇਵ ਕਰਨ ਤਾਂ
     ਪੁਰਾਣੇ revision ਵਾਲਾ 409-ਟਕਰਾਅ ਖਾਂਦਾ ਹੈ (ਚੁੱਪਚਾਪ overwrite ਨਹੀਂ)
  3. review-first — ਲਿਖਤ ਸਿੱਧੀ ਡਿਸਕ ਤੇ ਨਹੀਂ: ਪਹਿਲਾਂ ਡਰਾਫ਼ਟ ਕਤਾਰ ਵਿਚ,
     ਮਨਜ਼ੂਰੀ ਮਗਰੋਂ ਹੀ page ਬਣਦਾ (OpenDots ਦਾ review_space_page ਪੈਟਰਨ)

ਇਹ ਸਾਡੀ ApprovalGate ਤੋਂ ਵੱਖ ਹੈ — ਉਹ ਕਦਮ-ਪੱਧਰ ਦੀ ਮਨਜ਼ੂਰੀ ਹੈ,
ਇਹ *ਸਮਗਰੀ*-ਪੱਧਰ ਦੀ: ਲਿਖਿਆ ਕੀ ਜਾ ਰਿਹਾ, ਉਹ ਪਹਿਲਾਂ ਵੇਖੋ।
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional


class PageError(Exception):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


class ConflictError(PageError):
    def __init__(self, page_id: str, expected: int, actual: int):
        super().__init__(
            f"ਟਕਰਾਅ: page '{page_id}' revision {actual} ਹੈ, "
            f"ਤੁਸੀਂ {expected} ਤੋਂ ਲਿਖ ਰਹੇ ਸੀ — ਮੁੜ ਪੜ੍ਹੋ", 409)


class Pages:
    def __init__(self, path: Path):
        self.path = path
        self.pages: Dict[str, dict] = {}
        self.drafts: Dict[str, dict] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.pages = data.get("pages", {})
                self.drafts = data.get("drafts", {})
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self) -> None:
        self.path.write_text(json.dumps(
            {"pages": self.pages, "drafts": self.drafts},
            ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- ਡਰਾਫ਼ਟ-ਪਹਿਲਾਂ ਲਿਖਤ ----
    def draft(self, title: str, content: str,
              page_id: Optional[str] = None) -> dict:
        """ਨਵਾਂ page ਜਾਂ ਮੌਜੂਦਾ ਦਾ ਸੁਧਾਰ — ਦੋਵੇਂ ਪਹਿਲਾਂ ਡਰਾਫ਼ਟ।"""
        if not title.strip():
            raise PageError("title ਲਾਜ਼ਮੀ ਹੈ")
        if len(content) > 100_000:
            raise PageError("content 100k ਤੋਂ ਵੱਧ")
        if page_id and page_id not in self.pages:
            raise PageError(f"page '{page_id}' ਨਹੀਂ ਮਿਲਾ", 404)
        d = {"id": f"draft_{uuid.uuid4().hex[:10]}",
             "page_id": page_id,
             "title": title.strip()[:160], "content": content,
             "base_revision": self.pages[page_id]["revision"]
                              if page_id else 0,
             "created_at": time.time(), "status": "pending"}
        self.drafts[d["id"]] = d
        self._save()
        return d

    def approve(self, draft_id: str) -> dict:
        d = self.drafts.get(draft_id)
        if not d:
            raise PageError(f"draft '{draft_id}' ਨਹੀਂ ਮਿਲਿਆ", 404)
        if d["status"] != "pending":
            raise PageError(f"draft {d['status']} ਹੋ ਚੁੱਕਾ", 409)
        if d["page_id"]:
            pg = self.pages[d["page_id"]]
            if pg["revision"] != d["base_revision"]:
                d["status"] = "conflicted"
                self._save()
                raise ConflictError(d["page_id"], d["base_revision"],
                                    pg["revision"])
            pg["content"] = d["content"]
            pg["title"] = d["title"]
            pg["revision"] += 1
            pg["updated_at"] = time.time()
        else:
            pid = f"page_{uuid.uuid4().hex[:10]}"
            self.pages[pid] = {"id": pid, "title": d["title"],
                               "content": d["content"], "revision": 1,
                               "created_at": time.time(),
                               "updated_at": time.time()}
            d["page_id"] = pid
        d["status"] = "approved"
        self._save()
        return {"page_id": d["page_id"], "revision":
                self.pages[d["page_id"]]["revision"], "title": d["title"]}

    def reject(self, draft_id: str) -> dict:
        d = self.drafts.get(draft_id)
        if not d:
            raise PageError(f"draft '{draft_id}' ਨਹੀਂ ਮਿਲਿਆ", 404)
        d["status"] = "rejected"
        self._save()
        return {"draft_id": draft_id, "status": "rejected"}

    # ---- ਪੜ੍ਹਨਾ ----
    def read(self, page_id: str) -> dict:
        pg = self.pages.get(page_id)
        if not pg:
            raise PageError(f"page '{page_id}' ਨਹੀਂ ਮਿਲਾ", 404)
        return dict(pg)

    def list_pages(self) -> List[dict]:
        return [{"id": p["id"], "title": p["title"],
                 "revision": p["revision"]}
                for p in sorted(self.pages.values(),
                                key=lambda x: x["updated_at"],
                                reverse=True)]

    def pending_drafts(self) -> List[dict]:
        return [{"id": d["id"], "title": d["title"],
                 "page_id": d["page_id"],
                 "preview": d["content"][:120]}
                for d in self.drafts.values() if d["status"] == "pending"]

    def stats(self) -> dict:
        return {"pages": len(self.pages),
                "pending_drafts": len(self.pending_drafts()),
                "total_revisions": sum(p["revision"]
                                       for p in self.pages.values())}
