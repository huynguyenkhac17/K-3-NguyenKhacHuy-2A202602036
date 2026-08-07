"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


class AuditLogPlugin:
    """Ghi nhật ký mọi tương tác để điều tra sự cố (forensics).

    Bản thân lớp này KHÔNG chặn gì cả — các lớp guardrail mới là nơi chặn. Vai
    trò của audit log là làm cho những quyết định đó *truy vết được*: mỗi request
    có một ``request_id`` (correlation ID) nối liền input → output → quyết định
    của lớp nào, cùng độ trễ. Không có nó thì lúc incident không ai biết prompt
    nào đã lọt và bị lớp nào bỏ sót.
    """

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        # request_id -> (text đầu vào, mốc bắt đầu) để ghép input với output
        # và tính latency khi output tới.
        self._open: dict[str, dict] = {}

    def new_request_id(self) -> str:
        """Sinh correlation ID mới cho một lượt xử lý."""
        return "req-" + uuid.uuid4().hex[:12]

    def record_input(
        self, *, user_id: str, text: str, request_id: str | None = None
    ) -> str:
        """Lưu input + mốc bắt đầu, trả về request_id để dùng xuyên suốt.

        Args:
            user_id: ai gửi (để rate limit / điều tra theo người dùng)
            text: nội dung thô
            request_id: truyền vào nếu đã có, không thì tự sinh

        Returns:
            request_id — nhớ truyền lại vào record_output để nối cặp.
        """
        request_id = request_id or self.new_request_id()
        self._open[request_id] = {
            "user_id": user_id,
            "input": text,
            "start": time.time(),
            "start_iso": utc_now_iso(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Lưu output + lớp quyết định + độ trễ; append vào self.logs."""
        opened = self._open.pop(request_id, None) if request_id else None
        start = opened["start"] if opened else None
        entry = {
            "request_id": request_id,
            "user_id": user_id,
            "input": opened["input"] if opened else None,
            "input_at": opened["start_iso"] if opened else None,
            "output": text,
            "output_at": utc_now_iso(),
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round((time.time() - start) * 1000, 1) if start else None,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str = "outputs/audit_log.json"):
        """Ghi toàn bộ log ra đĩa (mảng JSON)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
