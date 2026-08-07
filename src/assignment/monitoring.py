"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def check_metrics(self) -> list[Alert]:
        """Tính các tỉ lệ và phát Alert khi vượt ngưỡng.

        Ba tín hiệu bất thường mà một agent bị tấn công thường bộc lộ:
          - block_rate cao đột biến → có thể đang bị dội một loạt injection.
          - rate_limit_hits nhiều → flooding / cost attack.
          - judge_fail_rate cao → model đang sinh nội dung không an toàn
            (bị prompt injection dẫn dắt, hoặc hallucination hàng loạt).

        Gọi lại được nhiều lần: mỗi lần tính lại từ đầu để không nhân đôi alert.
        """
        self.alerts = []

        if self.total_requests:
            block_rate = self.blocked_requests / self.total_requests
            if block_rate > self.block_rate_threshold:
                self.alerts.append(Alert(
                    metric="block_rate",
                    value=round(block_rate, 3),
                    threshold=self.block_rate_threshold,
                    message=(
                        f"Tỉ lệ chặn {block_rate:.0%} vượt ngưỡng "
                        f"{self.block_rate_threshold:.0%} — nghi ngờ bị dội tấn công."
                    ),
                ))

        if self.rate_limit_hits > self.rate_limit_hit_threshold:
            self.alerts.append(Alert(
                metric="rate_limit_hits",
                value=self.rate_limit_hits,
                threshold=self.rate_limit_hit_threshold,
                message=(
                    f"{self.rate_limit_hits} lần chạm rate limit vượt ngưỡng "
                    f"{self.rate_limit_hit_threshold} — nghi flooding / cost attack."
                ),
            ))

        if self.judge_checks:
            judge_fail_rate = self.judge_fails / self.judge_checks
            if judge_fail_rate > self.judge_fail_rate_threshold:
                self.alerts.append(Alert(
                    metric="judge_fail_rate",
                    value=round(judge_fail_rate, 3),
                    threshold=self.judge_fail_rate_threshold,
                    message=(
                        f"Judge fail {judge_fail_rate:.0%} vượt ngưỡng "
                        f"{self.judge_fail_rate_threshold:.0%} — output kém an toàn."
                    ),
                ))

        return self.alerts

    def export_json(self, filepath: str = "outputs/metrics.json"):
        """Ghi snapshot metrics + alerts ra JSON."""
        # Đảm bảo alerts đã được tính trước khi xuất.
        self.check_metrics()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
