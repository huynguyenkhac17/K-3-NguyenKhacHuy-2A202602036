"""Top-up: chạy lại đúng các safe query bị lỗi hạ tầng (429) và vá results.json.

Không chạy lại toàn bộ suite — chỉ những câu hỏi HỢP LỆ bị đánh nhầm 'blocked'
vì Gemini free tier hết quota tức thời. Sau đó tái tạo metrics.json nhất quán
từ results.json cuối cùng để báo cáo trích số ổn định.

    python scripts/topup_results.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

RESULTS = ROOT / "outputs" / "results.json"


def _is_error_row(row: dict) -> bool:
    return str(row.get("response_preview", "")).startswith("Error:")


async def _reprocess(inputs: list[str]) -> list[dict]:
    from assignment.pipeline import (
        build_production_plugins, build_observability, DefensePipeline,
    )
    from agents.agent import create_unsafe_agent

    plugins = build_production_plugins(use_llm_judge=False)
    audit, monitor = build_observability()
    agent, runner = create_unsafe_agent()
    engine = DefensePipeline(agent, runner, plugins, audit, monitor, use_llm_judge=False)

    rows = []
    for i, text in enumerate(inputs):
        if i:
            await asyncio.sleep(40)  # rộng tay để chắc chắn dưới RPM
        rows.append(await engine.process(f"safe-topup-{i}", text))
    return rows


def _rebuild_metrics(data: dict):
    """Tạo lại outputs/metrics.json khớp với results.json cuối cùng."""
    from assignment.monitoring import MonitoringAlert

    all_rows = data["safe_queries"] + data["attack_queries"] + data["edge_cases"]
    total = len(all_rows)
    blocked = sum(1 for r in all_rows if r.get("blocked"))
    judge = data.get("judge_sample", [])
    judge_fails = sum(1 for j in judge if j.get("verdict") == "FAIL")

    monitor = MonitoringAlert(
        total_requests=total,
        blocked_requests=blocked,
        rate_limit_hits=data["rate_limit"]["blocked"],
        judge_checks=len(judge),
        judge_fails=judge_fails,
    )
    monitor.check_metrics()
    monitor.export_json(str(ROOT / "outputs" / "metrics.json"))
    return monitor.snapshot()


async def main():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    err_idx = [i for i, q in enumerate(data["safe_queries"]) if _is_error_row(q)]
    if not err_idx:
        print("Không còn safe query nào lỗi — chỉ tái tạo metrics.")
    else:
        print(f"Chạy lại {len(err_idx)} safe query lỗi: {err_idx}")
        new_rows = await _reprocess([data["safe_queries"][i]["input"] for i in err_idx])
        for pos, i in enumerate(err_idx):
            data["safe_queries"][i] = new_rows[pos]

    RESULTS.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    snap = _rebuild_metrics(data)

    still_err = [i for i, q in enumerate(data["safe_queries"]) if _is_error_row(q)]
    blocked_safe = sum(1 for q in data["safe_queries"] if q.get("blocked"))
    print(f"Safe blocked cuối: {blocked_safe}/{len(data['safe_queries'])} "
          f"(còn lỗi: {still_err})")
    print(f"metrics: block_rate={snap['block_rate']:.2f} "
          f"judge_fail_rate={snap['judge_fail_rate']:.2f} alerts={len(snap['alerts'])}")


if __name__ == "__main__":
    asyncio.run(main())
