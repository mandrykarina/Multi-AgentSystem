from __future__ import annotations

import json
from pathlib import Path


EXPECTED = {
    "dataset_path": "pmr_dataset_100_with_answers.json",
    "selected_tasks": 100,
    "runs": 3,
    "seeds": [42, 43, 44],
}


def main() -> int:
    summary_path = Path("results/summary.json")
    if not summary_path.exists():
        print(f"summary_not_found={summary_path.as_posix()}")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "dataset_path_ok": Path(str(summary.get("dataset_path", ""))).name == EXPECTED["dataset_path"],
        "selected_tasks_ok": int(summary.get("selected_tasks", 0) or 0) == EXPECTED["selected_tasks"],
        "runs_ok": int(summary.get("runs", 0) or 0) == EXPECTED["runs"],
        "seeds_ok": list(summary.get("seeds", [])) == EXPECTED["seeds"],
        "failed_zero": int(summary.get("failed", 0) or 0) == 0,
        "expert_report_available": bool(summary.get("expert_report_available")),
        "stats_report_available": bool(summary.get("stats_report_available")),
    }
    for key, value in checks.items():
        print(f"{key}={str(value).lower()}")
    print(f"all_ok={str(all(checks.values())).lower()}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
