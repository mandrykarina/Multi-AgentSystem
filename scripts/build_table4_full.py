from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _mean_std_str(stat: dict[str, Any]) -> str:
    mean_value = float(stat.get("mean", 0.0) or 0.0)
    std_value = float(stat.get("std", 0.0) or 0.0)
    return f"{mean_value:.4f} ± {std_value:.4f}"


def main(argv: list[str]) -> int:
    base_dir = Path(argv[1]) if len(argv) > 1 else Path("results")
    summary_path = base_dir / "summary.json"
    expert_path = base_dir / "expert_report.json"
    if not summary_path.exists():
        print(f"summary_not_found={summary_path.as_posix()}")
        return 1
    if not expert_path.exists():
        print(f"expert_report_not_found={expert_path.as_posix()}")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expert_report = json.loads(expert_path.read_text(encoding="utf-8"))

    modes = ("direct", "cot", "ps", "pmr")
    by_mode = summary.get("by_mode", {})
    expert_summary = expert_report.get("summary_by_mode", {}) if isinstance(expert_report, dict) else {}
    lines = ["metric,direct,cot,ps,pmr"]

    lines.append(
        "ROUGE-L,"
        + ",".join(_mean_std_str(by_mode.get(mode, {}).get("rouge_l", {})) for mode in modes)
    )

    for title, key in (
        ("Содержательная корректность", "content_correctness"),
        ("Обоснованность выбора", "choice_justification"),
        ("Глубина альтернатив", "alternative_depth"),
    ):
        row = [title]
        for mode in modes:
            if mode in expert_summary and key in expert_summary[mode]:
                row.append(_mean_std_str(expert_summary[mode][key]))
            else:
                row.append("—")
        lines.append(",".join(row))

    lines.append(
        "Воспроизводимость,"
        + ",".join(f"{float(by_mode.get(mode, {}).get('reproducibility', 0.0) or 0.0):.4f}" for mode in modes)
    )
    lines.append(
        "Суммарное число токенов, нормализованное,"
        + ",".join(
            f"{float(by_mode.get(mode, {}).get('total_tokens_normalized_to_direct', 0.0) or 0.0):.4f}"
            for mode in modes
        )
    )
    lines.append(
        "Задержка ответа,"
        + ",".join(_mean_std_str(by_mode.get(mode, {}).get("latency_sec", {})) for mode in modes)
    )
    lines.append(
        "Доля корректных JSON-ответов,"
        + ",".join(
            ["—", "—", "—", f"{float(by_mode.get('pmr', {}).get('valid_json_rate', 0.0) or 0.0):.4f}"]
        )
    )

    out_path = base_dir / "table4_full.csv"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written={out_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
