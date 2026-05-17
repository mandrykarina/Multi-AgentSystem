from __future__ import annotations

import csv
import itertools
import statistics
from pathlib import Path
from typing import Any

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None  # type: ignore[assignment]


EXPERT_METRICS_BOUNDS: dict[str, tuple[int, int]] = {
    "content_correctness": (0, 2),
    "choice_justification": (1, 5),
    "alternative_depth": (1, 5),
}


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values))}


def _weighted_kappa_linear(rater_a: list[int], rater_b: list[int], min_value: int, max_value: int) -> float:
    if len(rater_a) != len(rater_b) or not rater_a:
        return 0.0
    categories = list(range(min_value, max_value + 1))
    k = len(categories)
    if k <= 1:
        return 1.0

    idx = {value: i for i, value in enumerate(categories)}
    n = len(rater_a)

    # Linear disagreement weights in [0, 1].
    weights: list[list[float]] = []
    for i in range(k):
        row: list[float] = []
        for j in range(k):
            row.append(abs(i - j) / float(k - 1))
        weights.append(row)

    obs = 0.0
    for a, b in zip(rater_a, rater_b):
        obs += weights[idx[a]][idx[b]]
    obs /= float(n)

    pa = [0.0] * k
    pb = [0.0] * k
    for a in rater_a:
        pa[idx[a]] += 1.0
    for b in rater_b:
        pb[idx[b]] += 1.0
    pa = [x / n for x in pa]
    pb = [x / n for x in pb]

    exp = 0.0
    for i in range(k):
        for j in range(k):
            exp += weights[i][j] * pa[i] * pb[j]

    if exp == 0.0:
        return 1.0
    return float(1.0 - (obs / exp))


def _fleiss_kappa(counts_per_item: list[list[int]]) -> float:
    if not counts_per_item:
        return 0.0
    n_items = len(counts_per_item)
    n_raters = sum(counts_per_item[0])
    if n_raters <= 1:
        return 1.0

    p_j: list[float] = [0.0] * len(counts_per_item[0])
    for row in counts_per_item:
        for j, count in enumerate(row):
            p_j[j] += count
    p_j = [x / float(n_items * n_raters) for x in p_j]

    p_i: list[float] = []
    for row in counts_per_item:
        numerator = sum(count * (count - 1) for count in row)
        p_i.append(numerator / float(n_raters * (n_raters - 1)))

    p_bar = sum(p_i) / float(n_items)
    p_e_bar = sum(x * x for x in p_j)
    denom = 1.0 - p_e_bar
    if denom == 0.0:
        return 1.0
    return float((p_bar - p_e_bar) / denom)


def load_expert_scores(path: Path | str) -> list[dict[str, Any]]:
    scores_path = Path(path)
    if not scores_path.exists():
        return []

    with scores_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not row:
            continue
        item_id = str(row.get("id", "")).strip()
        mode = str(row.get("mode", "")).strip()
        expert_id = str(row.get("expert_id", "")).strip()
        if not item_id or not mode or not expert_id:
            continue
        record: dict[str, Any] = {"id": item_id, "mode": mode, "expert_id": expert_id}
        valid = True
        for metric, (lo, hi) in EXPERT_METRICS_BOUNDS.items():
            raw = row.get(metric)
            if raw is None or str(raw).strip() == "":
                valid = False
                break
            value = int(str(raw))
            if value < lo or value > hi:
                raise ValueError(f"Некорректная экспертная оценка `{metric}`={value}; ожидается [{lo}, {hi}]")
            record[metric] = value
        if valid:
            parsed.append(record)
    return parsed


def build_expert_report(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {"available": False, "reason": "results/expert_scores.csv not found or empty"}

    by_item_mode: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in scores:
        key = (row["id"], row["mode"])
        by_item_mode.setdefault(key, []).append(row)

    per_item_mode_median: list[dict[str, Any]] = []
    for (item_id, mode), rows in sorted(by_item_mode.items()):
        medians: dict[str, float] = {}
        for metric in EXPERT_METRICS_BOUNDS:
            values = [int(r[metric]) for r in rows]
            medians[metric] = float(statistics.median(values))
        per_item_mode_median.append({"id": item_id, "mode": mode, **medians})

    by_mode: dict[str, dict[str, list[float]]] = {}
    for row in per_item_mode_median:
        mode = str(row["mode"])
        mode_bucket = by_mode.setdefault(
            mode,
            {metric: [] for metric in EXPERT_METRICS_BOUNDS},
        )
        for metric in EXPERT_METRICS_BOUNDS:
            mode_bucket[metric].append(float(row[metric]))

    summary_by_mode: dict[str, Any] = {}
    for mode, metric_values in by_mode.items():
        summary_by_mode[mode] = {
            metric: _mean_std(values) for metric, values in metric_values.items()
        }

    expert_ids = sorted({str(r["expert_id"]) for r in scores})

    agreement: dict[str, Any] = {}
    for metric, (lo, hi) in EXPERT_METRICS_BOUNDS.items():
        categories = list(range(lo, hi + 1))
        counts_per_item: list[list[int]] = []
        pairwise_values: list[float] = []

        for (_, _), rows in by_item_mode.items():
            if len(rows) < 2:
                continue
            values = [int(r[metric]) for r in rows]
            row_counts = [0] * len(categories)
            for value in values:
                row_counts[value - lo] += 1
            counts_per_item.append(row_counts)

        for expert_a, expert_b in itertools.combinations(expert_ids, 2):
            vals_a: list[int] = []
            vals_b: list[int] = []
            for rows in by_item_mode.values():
                by_expert = {str(r["expert_id"]): int(r[metric]) for r in rows}
                if expert_a in by_expert and expert_b in by_expert:
                    vals_a.append(by_expert[expert_a])
                    vals_b.append(by_expert[expert_b])
            if vals_a and vals_b:
                pairwise_values.append(_weighted_kappa_linear(vals_a, vals_b, lo, hi))

        agreement[metric] = {
            "fleiss_kappa": _fleiss_kappa(counts_per_item),
            "pairwise_weighted_kappa_mean": (
                float(statistics.mean(pairwise_values)) if pairwise_values else 0.0
            ),
            "item_count": len(counts_per_item),
        }

    return {
        "available": True,
        "records_count": len(scores),
        "item_mode_count": len(by_item_mode),
        "per_item_mode_median": per_item_mode_median,
        "summary_by_mode": summary_by_mode,
        "agreement": agreement,
    }


def build_stat_tests_report(run_reports: list[dict[str, Any]], reproducibility_by_mode: dict[str, float]) -> dict[str, Any]:
    if not run_reports:
        return {"available": False, "reason": "run_reports is empty"}
    if scipy_stats is None:
        return {"available": False, "reason": "scipy is unavailable"}

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in run_reports:
        by_mode.setdefault(str(row.get("mode", "")), []).append(row)

    baseline_mode = "pmr"
    comparisons = [mode for mode in ("direct", "cot", "ps") if mode in by_mode]

    metrics = ("rouge_l", "total_tokens", "latency_sec")
    tests: list[dict[str, Any]] = []

    for other in comparisons:
        if baseline_mode not in by_mode:
            continue
        base_rows = by_mode[baseline_mode]
        other_rows = by_mode[other]
        for metric in metrics:
            base_values = [float(r.get(metric, 0.0) or 0.0) for r in base_rows]
            other_values = [float(r.get(metric, 0.0) or 0.0) for r in other_rows]
            if len(base_values) < 2 or len(other_values) < 2:
                continue

            t_res = scipy_stats.ttest_ind(base_values, other_values, equal_var=False)
            u_res = scipy_stats.mannwhitneyu(base_values, other_values, alternative="two-sided")

            tests.append(
                {
                    "metric": metric,
                    "group_a": baseline_mode,
                    "group_b": other,
                    "t_test": {"statistic": float(t_res.statistic), "p_value": float(t_res.pvalue)},
                    "mann_whitney_u": {"statistic": float(u_res.statistic), "p_value": float(u_res.pvalue)},
                    "mean_a": float(statistics.mean(base_values)),
                    "mean_b": float(statistics.mean(other_values)),
                }
            )

    if not tests:
        return {
            "available": False,
            "reason": "insufficient sample size for statistical tests",
            "reproducibility_by_mode": reproducibility_by_mode,
        }
    return {"available": True, "tests": tests, "reproducibility_by_mode": reproducibility_by_mode}
