from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis_metrics import build_expert_report, build_stat_tests_report, load_expert_scores
from debug_utils import debug_print
from experiment_log import save_log_entry
from modes import run_cot, run_direct, run_ps
from pmr_api import run_agent1, run_pipeline, run_pipeline_v3, run_pmr

try:
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover
    rouge_scorer = None  # type: ignore[assignment]


DEFAULT_DATASET_CANDIDATES = [
    "pmr_dataset_100_with_answers.json",
    "pmr_dataset_30_with_answers.json",
]
RESULTS_DIR = Path("results")
RESULTS_SMOKE_DIR = Path("results_smoke")
FINAL_DATASET_NAME = "pmr_dataset_100_with_answers.json"
FINAL_RUNS = 3
FINAL_SEEDS = [42, 43, 44]


def _join_argv_parts(parts: list[str]) -> str:
    return " ".join(p for p in parts).strip()


def _strip_debug_flag(parts: list[str]) -> list[str]:
    return [p for p in parts if p != "--debug"]


def _has_flag(argv: list[str], flag: str) -> bool:
    return flag in argv


def _parse_seeds(raw: str) -> list[int]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("Список `--seeds` пуст.")
    return [int(x) for x in items]


def _default_dataset_path() -> str:
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return DEFAULT_DATASET_CANDIDATES[-1]


def _get_task_text_from_argv(argv: list[str]) -> str:
    if len(argv) >= 2:
        candidate = _join_argv_parts(_strip_debug_flag(argv[1:]))
        if candidate:
            return candidate
    return (
        "Пусть студенту нужно написать алгоритм классификации текста на 3 категории. "
        "Предложи процедуру решения: какие шаги и альтернативы."
    )


def _parse_dataset_mode(argv: list[str]) -> tuple[str, bool, set[str], int, list[int]]:
    args = _strip_debug_flag(argv[2:])
    if not args:
        raise ValueError("Для `--dataset` укажите `--all` или `--nums`.")

    dataset_path = _default_dataset_path()
    run_all = False
    selected_ids: set[str] = set()
    runs = 3
    seeds = [42, 43, 44]

    idx = 0
    if not args[0].startswith("--"):
        dataset_path = args[0]
        idx = 1

    while idx < len(args):
        token = args[idx]
        if token == "--all":
            run_all = True
            idx += 1
            continue
        if token == "--nums":
            if idx + 1 >= len(args):
                raise ValueError("Для `--nums` нужно указать список id, например: S1,S4,S10")
            raw = args[idx + 1]
            parsed = {x.strip() for x in raw.split(",") if x.strip()}
            if not parsed:
                raise ValueError("Список id после `--nums` пуст.")
            selected_ids.update(parsed)
            idx += 2
            continue
        if token == "--runs":
            if idx + 1 >= len(args):
                raise ValueError("Для `--runs` нужно указать целое число.")
            runs = int(args[idx + 1])
            if runs <= 0:
                raise ValueError("`--runs` должен быть > 0.")
            idx += 2
            continue
        if token == "--seeds":
            if idx + 1 >= len(args):
                raise ValueError("Для `--seeds` нужно указать список через запятую.")
            seeds = _parse_seeds(args[idx + 1])
            idx += 2
            continue
        raise ValueError(f"Неизвестный аргумент режима датасета: {token}")

    if run_all and selected_ids:
        raise ValueError("Нельзя одновременно использовать `--all` и `--nums`.")
    if not run_all and not selected_ids:
        raise ValueError("Укажите один режим выбора задач: `--all` или `--nums`.")
    if runs > len(seeds):
        raise ValueError("`--runs` не должен быть больше числа seed в `--seeds`.")

    return dataset_path, run_all, selected_ids, runs, seeds


def _load_dataset(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {path_str}")

    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Датасет должен быть JSON-массивом объектов.")

    validated: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Элемент датасета #{i} должен быть объектом.")
        for req in ("id", "task", "answer"):
            if req not in item or not isinstance(item[req], str):
                raise ValueError(f"Элемент #{i}: поле `{req}` обязательно и должно быть строкой.")
        validated.append(item)
    return validated


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _rouge_l(a: str, b: str) -> float:
    if rouge_scorer is None:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(scorer.score(a, b)["rougeL"].fmeasure)


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
    }


def _reproducibility_score(answers: list[str]) -> float:
    if len(answers) <= 1:
        return 1.0
    min_pair = 1.0
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            score = _rouge_l(_normalize_text(answers[i]), _normalize_text(answers[j]))
            min_pair = min(min_pair, score)
    return 1.0 if min_pair >= 0.99 else 0.0


def _task_mode_reproducibility(answers: list[str], expected_runs: int) -> float:
    # По статье задача считается воспроизводимой только при совпадении всех seed-запусков.
    if len(answers) != expected_runs:
        return 0.0
    return _reproducibility_score(answers)


def _ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _ensure_smoke_results_dir() -> Path:
    RESULTS_SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_SMOKE_DIR


def _is_final_article_run(dataset_path: str, run_all: bool, runs: int, seeds_to_use: list[int]) -> bool:
    return (
        run_all
        and Path(dataset_path).name == FINAL_DATASET_NAME
        and runs == FINAL_RUNS
        and seeds_to_use == FINAL_SEEDS
    )


def _mean_std_str(stat: dict[str, Any]) -> str:
    mean_value = float(stat.get("mean", 0.0) or 0.0)
    std_value = float(stat.get("std", 0.0) or 0.0)
    return f"{mean_value:.4f} ± {std_value:.4f}"


def _write_table4_full_csv(path: Path, summary_by_mode: dict[str, Any], expert_report: dict[str, Any]) -> None:
    modes = ("direct", "cot", "ps", "pmr")
    lines = ["metric,direct,cot,ps,pmr"]

    def values_from_summary(key: str) -> list[str]:
        out: list[str] = []
        for mode in modes:
            mode_entry = summary_by_mode.get(mode, {})
            if key in ("rouge_l", "total_tokens", "latency_sec"):
                out.append(_mean_std_str(mode_entry.get(key, {})))
            elif key == "reproducibility":
                out.append(f"{float(mode_entry.get('reproducibility', 0.0) or 0.0):.4f}")
            elif key == "total_tokens_normalized_to_direct":
                out.append(f"{float(mode_entry.get('total_tokens_normalized_to_direct', 0.0) or 0.0):.4f}")
            elif key == "valid_json_rate":
                if mode == "pmr":
                    out.append(f"{float(mode_entry.get('valid_json_rate', 0.0) or 0.0):.4f}")
                else:
                    out.append("—")
        return out

    lines.append(",".join(["ROUGE-L", *values_from_summary("rouge_l")]))

    expert_summary = expert_report.get("summary_by_mode", {}) if isinstance(expert_report, dict) else {}
    for metric_name, metric_key in (
        ("Содержательная корректность", "content_correctness"),
        ("Обоснованность выбора", "choice_justification"),
        ("Глубина альтернатив", "alternative_depth"),
    ):
        row: list[str] = [metric_name]
        for mode in modes:
            if mode in expert_summary and metric_key in expert_summary[mode]:
                row.append(_mean_std_str(expert_summary[mode][metric_key]))
            else:
                row.append("—")
        lines.append(",".join(row))

    lines.append(",".join(["Воспроизводимость", *values_from_summary("reproducibility")]))
    lines.append(
        ",".join(
            [
                "Суммарное число токенов, нормализованное",
                *values_from_summary("total_tokens_normalized_to_direct"),
            ]
        )
    )
    lines.append(",".join(["Задержка ответа", *values_from_summary("latency_sec")]))
    lines.append(",".join(["Доля корректных JSON-ответов", *values_from_summary("valid_json_rate")]))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_runtime_log_to_results(target_dir: Path) -> None:
    source = Path("experiment.log.jsonl")
    if source.exists():
        target = target_dir / "experiment.log.jsonl"
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _build_final_run_validation(summary: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "dataset_path": FINAL_DATASET_NAME,
        "selected_tasks": 100,
        "runs": FINAL_RUNS,
        "seeds": FINAL_SEEDS,
    }
    checks = {
        "dataset_path_ok": Path(str(summary.get("dataset_path", ""))).name == expected["dataset_path"],
        "selected_tasks_ok": int(summary.get("selected_tasks", 0) or 0) == expected["selected_tasks"],
        "runs_ok": int(summary.get("runs", 0) or 0) == expected["runs"],
        "seeds_ok": list(summary.get("seeds", [])) == expected["seeds"],
        "failed_zero": int(summary.get("failed", 0) or 0) == 0,
    }
    checks["all_ok"] = all(checks.values())
    return checks


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_table4_csv(path: Path, summary_by_mode: dict[str, Any]) -> None:
    lines = [
        "mode,count,rouge_l_mean,rouge_l_std,total_tokens_mean,total_tokens_std,total_tokens_normalized_to_direct,"
        "latency_sec_mean,latency_sec_std,reproducibility,valid_json_rate"
    ]
    for mode in ("direct", "cot", "ps", "pmr"):
        row = summary_by_mode.get(mode, {})
        lines.append(
            ",".join(
                [
                    mode,
                    str(row.get("count", 0)),
                    str(row.get("rouge_l", {}).get("mean", 0.0)),
                    str(row.get("rouge_l", {}).get("std", 0.0)),
                    str(row.get("total_tokens", {}).get("mean", 0.0)),
                    str(row.get("total_tokens", {}).get("std", 0.0)),
                    str(row.get("total_tokens_normalized_to_direct", 0.0)),
                    str(row.get("latency_sec", {}).get("mean", 0.0)),
                    str(row.get("latency_sec", {}).get("std", 0.0)),
                    str(row.get("reproducibility", 0.0)),
                    str(row.get("valid_json_rate", "")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_log_entry(
    item: dict[str, Any],
    system_type: str,
    seed: int,
    result: dict[str, Any],
    reference_answer: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": item.get("id", ""),
        "difficulty": item.get("difficulty", ""),
        "domain": item.get("domain", ""),
        "task": item.get("task", ""),
        "system_type": system_type,
        "model": result.get("model", ""),
        "seed": seed,
        "temperature": result.get("temperature"),
        "max_tokens": result.get("max_tokens"),
        "raw_response": result.get("raw_response", ""),
        "parsed_response": result.get("parsed_response"),
        "final_answer": result.get("final_answer", ""),
        "reference_answer": reference_answer,
        "prompt_tokens": int(result.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(result.get("completion_tokens", 0) or 0),
        "total_tokens": int(result.get("total_tokens", 0) or 0),
        "latency_sec": float(result.get("latency_sec", 0.0) or 0.0),
        "retry_count": int(result.get("retry_count", 0) or 0),
        "parse_status": result.get("parse_status", "ok"),
        "timestamp": _iso_now(),
        "api_endpoint": result.get("api_endpoint"),
        "api_type": result.get("api_type"),
        "sdk_name": result.get("sdk_name"),
        "sdk_version": result.get("sdk_version"),
        "request_model": result.get("request_model"),
        "response_model": result.get("response_model"),
        "system_fingerprint": result.get("system_fingerprint"),
        "response_created": result.get("response_created"),
        "response_format": result.get("response_format"),
        "finish_reason": result.get("finish_reason"),
    }
    if system_type == "pmr":
        entry["agent1_result"] = result.get("agent1_result")
        entry["agent2_result"] = result.get("agent2_result")
        entry["agent3_result"] = result.get("agent3_result")
    return entry


def _build_failed_log_entry(
    item: dict[str, Any],
    system_type: str,
    seed: int,
    reference_answer: str,
    error_text: str,
) -> dict[str, Any]:
    return {
        "id": item.get("id", ""),
        "difficulty": item.get("difficulty", ""),
        "domain": item.get("domain", ""),
        "task": item.get("task", ""),
        "system_type": system_type,
        "model": "",
        "seed": seed,
        "temperature": None,
        "max_tokens": None,
        "raw_response": "",
        "parsed_response": None,
        "final_answer": "",
        "reference_answer": reference_answer,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_sec": 0.0,
        "retry_count": 0,
        "parse_status": "failed",
        "timestamp": _iso_now(),
        "error": error_text,
        "api_endpoint": None,
        "api_type": None,
        "sdk_name": None,
        "sdk_version": None,
        "request_model": None,
        "response_model": None,
        "system_fingerprint": None,
        "response_created": None,
        "response_format": None,
        "finish_reason": None,
    }


def _run_dataset_mode(
    dataset_path: str,
    run_all: bool,
    selected_ids: set[str],
    runs: int,
    seeds: list[int],
    *,
    debug: bool = False,
) -> int:
    try:
        rows = _load_dataset(dataset_path)
    except Exception as e:
        print(f"Ошибка чтения датасета: {e}", file=sys.stderr)
        return 1

    if run_all:
        selected = rows
    else:
        by_id = {row["id"]: row for row in rows}
        missing = [id_ for id_ in sorted(selected_ids) if id_ not in by_id]
        if missing:
            print(f"Ошибка: в датасете не найдены id: {', '.join(missing)}", file=sys.stderr)
            return 1
        selected = [by_id[id_] for id_ in sorted(selected_ids)]

    seeds_to_use = seeds[:runs]
    is_final_run = _is_final_article_run(dataset_path, run_all, runs, seeds_to_use)
    out_dir = _ensure_results_dir() if is_final_run else _ensure_smoke_results_dir()
    run_reports: list[dict[str, Any]] = []
    failed_count = 0
    mode_attempts: dict[str, int] = {m: 0 for m in ("direct", "cot", "ps", "pmr")}
    pmr_valid_json_runs = 0
    reproducibility_values: dict[str, list[float]] = {m: [] for m in ("direct", "cot", "ps", "pmr")}

    mode_runners = {"direct": run_direct, "cot": run_cot, "ps": run_ps, "pmr": run_pmr}

    for item in selected:
        task_text = item["task"]
        reference_answer = item["answer"]
        task_id = str(item.get("id", ""))
        debug_print(debug, "dataset.task_start", task_id=task_id, payload={"runs": runs, "seeds": seeds_to_use})
        per_mode_answers: dict[str, list[str]] = {"direct": [], "cot": [], "ps": [], "pmr": []}

        for seed in seeds_to_use:
            for mode_name, runner in mode_runners.items():
                mode_attempts[mode_name] += 1
                debug_print(debug, "mode.run_start", mode=mode_name, task_id=task_id, seed=seed)
                try:
                    mode_result = runner(task_text, seed=seed, debug=debug)
                    final_answer = str(mode_result.get("final_answer", "")).strip()
                    per_mode_answers[mode_name].append(final_answer)
                    log_entry = _build_log_entry(item, mode_name, seed, mode_result, reference_answer)
                    save_log_entry(log_entry)

                    parse_status = str(mode_result.get("parse_status", "ok"))
                    if mode_name == "pmr" and parse_status in ("ok", "retry_success"):
                        pmr_valid_json_runs += 1

                    run_reports.append(
                        {
                            "id": item["id"],
                            "mode": mode_name,
                            "seed": seed,
                            "rouge_l": _rouge_l(final_answer, reference_answer),
                            "total_tokens": int(mode_result.get("total_tokens", 0) or 0),
                            "latency_sec": float(mode_result.get("latency_sec", 0.0) or 0.0),
                            "parse_status": parse_status,
                            "status": "ok",
                        }
                    )
                    debug_print(
                        debug,
                        "mode.run_success",
                        mode=mode_name,
                        task_id=task_id,
                        seed=seed,
                        payload={
                            "latency_sec": mode_result.get("latency_sec"),
                            "total_tokens": mode_result.get("total_tokens"),
                            "finish_reason": mode_result.get("finish_reason"),
                            "parse_status": parse_status,
                        },
                    )
                except Exception as e:
                    failed_count += 1
                    error_text = str(e)
                    failed_entry = _build_failed_log_entry(
                        item=item,
                        system_type=mode_name,
                        seed=seed,
                        reference_answer=reference_answer,
                        error_text=error_text,
                    )
                    save_log_entry(failed_entry)
                    run_reports.append(
                        {
                            "id": item["id"],
                            "mode": mode_name,
                            "seed": seed,
                            "rouge_l": 0.0,
                            "total_tokens": 0,
                            "latency_sec": 0.0,
                            "parse_status": "failed",
                            "status": "failed",
                        }
                    )
                    print(
                        json.dumps(
                            {"id": item.get("id", ""), "mode": mode_name, "seed": seed, "status": "failed", "error": error_text},
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    debug_print(
                        debug,
                        "mode.run_failed",
                        mode=mode_name,
                        task_id=task_id,
                        seed=seed,
                        payload={"exception_type": type(e).__name__, "message": error_text},
                    )

        repro_report = {
            mode: _task_mode_reproducibility(answers, runs) for mode, answers in per_mode_answers.items()
        }
        for mode, value in repro_report.items():
            reproducibility_values[mode].append(value)

        print(
            json.dumps(
                {"id": item["id"], "status": "ok" if all(len(v) == runs for v in per_mode_answers.values()) else "partial", "reproducibility": repro_report},
                ensure_ascii=False,
                indent=2,
            )
        )
        debug_print(debug, "dataset.task_done", task_id=task_id, payload={"reproducibility": repro_report})

    summary_by_mode: dict[str, Any] = {}
    for mode in ("direct", "cot", "ps", "pmr"):
        mode_rows = [r for r in run_reports if r["mode"] == mode and r["status"] == "ok"]
        rouge_values = [float(r["rouge_l"]) for r in mode_rows]
        token_values = [float(r["total_tokens"]) for r in mode_rows]
        latency_values = [float(r["latency_sec"]) for r in mode_rows]
        entry: dict[str, Any] = {
            "count": len(mode_rows),
            "rouge_l": _mean_std(rouge_values),
            "total_tokens": _mean_std(token_values),
            "latency_sec": _mean_std(latency_values),
            "reproducibility": float(statistics.mean(reproducibility_values[mode])) if reproducibility_values[mode] else 0.0,
        }
        if mode == "pmr":
            total_pmr_attempts = mode_attempts["pmr"]
            entry["valid_json_rate"] = float(pmr_valid_json_runs / total_pmr_attempts) if total_pmr_attempts else 0.0
        else:
            entry["valid_json_rate"] = None
        summary_by_mode[mode] = entry

    direct_token_mean = summary_by_mode.get("direct", {}).get("total_tokens", {}).get("mean", 0.0) or 0.0
    for mode in ("direct", "cot", "ps", "pmr"):
        token_mean = float(summary_by_mode[mode]["total_tokens"]["mean"])
        summary_by_mode[mode]["total_tokens_normalized_to_direct"] = (
            float(token_mean / direct_token_mean) if direct_token_mean else 0.0
        )

    reproducibility_by_mode = {mode: float(summary_by_mode[mode]["reproducibility"]) for mode in ("direct", "cot", "ps", "pmr")}
    expert_scores = load_expert_scores(out_dir / "expert_scores.csv")
    expert_report = build_expert_report(expert_scores)
    stats_report = build_stat_tests_report(run_reports, reproducibility_by_mode)

    summary = {
        "dataset_path": dataset_path,
        "selected_tasks": len(selected),
        "runs": runs,
        "seeds": seeds_to_use,
        "failed": failed_count,
        "is_final_article_run": is_final_run,
        "output_dir": out_dir.as_posix(),
        "attempts_by_mode": mode_attempts,
        "by_mode": summary_by_mode,
        "expert_report_available": bool(expert_report.get("available")),
        "stats_report_available": bool(stats_report.get("available")),
    }
    summary["final_run_validation"] = _build_final_run_validation(summary)
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))

    _write_json(out_dir / "summary.json", summary)
    _write_table4_csv(out_dir / "table4_results.csv", summary_by_mode)
    _write_table4_full_csv(out_dir / "table4_full.csv", summary_by_mode, expert_report)
    _write_json(out_dir / "expert_report.json", expert_report)
    _write_json(out_dir / "stats_report.json", stats_report)
    _copy_runtime_log_to_results(out_dir)

    return 0 if failed_count == 0 else 1


def main(argv: list[str]) -> int:
    debug = _has_flag(argv, "--debug")
    debug_print(debug, "cli.args_parsed", payload={"argv": _strip_debug_flag(argv)})

    if len(argv) >= 2 and argv[1] == "--dataset":
        try:
            dataset_path, run_all, selected_ids, runs, seeds = _parse_dataset_mode(argv)
        except Exception as e:
            print(f"Ошибка аргументов: {e}", file=sys.stderr)
            return 1
        debug_print(
            debug,
            "dataset.mode_selected",
            payload={
                "dataset_path": dataset_path,
                "run_all": run_all,
                "selected_ids": sorted(selected_ids),
                "runs": runs,
                "seeds": seeds,
            },
        )
        return _run_dataset_mode(dataset_path, run_all, selected_ids, runs, seeds, debug=debug)

    if len(argv) >= 2 and argv[1] == "--agent2":
        task_parts = [part for part in argv[2:] if part not in {"--with-agent3", "--debug"}]
        task_text = _join_argv_parts(task_parts) or _get_task_text_from_argv(argv)
        try:
            if _has_flag(argv, "--with-agent3"):
                debug_print(debug, "single.mode_selected", payload={"mode": "agent3_via_agent2"})
                pipeline_result = run_pipeline_v3(task_text, debug=debug)
                result = pipeline_result["agent3_result"]
            else:
                debug_print(debug, "single.mode_selected", payload={"mode": "agent2"})
                pipeline_result = run_pipeline(task_text, debug=debug)
                result = pipeline_result["agent2_result"]
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            debug_print(
                debug,
                "single.mode_failed",
                payload={"mode": "agent2_path", "exception_type": type(e).__name__, "message": str(e)},
            )
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if len(argv) >= 2 and argv[1] == "--agent3":
        task_text = _join_argv_parts(_strip_debug_flag(argv[2:])) or _get_task_text_from_argv(argv)
        try:
            debug_print(debug, "single.mode_selected", payload={"mode": "agent3"})
            pipeline_result = run_pipeline_v3(task_text, debug=debug)
            result = pipeline_result["agent3_result"]
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            debug_print(
                debug,
                "single.mode_failed",
                payload={"mode": "agent3", "exception_type": type(e).__name__, "message": str(e)},
            )
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    task_text = _get_task_text_from_argv(argv)
    try:
        debug_print(debug, "single.mode_selected", payload={"mode": "agent1"})
        result = run_agent1(task_text, debug=debug)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        debug_print(
            debug,
            "single.mode_failed",
            payload={"mode": "agent1", "exception_type": type(e).__name__, "message": str(e)},
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

