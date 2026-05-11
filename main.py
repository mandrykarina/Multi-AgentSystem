from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pmr_api import run_agent1, run_pipeline, run_pipeline_v3


DEFAULT_DATASET_PATH = "pmr_dataset_30_with_answers.json"


def _join_argv_parts(parts: list[str]) -> str:
    return " ".join(p for p in parts).strip()


def _has_flag(argv: list[str], flag: str) -> bool:
    return flag in argv


def _get_task_text_from_argv(argv: list[str]) -> str:
    """
    Для режима Агент 1 ожидаем: `python main.py "<task_text>"`
    """
    if len(argv) >= 2:
        candidate = _join_argv_parts(argv[1:])
        if candidate:
            return candidate
    return (
        "Пусть студенту нужно написать алгоритм классификации текста на 3 категории. "
        "Предложи процедуру решения: какие шаги и альтернативы."
    )


def _parse_dataset_mode(argv: list[str]) -> tuple[str, bool, set[str]]:
    """
    Парсинг режима:
    - python main.py --dataset --all
    - python main.py --dataset <path> --all
    - python main.py --dataset <path> --nums S1,S4
    """
    args = argv[2:]
    if not args:
        raise ValueError("Для `--dataset` укажите `--all` или `--nums`.")

    dataset_path = DEFAULT_DATASET_PATH
    run_all = False
    selected_ids: set[str] = set()

    idx = 0
    # Если первый токен не флаг, считаем что это путь к датасету.
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
        raise ValueError(f"Неизвестный аргумент режима датасета: {token}")

    if run_all and selected_ids:
        raise ValueError("Нельзя одновременно использовать `--all` и `--nums`.")
    if not run_all and not selected_ids:
        raise ValueError("Укажите один режим выбора задач: `--all` или `--nums`.")

    return dataset_path, run_all, selected_ids


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


def _run_dataset_mode(dataset_path: str, run_all: bool, selected_ids: set[str]) -> int:
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
            print(
                f"Ошибка: в датасете не найдены id: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 1
        selected = [by_id[id_] for id_ in sorted(selected_ids)]

    results: list[dict[str, Any]] = []
    success_count = 0
    failed_count = 0

    for item in selected:
        task_id = item["id"]
        task_text = item["task"]
        dataset_answer = item["answer"]

        try:
            pipeline_result = run_pipeline(task_text)
            agent2_result = pipeline_result["agent2_result"]
            ai_final_answer = str(agent2_result.get("final_answer", ""))

            report = {
                "id": task_id,
                "task": task_text,
                "ai_final_answer": ai_final_answer,
                "dataset_answer": dataset_answer,
                "status": "ok",
            }
            success_count += 1
        except Exception as e:
            report = {
                "id": task_id,
                "task": task_text,
                "status": "failed",
                "error": str(e),
                "dataset_answer": dataset_answer,
            }
            failed_count += 1

        results.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))

    summary = {
        "dataset_path": dataset_path,
        "total": len(selected),
        "ok": success_count,
        "failed": failed_count,
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0 if failed_count == 0 else 1


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--dataset":
        try:
            dataset_path, run_all, selected_ids = _parse_dataset_mode(argv)
        except Exception as e:
            print(f"Ошибка аргументов: {e}", file=sys.stderr)
            return 1
        return _run_dataset_mode(dataset_path, run_all, selected_ids)

    if len(argv) >= 2 and argv[1] == "--agent2":
        # Ожидаем: python main.py --agent2 "<task_text>"
        task_parts = [part for part in argv[2:] if part != "--with-agent3"]
        task_text = _join_argv_parts(task_parts) or _get_task_text_from_argv(argv)
        try:
            if _has_flag(argv, "--with-agent3"):
                pipeline_result = run_pipeline_v3(task_text)
                result = pipeline_result["agent3_result"]
            else:
                pipeline_result = run_pipeline(task_text)
                result = pipeline_result["agent2_result"]
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 1

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if len(argv) >= 2 and argv[1] == "--agent3":
        # Ожидаем: python main.py --agent3 "<task_text>"
        task_text = _join_argv_parts(argv[2:]) or _get_task_text_from_argv(argv)
        try:
            pipeline_result = run_pipeline_v3(task_text)
            result = pipeline_result["agent3_result"]
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 1

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    task_text = _get_task_text_from_argv(argv)
    try:
        result: dict[str, Any] = run_agent1(task_text)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

