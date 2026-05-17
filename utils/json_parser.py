from __future__ import annotations

import json
from typing import Any


def parse_json_strict(s: str) -> Any:
    """
    Строгое распознавание JSON.

    Никаких преобразований формата не делаем: если JSON невалидный — кидаем ошибку.
    """
    return json.loads(s)


def try_parse_json_from_messy_text(s: str) -> Any:
    """
    Пытаемся вытащить JSON-объект из "мусорного" текста:
    - сначала пробуем json.loads
    - затем извлекаем подстроку между первой `{` и последней `}` и снова json.loads
    """
    s = s.strip()
    try:
        return parse_json_strict(s)
    except json.JSONDecodeError:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # Оригинальная ошибка понятнее, чем "мы не нашли скобки"
        raise ValueError("Ответ модели не содержит JSON-объект (нет корректных фигурных скобок).")

    candidate = s[start : end + 1]
    return parse_json_strict(candidate)


def validate_agent1_output(obj: Any) -> dict[str, Any]:
    """
    Валидируем минимальную схему ответа Агента 1.
    """
    if not isinstance(obj, dict):
        raise ValueError("JSON-ответ должен быть объектом (dict).")

    if "task_type" not in obj or not isinstance(obj["task_type"], str) or not obj["task_type"].strip():
        raise ValueError("Поле `task_type` обязательно и должно быть непустой строкой.")

    if "notes" not in obj or not isinstance(obj["notes"], list) or not all(
        isinstance(x, str) and x.strip() for x in obj["notes"]
    ):
        raise ValueError("Поле `notes` обязательно и должно быть массивом непустых строк.")

    for f in ("task_type",):
        if f not in obj or not isinstance(obj[f], str) or not obj[f].strip():
            raise ValueError(f"Поле `{f}` обязательно и должно быть непустой строкой.")

    if "plan" not in obj or not isinstance(obj["plan"], list) or not all(isinstance(x, str) for x in obj["plan"]):
        raise ValueError("Поле `plan` обязательно и должно быть массивом строк.")

    if "alternatives" not in obj or not isinstance(obj["alternatives"], list):
        raise ValueError("Поле `alternatives` обязательно и должно быть массивом.")

    for i, alt in enumerate(obj["alternatives"]):
        if not isinstance(alt, dict):
            raise ValueError(f"Элемент `alternatives[{i}]` должен быть объектом.")
        if "method" not in alt or not isinstance(alt["method"], str) or not alt["method"].strip():
            raise ValueError(f"Поле `alternatives[{i}].method` обязательно и должно быть непустой строкой.")
        if "rejection_reason" not in alt or not isinstance(alt["rejection_reason"], str) or not alt["rejection_reason"].strip():
            raise ValueError(
                f"Поле `alternatives[{i}].rejection_reason` обязательно и должно быть непустой строкой."
            )

    return obj  # type: ignore[return-value]


def validate_agent2_output(obj: Any) -> dict[str, Any]:
    """
    Минимальная валидация JSON-ответа Агент 2.
    """
    if not isinstance(obj, dict):
        raise ValueError("JSON-ответ Агент 2 должен быть объектом (dict).")

    if "solution_steps" not in obj or not isinstance(obj["solution_steps"], list):
        raise ValueError("Поле `solution_steps` обязательно и должно быть массивом.")

    for i, step_obj in enumerate(obj["solution_steps"]):
        if not isinstance(step_obj, dict):
            raise ValueError(f"Элемент `solution_steps[{i}]` должен быть объектом.")
        if "step_id" not in step_obj or not isinstance(step_obj["step_id"], int):
            raise ValueError(f"Поле `solution_steps[{i}].step_id` обязательно и должно быть int.")
        if "action" not in step_obj or not isinstance(step_obj["action"], str) or not step_obj["action"].strip():
            raise ValueError(f"Поле `solution_steps[{i}].action` обязательно и должно быть непустой строкой.")
        if (
            "plan_alignment" not in step_obj
            or not isinstance(step_obj["plan_alignment"], str)
            or not step_obj["plan_alignment"].strip()
        ):
            raise ValueError(
                f"Поле `solution_steps[{i}].plan_alignment` обязательно и должно быть непустой строкой."
            )
        if "result" not in step_obj or not isinstance(step_obj["result"], str) or not step_obj["result"].strip():
            raise ValueError(
                f"Поле `solution_steps[{i}].result` обязательно и должно быть непустой строкой."
            )

    if "final_answer" not in obj or not isinstance(obj["final_answer"], str) or not obj["final_answer"].strip():
        raise ValueError("Поле `final_answer` обязательно и должно быть непустой строкой.")

    if "adaptation_points" not in obj or not isinstance(obj["adaptation_points"], list) or not all(
        isinstance(x, str) for x in obj["adaptation_points"]
    ):
        raise ValueError("Поле `adaptation_points` обязательно и должно быть массивом строк.")

    return obj  # type: ignore[return-value]


def validate_agent3_output(obj: Any) -> dict[str, Any]:
    """
    Строгая валидация JSON-ответа Агент 3 (Reflective Analyst).
    """
    if not isinstance(obj, dict):
        raise ValueError("JSON-ответ Агент 3 должен быть объектом (dict).")

    aa = obj.get("alignment_assessment")
    if not isinstance(aa, dict):
        raise ValueError("Поле `alignment_assessment` обязательно и должно быть объектом.")
    score = aa.get("score")
    if not isinstance(score, int) or score < 0 or score > 10:
        raise ValueError("Поле `alignment_assessment.score` должно быть целым числом от 0 до 10.")
    explanation = aa.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Поле `alignment_assessment.explanation` обязательно и должно быть непустой строкой.")

    reflections = obj.get("step_reflection")
    if not isinstance(reflections, list):
        raise ValueError("Поле `step_reflection` обязательно и должно быть массивом.")
    for i, item in enumerate(reflections):
        if not isinstance(item, dict):
            raise ValueError(f"Элемент `step_reflection[{i}]` должен быть объектом.")
        if "step_id" not in item or not isinstance(item["step_id"], int):
            raise ValueError(f"Поле `step_reflection[{i}].step_id` обязательно и должно быть int.")
        if "assessment" not in item or not isinstance(item["assessment"], str) or not item["assessment"].strip():
            raise ValueError(f"Поле `step_reflection[{i}].assessment` обязательно и должно быть непустой строкой.")

    risks = obj.get("risks")
    if not isinstance(risks, list) or not all(isinstance(x, str) and x.strip() for x in risks):
        raise ValueError("Поле `risks` обязательно и должно быть массивом непустых строк.")

    final_verdict = obj.get("final_verdict")
    if not isinstance(final_verdict, str) or not final_verdict.strip():
        raise ValueError("Поле `final_verdict` обязательно и должно быть непустой строкой.")

    return obj  # type: ignore[return-value]

