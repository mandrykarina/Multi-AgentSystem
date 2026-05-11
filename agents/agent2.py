from __future__ import annotations

import json
from typing import Any

from llm_client import call_llm
from utils.json_parser import try_parse_json_from_messy_text, validate_agent2_output


SOLVER_SYSTEM_PROMPT = (
    "Ты — Solver / Executor в системе Procedural Meta-Reflection (PMR).\n"
    "Твоя задача: выполнить задачу ПО ПРОЦЕДУРЕ, заданной `plan` от Агент 1.\n"
    "Ты НЕ выбираешь процедуру заново с нуля.\n\n"
    "Требования:\n"
    "1) Следуй шагам из `plan`.\n"
    "2) Для каждого шага верни:\n"
    "   - `action`: что делаем конкретно;\n"
    "   - `comment`: короткий процедурный комментарий (зачем этот шаг);\n"
    "3) Если на практике приходится адаптировать план — перечисли это в `adaptation_points`.\n"
    "4) В конце дай `final_answer`: краткий итог решения.\n\n"
    "Формат ответа: строго ТОЛЬКО JSON-объект.\n"
    "Нельзя добавлять текст вне JSON.\n"
    "Нельзя использовать тройные backticks.\n"
    "Все ключи и строки должны быть в двойных кавычках.\n"
    "Ожидаемая схема:\n"
    '{\n'
    '  "solution_steps": [\n'
    '    {"step": 1, "action": "string", "comment": "string"}\n'
    "  ],\n"
    '  "final_answer": "string",\n'
    '  "adaptation_points": ["string"]\n'
    "}\n"
)


def execute_task(
    task_text: str,
    agent1_result: dict[str, Any],
    *,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """
    Исполняет задачу по плану Агент 1 и возвращает структуру для PMR.
    """
    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError("`task_text` должен быть непустой строкой.")
    if not isinstance(agent1_result, dict):
        raise ValueError("`agent1_result` должен быть словарём (dict).")

    # Достаём, но не делаем строгую валидацию тут: Агент 1 уже валидировался.
    task_type = agent1_result.get("task_type")
    plan = agent1_result.get("plan")
    alternatives = agent1_result.get("alternatives")
    notes = agent1_result.get("notes")

    # Компонуем user prompt через JSON, чтобы модели проще было читать структуру.
    agent1_compact = {
        "task_type": task_type,
        "plan": plan,
        "alternatives": alternatives,
        "notes": notes,
    }

    user_prompt_base = (
        "Выполни задачу по следующей процедуре (PMR / Agent 2).\n\n"
        f"task_text:\n{task_text}\n\n"
        f"agent1_result (используй как основу):\n{json.dumps(agent1_compact, ensure_ascii=False)}\n\n"
        "Сгенерируй ответ строго по JSON-схеме без текста вне JSON."
    )

    last_error: Exception | None = None
    settings_context = (
        f"temperature=используется конфигом, max_output_tokens=используется конфигом. "
        f"Возврат только JSON."
    )

    # Важно: первые попытки дают больше свободы, но все равно просим "только JSON".
    for attempt in range(1, max_attempts + 1):
        user_prompt = user_prompt_base
        if attempt > 1:
            user_prompt += (
                "\n\nВажно (повторная попытка): предыдущий ответ не был распознан как валидный JSON.\n"
                "Верни ТОЛЬКО валидный JSON-объект строго по схеме. Никакого текста вне JSON."
            )

        raw_text = call_llm(
            system_prompt=SOLVER_SYSTEM_PROMPT,
            user_prompt=user_prompt + "\n\n" + settings_context,
            temperature=0.25,
            max_tokens=1500,
        )

        try:
            parsed = try_parse_json_from_messy_text(raw_text)
            validated = validate_agent2_output(parsed)
            return validated
        except Exception as e:
            last_error = e

    raise RuntimeError(
        "Не удалось получить валидный JSON от Агент 2 после "
        f"{max_attempts} попыток. Ошибка: {last_error}"
    )

