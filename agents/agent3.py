from __future__ import annotations

import json
from typing import Any

from llm_client import call_llm
from utils.json_parser import try_parse_json_from_messy_text, validate_agent3_output


REFLECTIVE_ANALYST_SYSTEM_PROMPT = (
    "Ты — Reflective Analyst в системе Procedural Meta-Reflection (PMR).\n"
    "Твоя задача: оценить качество уже полученного решения, а не решать задачу заново.\n"
    "Ты анализируешь соответствие решения Агент 2 процедуре Агент 1.\n\n"
    "Сделай следующее:\n"
    "1) Дай оценку согласованности решения с процедурой (`alignment_assessment`).\n"
    "2) Для каждого ключевого шага дай рефлексию: что сделано, какая проблема/недочет, как улучшить.\n"
    "3) Перечисли основные риски.\n"
    "4) Сформулируй итоговый вердикт.\n\n"
    "Формат ответа: верни ТОЛЬКО строгий JSON-объект.\n"
    "Никакого текста вне JSON.\n"
    "Нельзя использовать тройные backticks.\n"
    "Все строки и ключи — в двойных кавычках.\n"
    "Схема:\n"
    '{\n'
    '  "alignment_assessment": {"score": 0, "summary": "string"},\n'
    '  "step_reflection": [\n'
    '    {"step": 1, "what_was_done": "string", "issue": "string", "improvement": "string"}\n'
    "  ],\n"
    '  "risks": ["string"],\n'
    '  "final_verdict": "string"\n'
    "}\n"
    "Ограничение: `alignment_assessment.score` должен быть целым числом от 0 до 10."
)


def reflect_solution(
    task_text: str,
    agent1_result: dict[str, Any],
    agent2_result: dict[str, Any],
    max_attempts: int = 2,
) -> dict[str, Any]:
    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError("`task_text` должен быть непустой строкой.")
    if not isinstance(agent1_result, dict):
        raise ValueError("`agent1_result` должен быть словарём (dict).")
    if not isinstance(agent2_result, dict):
        raise ValueError("`agent2_result` должен быть словарём (dict).")

    user_payload = {
        "task_text": task_text,
        "agent1_result": agent1_result,
        "agent2_result": agent2_result,
    }

    user_prompt_base = (
        "Проведи рефлексию решения в роли Agent 3 (Reflective Analyst).\n"
        "Оцени, насколько решение Agent 2 следует процедуре Agent 1.\n\n"
        f"Входные данные:\n{json.dumps(user_payload, ensure_ascii=False)}\n\n"
        "Верни строго JSON по схеме."
    )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        user_prompt = user_prompt_base
        if attempt > 1:
            user_prompt += (
                "\n\nВажно: предыдущий ответ не прошел валидацию JSON.\n"
                "Повтори ответ: только валидный JSON-объект по схеме, без текста вне JSON."
            )

        raw_text = call_llm(
            system_prompt=REFLECTIVE_ANALYST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=1500,
        )
        try:
            parsed = try_parse_json_from_messy_text(raw_text)
            return validate_agent3_output(parsed)
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Не удалось получить валидный JSON от Агент 3 после {max_attempts} попыток. Ошибка: {last_error}"
    )

