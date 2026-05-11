from __future__ import annotations

from typing import Any

from config import load_settings
from llm_client import call_llm
from utils.json_parser import try_parse_json_from_messy_text, validate_agent1_output


PROCEDURAL_ANALYST_SYSTEM_PROMPT = (
    "Ты — Procedural Analyst в рамках PMR (Procedural Meta-Reflection).\n"
    "Твоя задача: проанализировать текст задачи ДО решения и предложить процедуру решения.\n"
    "Ты НЕ решаешь задачу полностью и НЕ выполняешь вычисления.\n\n"
    "Что нужно сделать:\n"
    "1) определить `task_type`;\n"
    "2) предложить `plan` — список шагов процедуры;\n"
    "3) перечислить `alternatives` — альтернативные методы/подходы и кратко объяснить `why_rejected` "
    "(почему они не выбраны);\n"
    "4) дать `notes` — короткие замечания/ограничения/что проверить.\n\n"
    "Формат ответа: верни ТОЛЬКО строгий JSON-объект.\n"
    "Нельзя добавлять никакой текст вне JSON.\n"
    "Нельзя использовать тройные backticks.\n"
    "Все ключи и строки должны быть заключены в двойные кавычки.\n"
    "Ожидаемая схема:\n"
    '{\n'
    '  "task_type": "string",\n'
    '  "plan": ["шаг 1", "шаг 2"],\n'
    '  "alternatives": [\n'
    '    { "method": "string", "why_rejected": "string" }\n'
    "  ],\n"
    '  "notes": "string"\n'
    "}\n"
)


def analyze_task(task_text: str, *, max_attempts: int = 2) -> dict[str, Any]:
    """
    Анализирует задачу и возвращает структуру для PMR (только Agent 1).
    """
    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError("`task_text` должен быть непустой строкой.")

    settings = load_settings()
    temperature = settings.temperature
    max_tokens = settings.max_output_tokens

    user_prompt_base = (
        "Проанализируй следующую задачу и верни JSON строго по схеме.\n\n"
        f"Задача:\n{task_text}\n"
    )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        user_prompt = user_prompt_base
        if attempt > 1:
            user_prompt += (
                "\n\nВажно: предыдущий ответ не удалось корректно прочитать как JSON.\n"
                "Сделай повторную попытку: верни ТОЛЬКО валидный JSON-объект, без любого текста вне JSON."
            )

        raw_text = call_llm(
            system_prompt=PROCEDURAL_ANALYST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            parsed = try_parse_json_from_messy_text(raw_text)
            validated = validate_agent1_output(parsed)
            return validated
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Не удалось получить валидный JSON от Агент 1 после {max_attempts} попыток. Ошибка: {last_error}")

