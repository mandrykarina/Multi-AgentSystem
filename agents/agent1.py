from __future__ import annotations

from typing import Any

from config import load_settings
from debug_utils import debug_print
from llm_client import call_llm
from utils.json_parser import try_parse_json_from_messy_text, validate_agent1_output


PROCEDURAL_ANALYST_SYSTEM_PROMPT = (
    "Ты — аналитик процедур в рамках процедурной мета-рефлексии. "
    "Твоя задача: проанализировать текст задачи до её решения и предложить процедуру решения. "
    "Ты не решаешь задачу полностью и не выполняешь вычисления.\n\n"
    "Что нужно сделать:\n"
    "1) определить тип задачи (task_type);\n"
    "2) предложить plan — список шагов процедуры;\n"
    "3) перечислить alternatives — альтернативные методы и кратко объяснить rejection_reason "
    "(почему они не выбраны);\n"
    "4) дать notes — массив коротких замечаний, ограничений и проверок.\n\n"
    "Формат ответа: верни только строгий JSON-объект. "
    "Нельзя добавлять никакой текст вне JSON. "
    "Нельзя использовать тройные обратные кавычки. "
    "Все ключи и строки должны быть заключены в двойные кавычки.\n"
    "Ожидаемая схема:\n"
    '{\n'
    '  "task_type": "string",\n'
    '  "plan": ["шаг 1", "шаг 2"],\n'
    '  "alternatives": [\n'
    '    { "method": "string", "rejection_reason": "string" }\n'
    "  ],\n"
    '  "notes": ["string"]\n'
    "}\n"
)


def analyze_task(
    task_text: str,
    *,
    max_attempts: int = 2,
    seed: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
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
        debug_print(
            debug,
            "agent1.attempt_start",
            mode="pmr",
            seed=seed,
            payload={"attempt": attempt, "max_attempts": max_attempts},
        )
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
            seed=seed,
            debug=debug,
        )

        try:
            parsed = try_parse_json_from_messy_text(raw_text.text)
            validated = validate_agent1_output(parsed)
            validated["_meta"] = {
                "raw_response": raw_text.text,
                "prompt_tokens": raw_text.prompt_tokens,
                "completion_tokens": raw_text.completion_tokens,
                "total_tokens": raw_text.total_tokens,
                "latency_sec": raw_text.latency_sec,
                "model": raw_text.model,
                "seed": raw_text.seed,
                "temperature": raw_text.temperature,
                "max_tokens": raw_text.max_tokens,
                "retry_count": attempt - 1,
                "parse_status": "ok" if attempt == 1 else "retry_success",
                "api_endpoint": raw_text.api_endpoint,
                "api_type": raw_text.api_type,
                "sdk_name": raw_text.sdk_name,
                "sdk_version": raw_text.sdk_version,
                "request_model": raw_text.request_model,
                "response_model": raw_text.response_model,
                "system_fingerprint": raw_text.system_fingerprint,
                "response_created": raw_text.response_created,
                "response_format": raw_text.response_format,
                "finish_reason": raw_text.finish_reason,
            }
            debug_print(
                debug,
                "agent1.attempt_success",
                mode="pmr",
                seed=seed,
                payload={"attempt": attempt, "parse_status": validated["_meta"]["parse_status"]},
            )
            return validated
        except Exception as e:
            last_error = e
            debug_print(
                debug,
                "agent1.attempt_failed",
                mode="pmr",
                seed=seed,
                payload={"attempt": attempt, "exception_type": type(e).__name__, "message": str(e)},
            )

    raise RuntimeError(f"Не удалось получить валидный JSON от Агент 1 после {max_attempts} попыток. Ошибка: {last_error}")

