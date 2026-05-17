from __future__ import annotations

import json
from typing import Any

from config import load_settings
from debug_utils import debug_print
from llm_client import call_llm
from utils.json_parser import try_parse_json_from_messy_text, validate_agent2_output


SOLVER_SYSTEM_PROMPT = (
    "Ты — исполнитель в системе процедурной мета-рефлексии. "
    "Твоя задача: выполнить задачу строго по процедуре, заданной plan от первого агента. "
    "Ты не выбираешь процедуру заново.\n\n"
    "Требования:\n"
    "1) Следуй шагам из plan.\n"
    "2) Для каждого шага верни action (что делаем), plan_alignment "
    "(почему шаг соответствует плану Agent 1) и result (результат шага).\n"
    "3) Если на практике приходится адаптировать план — перечисли это в adaptation_points.\n"
    "4) В конце дай final_answer — краткий итог решения.\n\n"
    "Формат ответа: строго только JSON-объект. Нельзя добавлять текст вне JSON.\n"
    "Ожидаемая схема:\n"
    '{\n'
    '  "solution_steps": [\n'
    '    {"step_id": 1, "action": "string", "plan_alignment": "string", "result": "string"}\n'
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
    seed: int | None = None,
    debug: bool = False,
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

    settings = load_settings()
    user_prompt_base = (
        "Выполни задачу по следующей процедуре (PMR / Agent 2).\n\n"
        f"task_text:\n{task_text}\n\n"
        f"agent1_result (используй как основу):\n{json.dumps(agent1_compact, ensure_ascii=False)}\n\n"
        "Сгенерируй ответ строго по JSON-схеме без текста вне JSON."
    )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        debug_print(
            debug,
            "agent2.attempt_start",
            mode="pmr",
            seed=seed,
            payload={"attempt": attempt, "max_attempts": max_attempts},
        )
        user_prompt = user_prompt_base
        if attempt > 1:
            user_prompt += (
                "\n\nВажно (повторная попытка): предыдущий ответ не был распознан как валидный JSON.\n"
                "Верни ТОЛЬКО валидный JSON-объект строго по схеме. Никакого текста вне JSON."
            )

        llm_response = call_llm(
            system_prompt=SOLVER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
            seed=seed,
            debug=debug,
        )

        try:
            parsed = try_parse_json_from_messy_text(llm_response.text)
            validated = validate_agent2_output(parsed)
            validated["_meta"] = {
                "raw_response": llm_response.text,
                "prompt_tokens": llm_response.prompt_tokens,
                "completion_tokens": llm_response.completion_tokens,
                "total_tokens": llm_response.total_tokens,
                "latency_sec": llm_response.latency_sec,
                "model": llm_response.model,
                "seed": llm_response.seed,
                "temperature": llm_response.temperature,
                "max_tokens": llm_response.max_tokens,
                "retry_count": attempt - 1,
                "parse_status": "ok" if attempt == 1 else "retry_success",
                "api_endpoint": llm_response.api_endpoint,
                "api_type": llm_response.api_type,
                "sdk_name": llm_response.sdk_name,
                "sdk_version": llm_response.sdk_version,
                "request_model": llm_response.request_model,
                "response_model": llm_response.response_model,
                "system_fingerprint": llm_response.system_fingerprint,
                "response_created": llm_response.response_created,
                "response_format": llm_response.response_format,
                "finish_reason": llm_response.finish_reason,
            }
            debug_print(
                debug,
                "agent2.attempt_success",
                mode="pmr",
                seed=seed,
                payload={"attempt": attempt, "parse_status": validated["_meta"]["parse_status"]},
            )
            return validated
        except Exception as e:
            last_error = e
            debug_print(
                debug,
                "agent2.attempt_failed",
                mode="pmr",
                seed=seed,
                payload={"attempt": attempt, "exception_type": type(e).__name__, "message": str(e)},
            )

    raise RuntimeError(
        "Не удалось получить валидный JSON от Агент 2 после "
        f"{max_attempts} попыток. Ошибка: {last_error}"
    )

