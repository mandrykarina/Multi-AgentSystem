from __future__ import annotations

import json
from typing import Any

from config import load_settings
from debug_utils import debug_print
from llm_client import call_llm
from utils.json_parser import try_parse_json_from_messy_text, validate_agent3_output


REFLECTIVE_ANALYST_SYSTEM_PROMPT = (
    "Ты — рефлексивный аналитик в системе процедурной мета-рефлексии. "
    "Твоя задача: оценить качество уже полученного решения, а не решать задачу заново. "
    "Ты анализируешь соответствие решения второго агента процедуре первого.\n\n"
    "Что нужно сделать:\n"
    "1) Дай оценку согласованности решения с процедурой (alignment_assessment).\n"
    "2) Для каждого ключевого шага дай краткую assessment-оценку, насколько шаг корректен и связан с планом.\n"
    "3) Перечисли основные риски.\n"
    "4) Сформулируй итоговый вердикт.\n\n"
    "Формат ответа: верни только строгий JSON-объект.\n"
    "Схема:\n"
    '{\n'
    '  "alignment_assessment": {"score": 0, "explanation": "string"},\n'
    '  "step_reflection": [\n'
    '    {"step_id": 1, "assessment": "string"}\n'
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
    seed: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError("`task_text` должен быть непустой строкой.")
    if not isinstance(agent1_result, dict):
        raise ValueError("`agent1_result` должен быть словарём (dict).")
    if not isinstance(agent2_result, dict):
        raise ValueError("`agent2_result` должен быть словарём (dict).")

    settings = load_settings()
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
        debug_print(
            debug,
            "agent3.attempt_start",
            mode="pmr",
            seed=seed,
            payload={"attempt": attempt, "max_attempts": max_attempts},
        )
        user_prompt = user_prompt_base
        if attempt > 1:
            user_prompt += (
                "\n\nВажно: предыдущий ответ не прошел валидацию JSON.\n"
                "Повтори ответ: только валидный JSON-объект по схеме, без текста вне JSON."
            )

        llm_response = call_llm(
            system_prompt=REFLECTIVE_ANALYST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
            seed=seed,
            debug=debug,
        )
        try:
            parsed = try_parse_json_from_messy_text(llm_response.text)
            validated = validate_agent3_output(parsed)
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
                "agent3.attempt_success",
                mode="pmr",
                seed=seed,
                payload={"attempt": attempt, "parse_status": validated["_meta"]["parse_status"]},
            )
            return validated
        except Exception as e:
            last_error = e
            debug_print(
                debug,
                "agent3.attempt_failed",
                mode="pmr",
                seed=seed,
                payload={"attempt": attempt, "exception_type": type(e).__name__, "message": str(e)},
            )

    raise RuntimeError(
        f"Не удалось получить валидный JSON от Агент 3 после {max_attempts} попыток. Ошибка: {last_error}"
    )

