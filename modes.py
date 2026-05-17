from __future__ import annotations

import re
from typing import Any

from config import load_settings
from debug_utils import debug_print, text_preview
from llm_client import call_llm


DIRECT_SYSTEM_PROMPT = (
    "Ты решаешь задачу. "
    "Дай краткий, точный и проверяемый ответ."
)

COT_SYSTEM_PROMPT = (
    "Ты решаешь задачу. "
    "Опиши решение пошагово, затем отдельно укажи итоговый ответ."
)

PS_SYSTEM_PROMPT = (
    "Ты решаешь задачу. "
    "Сначала составь план решения задачи, затем выполни решение по этому плану. "
    "В конце отдельно укажи итоговый ответ."
)


def _base_result(response_text: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_response": response_text,
        "parsed_response": meta.get("parsed_response"),
        "prompt_tokens": meta["prompt_tokens"],
        "completion_tokens": meta["completion_tokens"],
        "total_tokens": meta["total_tokens"],
        "latency_sec": meta["latency_sec"],
        "model": meta["model"],
        "seed": meta["seed"],
        "temperature": meta["temperature"],
        "max_tokens": meta["max_tokens"],
        "retry_count": meta["retry_count"],
        "parse_status": meta["parse_status"],
        "api_endpoint": meta.get("api_endpoint"),
        "api_type": meta.get("api_type"),
        "sdk_name": meta.get("sdk_name"),
        "sdk_version": meta.get("sdk_version"),
        "request_model": meta.get("request_model"),
        "response_model": meta.get("response_model"),
        "system_fingerprint": meta.get("system_fingerprint"),
        "response_created": meta.get("response_created"),
        "response_format": meta.get("response_format"),
        "finish_reason": meta.get("finish_reason"),
    }


_FINAL_ANSWER_PATTERNS = [
    re.compile(r"(?:^|\n)\s*итоговый ответ\s*[:\-]\s*(.+)$", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:^|\n)\s*ответ\s*[:\-]\s*(.+)$", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:^|\n)\s*итог\s*[:\-]\s*(.+)$", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:^|\n)\s*финальный ответ\s*[:\-]\s*(.+)$", re.IGNORECASE | re.DOTALL),
]


def extract_final_answer(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    for pattern in _FINAL_ANSWER_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    fallback = lines[-1] if lines else cleaned
    return fallback


def run_direct(task_text: str, *, seed: int | None = None, debug: bool = False) -> dict[str, Any]:
    settings = load_settings()
    response = call_llm(
        system_prompt=DIRECT_SYSTEM_PROMPT,
        user_prompt=task_text,
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        seed=seed,
        debug=debug,
    )

    result = _base_result(
        response.text,
        {
            "parsed_response": None,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "latency_sec": response.latency_sec,
            "model": response.model,
            "seed": response.seed,
            "temperature": response.temperature,
            "max_tokens": response.max_tokens,
            "retry_count": 0,
            "parse_status": "ok",
            "api_endpoint": response.api_endpoint,
            "api_type": response.api_type,
            "sdk_name": response.sdk_name,
            "sdk_version": response.sdk_version,
            "request_model": response.request_model,
            "response_model": response.response_model,
            "system_fingerprint": response.system_fingerprint,
            "response_created": response.response_created,
            "response_format": response.response_format,
            "finish_reason": response.finish_reason,
        },
    )
    result["final_answer"] = response.text.strip()
    return result


def run_cot(task_text: str, *, seed: int | None = None, debug: bool = False) -> dict[str, Any]:
    settings = load_settings()
    response = call_llm(
        system_prompt=COT_SYSTEM_PROMPT,
        user_prompt=task_text,
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        seed=seed,
        debug=debug,
    )

    result = _base_result(
        response.text,
        {
            "parsed_response": None,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "latency_sec": response.latency_sec,
            "model": response.model,
            "seed": response.seed,
            "temperature": response.temperature,
            "max_tokens": response.max_tokens,
            "retry_count": 0,
            "parse_status": "ok",
            "api_endpoint": response.api_endpoint,
            "api_type": response.api_type,
            "sdk_name": response.sdk_name,
            "sdk_version": response.sdk_version,
            "request_model": response.request_model,
            "response_model": response.response_model,
            "system_fingerprint": response.system_fingerprint,
            "response_created": response.response_created,
            "response_format": response.response_format,
            "finish_reason": response.finish_reason,
        },
    )
    result["final_answer"] = extract_final_answer(response.text)
    debug_print(
        debug,
        "final_answer.selected",
        mode="cot",
        payload={"preview": text_preview(result["final_answer"], max_len=120)},
    )
    return result


def run_ps(
    task_text: str,
    *,
    seed: int | None = None,
    max_attempts: int = 2,
    debug: bool = False,
) -> dict[str, Any]:
    settings = load_settings()
    debug_print(
        debug,
        "ps.attempt_start",
        mode="ps",
        seed=seed,
        payload={"attempt": 1, "max_attempts": max_attempts},
    )
    response = call_llm(
        system_prompt=PS_SYSTEM_PROMPT,
        user_prompt=task_text,
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        seed=seed,
        debug=debug,
    )
    result = _base_result(
        response.text,
        {
            "parsed_response": None,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "latency_sec": response.latency_sec,
            "model": response.model,
            "seed": response.seed,
            "temperature": response.temperature,
            "max_tokens": response.max_tokens,
            "retry_count": 0,
            "parse_status": "ok",
            "api_endpoint": response.api_endpoint,
            "api_type": response.api_type,
            "sdk_name": response.sdk_name,
            "sdk_version": response.sdk_version,
            "request_model": response.request_model,
            "response_model": response.response_model,
            "system_fingerprint": response.system_fingerprint,
            "response_created": response.response_created,
            "response_format": response.response_format,
            "finish_reason": response.finish_reason,
        },
    )
    result["final_answer"] = extract_final_answer(response.text)
    debug_print(
        debug,
        "final_answer.selected",
        mode="ps",
        payload={"preview": text_preview(result["final_answer"], max_len=120)},
    )
    return result
