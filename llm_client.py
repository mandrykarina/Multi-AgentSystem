from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import openai as openai_pkg
from debug_utils import debug_print, text_preview
from openai import OpenAI

from config import load_settings


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_sec: float
    model: str
    seed: int | None
    temperature: float
    max_tokens: int
    api_endpoint: str
    api_type: str
    sdk_name: str
    sdk_version: str
    request_model: str
    response_model: str | None
    system_fingerprint: str | None
    response_created: int | None
    response_format: str | None
    finish_reason: str | None


def _extract_chat_text(response) -> str:
    """
    Извлекаем text из ответа `chat.completions.create`.
    """
    try:
        content = response.choices[0].message.content  # type: ignore[attr-defined]
        if isinstance(content, str) and content.strip():
            return content
    except Exception:
        pass

    # На всякий случай: некоторые адаптеры возвращают текст иначе.
    try:
        text = response.choices[0].text  # type: ignore[attr-defined]
        if isinstance(text, str) and text.strip():
            return text
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Не удалось извлечь текст из ответа модели (chat.completions). "
            f"Проверьте версию `openai` SDK. Детали: {e}"
        ) from e

    raise RuntimeError("Ответ модели пришёл, но текст не удалось извлечь.")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    seed: int | None = None,
    debug: bool = False,
) -> LLMResponse:
    """
    Универсальный вызов OpenAI-compatible API (под Yandex AI Studio).

    Возвращает структурированный LLMResponse (текст + usage + latency + параметры).
    """
    settings = load_settings()
    effective_seed = settings.seed if seed is None else seed

    client = OpenAI(
        api_key=settings.yandex_api_key,
        base_url=settings.yandex_base_url,
        project=settings.yandex_folder_id,
        timeout=60.0,
    )

    # Yandex использует модель формата: `gpt://{folder_id}/{model_name}`
    model = f"gpt://{settings.yandex_folder_id}/{settings.yandex_model}"

    # Важно: не используем `client.responses.create`, т.к. у Yandex это ходит в `/v1/responses`,
    # и в вашем случае IAM может не выдавать права на этот эндпоинт.
    # Используем классический `chat/completions` эндпоинт.
    request_kwargs: dict[str, Any] = {}
    if effective_seed is not None:
        # Передаем seed в body запроса, если backend его поддерживает.
        request_kwargs["extra_body"] = {"seed": effective_seed}

    debug_print(
        debug,
        "llm.request",
        payload={
            "api_endpoint": f"{settings.yandex_base_url.rstrip('/')}/chat/completions",
            "request_model": model,
            "seed": effective_seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "sdk_name": "openai",
            "sdk_version": str(getattr(openai_pkg, "__version__", "unknown")),
            "system_prompt_len": len(system_prompt or ""),
            "user_prompt_len": len(user_prompt or ""),
            "user_prompt_preview": text_preview(user_prompt, max_len=120),
        },
    )

    started = perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        **request_kwargs,
    )
    latency_sec = perf_counter() - started

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

    result = LLMResponse(
        text=_extract_chat_text(response),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_sec=latency_sec,
        model=settings.yandex_model,
        seed=effective_seed,
        temperature=temperature,
        max_tokens=max_tokens,
        api_endpoint=f"{settings.yandex_base_url.rstrip('/')}/chat/completions",
        api_type="openai_compatible",
        sdk_name="openai",
        sdk_version=str(getattr(openai_pkg, "__version__", "unknown")),
        request_model=model,
        response_model=getattr(response, "model", None),
        system_fingerprint=getattr(response, "system_fingerprint", None),
        response_created=getattr(response, "created", None),
        response_format="text",
        finish_reason=getattr(response.choices[0], "finish_reason", None) if getattr(response, "choices", None) else None,
    )
    debug_print(
        debug,
        "llm.response",
        payload={
            "response_model": result.response_model,
            "finish_reason": result.finish_reason,
            "response_created": result.response_created,
            "system_fingerprint": result.system_fingerprint,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "latency_sec": result.latency_sec,
            "response_preview": text_preview(result.text, max_len=120),
        },
    )
    return result

