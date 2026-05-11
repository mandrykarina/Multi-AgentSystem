from __future__ import annotations

from openai import OpenAI

from config import load_settings


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
) -> str:
    """
    Универсальный вызов OpenAI-compatible API (под Yandex AI Studio).

    Возвращает только текст ответа модели.
    """
    settings = load_settings()

    client = OpenAI(
        api_key=settings.yandex_api_key,
        base_url=settings.yandex_base_url,
        project=settings.yandex_folder_id,
    )

    # Yandex использует модель формата: `gpt://{folder_id}/{model_name}`
    model = f"gpt://{settings.yandex_folder_id}/{settings.yandex_model}"

    # Важно: не используем `client.responses.create`, т.к. у Yandex это ходит в `/v1/responses`,
    # и в вашем случае IAM может не выдавать права на этот эндпоинт.
    # Используем классический `chat/completions` эндпоинт.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return _extract_chat_text(response)

