from __future__ import annotations

import os
from dataclasses import dataclass

try:
    # Удобство разработки: автоматически подгружаем `.env` если он существует.
    # Секреты всё равно не коммитим в репозиторий.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Если зависимости нет или `.env` отсутствует — работаем с env-переменными.
    pass


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Не найдена переменная окружения `{name}`. "
            "Укажите её перед запуском (см. `.env.example`)."
        )
    return value


@dataclass(frozen=True)
class Settings:
    # Yandex AI Studio (OpenAI-compatible)
    yandex_api_key: str
    yandex_folder_id: str
    yandex_model: str
    yandex_base_url: str

    # Generation params
    temperature: float
    max_output_tokens: int


def load_settings() -> Settings:
    # Note: base_url по умолчанию для OpenAI-compatible сервиса Yandex.
    yandex_base_url = os.getenv(
        "YANDEX_BASE_URL",
        "https://ai.api.cloud.yandex.net/v1",
    )

    temperature = float(os.getenv("TEMPERATURE", "0.2"))
    max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "1500"))

    return Settings(
        yandex_api_key=_require_env("YANDEX_API_KEY"),
        yandex_folder_id=_require_env("YANDEX_FOLDER_ID"),
        yandex_model=_require_env("YANDEX_MODEL"),
        yandex_base_url=yandex_base_url,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

