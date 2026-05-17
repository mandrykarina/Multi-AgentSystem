from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_preview(value: str, *, max_len: int = 160) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[:max_len] + "...(+truncated)"


def debug_print(
    enabled: bool,
    event: str,
    *,
    mode: str | None = None,
    task_id: str | None = None,
    seed: int | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    if not enabled:
        return
    message = {
        "ts": _iso_now(),
        "event": event,
        "mode": mode,
        "task_id": task_id,
        "seed": seed,
        "payload": payload or {},
    }
    print(f"[DEBUG] {json.dumps(message, ensure_ascii=False)}")
