from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_LOG_PATH = Path("experiment.log.jsonl")


def save_log_entry(entry: dict[str, Any], *, log_path: Path | str = DEFAULT_LOG_PATH) -> None:
    path = Path(log_path)
    line = json.dumps(entry, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
