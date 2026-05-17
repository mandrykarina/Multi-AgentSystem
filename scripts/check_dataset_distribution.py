from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> int:
    dataset_path = Path("pmr_dataset_100_with_answers.json")
    if not dataset_path.exists():
        print(f"dataset_not_found={dataset_path.as_posix()}")
        return 1

    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        print("dataset_format_error=not_array")
        return 1

    difficulty_counter = Counter(str(row.get("difficulty", "")) for row in rows if isinstance(row, dict))
    domain_counter = Counter(str(row.get("domain", "")) for row in rows if isinstance(row, dict))

    print(f"total={len(rows)}")
    print(f"simple={difficulty_counter.get('simple', 0)}")
    print(f"medium={difficulty_counter.get('medium', 0)}")
    print(f"hard={difficulty_counter.get('hard', 0)}")
    print("domains=" + ",".join(sorted(d for d in domain_counter if d)))

    ok = (
        len(rows) == 100
        and difficulty_counter.get("simple", 0) == 30
        and difficulty_counter.get("medium", 0) == 40
        and difficulty_counter.get("hard", 0) == 30
    )
    print(f"distribution_ok={str(ok).lower()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
