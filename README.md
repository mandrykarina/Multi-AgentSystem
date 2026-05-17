# PMR (Procedural Meta-Reflection): сравнение 4 режимов генерации на одной LLM

Этот проект реализует единый воспроизводимый контур экспериментов для сравнения четырех подходов решения задач на одной и той же модели:

- `direct` — прямой вызов (базовый режим);
- `cot` — chain-of-thought (пошаговое рассуждение);
- `ps` — plan-and-solve (текстовый baseline: сначала план, затем решение);
- `pmr` — Procedural Meta-Reflection (трехагентный конвейер: анализ процедуры -> исполнение -> рефлексия).

Основная цель проекта — не просто получить итоговый ответ, а измерить:

- качество ответа;
- процедурную прозрачность;
- воспроизводимость при повторных запусках с разными `seed`;
- инженерную стоимость (токены, задержка).

---

## 1) Что делает проект

Проект предоставляет:

1. **API-слой** для запуска каждого режима и полного пайплайна.
2. **CLI** для одиночных запусков и массового прогона по датасету.
3. **Единое JSONL-логирование** результатов и метаданных API.
4. **Агрегацию метрик** (`mean ± std`) по каждому режиму.
5. **Повторяемость эксперимента** через явные `--runs` и `--seeds`.

---

## 2) Архитектура

### 2.1 PMR-конвейер (трехагентный)

- **Agent 1 (Procedural Analyst)**: определяет тип задачи, план, альтернативы и ограничения.
- **Agent 2 (Executor)**: решает задачу строго по плану Agent 1.
- **Agent 3 (Reflective Analyst)**: оценивает согласованность решения с планом и выделяет риски.

### 2.2 Базовые режимы

- **Direct**: минимальная инструкция, без явной процедуры.
- **CoT**: пошаговое рассуждение в одном вызове.
- **PS**: план + решение в одном вызове (свободный текстовый baseline).

### 2.3 Поток выполнения в датасетном режиме

1. Чтение задач датасета.
2. Для каждой задачи — запуск каждого режима на каждом `seed`.
3. Сохранение каждого прогона в `experiment.log.jsonl`.
4. Расчет агрегатов и печать итоговой сводки.

---

## 3) Структура проекта

Ключевые файлы:

- `main.py` — CLI, датасетные запуски, агрегация, логирование.
- `pmr_api.py` — публичный API режимов и PMR-пайплайна.
- `modes.py` — реализации `run_direct`, `run_cot`, `run_ps`.
- `llm_client.py` — вызов OpenAI-compatible API и извлечение метаданных ответа.
- `config.py` — загрузка настроек из переменных окружения.
- `experiment_log.py` — запись JSONL-лога.
- `agents/agent1.py`, `agents/agent2.py`, `agents/agent3.py` — PMR-агенты.
- `utils/json_parser.py` — разбор/валидация JSON-ответов.
- `APPENDIX_A.md` — полные системные промпты агентов.

---

## 4) Требования

- Python 3.11+
- Доступ к Yandex AI Studio (OpenAI-compatible endpoint)

Установка зависимостей:

```bash
python -m pip install -r requirements.txt
```

---

## 5) Настройка окружения

Скопируйте `.env.example` в `.env` и заполните значения.

### Обязательные переменные

- `YANDEX_API_KEY`
- `YANDEX_FOLDER_ID`
- `YANDEX_MODEL` (например, `yandexgpt-5-pro`)

### Опциональные переменные

- `YANDEX_BASE_URL` (default: `https://ai.api.cloud.yandex.net/v1`)
- `TEMPERATURE` (default: `0.2`)
- `MAX_OUTPUT_TOKENS` (default: `1500`)
- `SEED` (default: не задан; переопределяется через CLI)
- `RUNS` (ориентир для эксперимента, обычно `3`)

---

## 6) Быстрый старт

### 6.1 Одиночные запуски

Agent 1:

```bash
python main.py "Сформируй процедуру решения задачи по классификации текста."
```

Agent 2 (через Agent1 -> Agent2):

```bash
python main.py --agent2 "Реши задачу и покажи шаги."
```

Agent 3 (через Agent1 -> Agent2 -> Agent3):

```bash
python main.py --agent3 "Реши задачу и проведи рефлексию."
```

---

## 7) Датасетный эксперимент

### 7.1 Стандартная команда воспроизведения

```bash
python main.py --dataset --all --runs 3 --seeds 42,43,44
```

Windows-скрипт с пост-проверками:

```bat
scripts\run_full_experiment.bat
```

### 7.2 Другие варианты

Все задачи из указанного файла:

```bash
python main.py --dataset "pmr_dataset_30_with_answers.json" --all --runs 3 --seeds 42,43,44
```

Выбор задач по `id`:

```bash
python main.py --dataset "pmr_dataset_30_with_answers.json" --nums S1,S4,S10 --runs 3 --seeds 42,43,44
```

По умолчанию CLI пытается найти:

1. `pmr_dataset_100_with_answers.json`
2. если отсутствует — `pmr_dataset_30_with_answers.json`

---

## 8) Логирование (JSONL)

Файл: `experiment.log.jsonl`  
Формат: **одна строка = один JSON-объект одного прогона**.

### 8.1 Основные поля

- `id`, `difficulty`, `domain`, `task`
- `system_type` (`direct` / `cot` / `ps` / `pmr`)
- `raw_response`, `parsed_response`, `final_answer`, `reference_answer`
- `prompt_tokens`, `completion_tokens`, `total_tokens`, `latency_sec`
- `retry_count`, `parse_status`, `timestamp`

### 8.2 API/SDK метаданные

- `api_endpoint`
- `api_type`
- `sdk_name`
- `sdk_version`
- `request_model`
- `response_model`
- `system_fingerprint`
- `response_created`
- `response_format`
- `finish_reason`

### 8.3 Дополнительно для PMR

- `agent1_result`
- `agent2_result`
- `agent3_result`

### 8.4 Ошибочные прогоны

При ошибках режимов создаются записи с:

- `parse_status = "failed"`
- полем `error`
- нулевыми/`null` значениями вычислительных метрик и API-полей.

---

## 9) Агрегируемые метрики

После прогона по датасету проект выводит по каждому режиму:

- `count`
- `rouge_l` (`mean`, `std`)
- `total_tokens` (`mean`, `std`)
- `latency_sec` (`mean`, `std`)
- `reproducibility` (итог по режиму)
- `total_tokens_normalized_to_direct`
- `valid_json_rate` (только для `pmr`, со знаменателем по всем PMR-запускам, включая `failed`)

Дополнительно считается воспроизводимость задачи как бинарный признак (`1/0`) по правилу статьи: минимальный попарный ROUGE-L между seed-запусками должен быть `>= 0.99`. Итоговая воспроизводимость режима — среднее по задачам.

После датасетного прогона артефакты разделяются так:

- `results/` — только финальные артефакты публикационного прогона (`--dataset --all --runs 3 --seeds 42,43,44`);
- `results_smoke/` — короткие технические/smoke прогоны.

В финальном `results/` сохраняются:

- `summary.json`
- `table4_results.csv`
- `table4_full.csv`
- `expert_report.json` (если доступен `results/expert_scores.csv`)
- `stats_report.json`
- `experiment.log.jsonl`

---

## 10) Скрипты запуска

- `run.bat`  
  - без аргументов запускает стандартный эксперимент воспроизведения;
  - с аргументами прокидывает их в `main.py`.

---

## 11) Debug-режим

Для детальной трассировки используйте флаг `--debug`.

Примеры:

```bash
python main.py --dataset --all --runs 1 --seeds 42 --debug
python main.py --agent3 "Реши задачу и проведи рефлексию." --debug
```

Что появляется в консоли:

- структурированные сообщения с префиксом `[DEBUG]`;
- события этапов выполнения (`cli.args_parsed`, `dataset.task_start`, `mode.run_start`, `llm.request`, `llm.response`, `mode.run_success`, `mode.run_failed`, `dataset.task_done`);
- краткие метрики (`latency_sec`, `total_tokens`, `finish_reason`, `parse_status`) и детали ошибок.

Важно:

- debug-вывод не меняет схему `experiment.log.jsonl`;
- секреты (`YANDEX_API_KEY`) и полный текст prompt/response не печатаются.

---

## 12) Использование из Python-кода

Пример:

```python
import json
from pmr_api import run_all_modes, run_pipeline_v3

task = "Выбери метод поиска кратчайшего пути и обоснуй выбор."

# Все режимы на одном seed
all_modes = run_all_modes(task, seed=42)
print(json.dumps(all_modes, ensure_ascii=False, indent=2))

# Полный PMR-пайплайн (Agent1 -> Agent2 -> Agent3)
pmr = run_pipeline_v3(task, seed=42)
print(json.dumps(pmr, ensure_ascii=False, indent=2))
```

---

## 13) Где смотреть промпты

Полные системные промпты агентов вынесены в:

- `APPENDIX_A.md`

Это упрощает синхронизацию кода, документации и текста статьи.

---

## 14) Экспертные оценки и статистические тесты

Проект поддерживает постобработку экспертных оценок и статистическую проверку:

- вход: `results/expert_scores.csv` (финал) или `results_smoke/expert_scores.csv` (smoke)
- экспертные метрики:
  - `content_correctness` (0-2)
  - `choice_justification` (1-5)
  - `alternative_depth` (1-5)
- агрегирование: медиана по 3 экспертам для каждой пары `(id, mode)`
- согласованность: Fleiss kappa + средняя попарная weighted kappa
- статистика: двухвыборочный `t-test` и `Mann-Whitney U` (PMR vs baseline-режимы)

Ожидаемые колонки `expert_scores.csv`:

```csv
id,mode,expert_id,content_correctness,choice_justification,alternative_depth
S1,direct,E1,1,2,1
S1,direct,E2,1,3,1
S1,direct,E3,2,2,1
```

---

## 15) Таймаут API

В `llm_client.py` установлен timeout `60s` для запросов к OpenAI-compatible API Yandex.

---

## 16) Лицензия

Проект распространяется под лицензией MIT, см. файл `LICENSE`.

---

## 17) Проверка финального прогона

После полного запуска проверьте финальный summary:

```bash
python scripts/validate_final_summary.py
```

Проверка датасета:

```bash
python scripts/check_dataset_distribution.py
```

Пересборка `table4_full.csv` из `summary.json` и `expert_report.json`:

```bash
python scripts/build_table4_full.py results
python scripts/build_table4_full.py results_smoke
```

