# PMR (Procedural Meta-Reflection) — Agent 1 + Agent 2 + Agent 3

Проект работает и как CLI, и как импортируемый API для других скриптов.

## Требования
- Python 3.11+
- Доступ к Yandex AI Studio (OpenAI-compatible API).

## Установка
```bash
python -m pip install -r requirements.txt
```

## Настройка env-переменных
Скопируйте `.env.example` в `.env` и укажите значения.

Важно: в код секреты не зашиваются — используются env-переменные.

Используются:
- `YANDEX_API_KEY`
- `YANDEX_FOLDER_ID`
- `YANDEX_MODEL`

Опционально:
- `YANDEX_BASE_URL` (по умолчанию `https://ai.api.cloud.yandex.net/v1`)
- `TEMPERATURE` (по умолчанию `0.2`)
- `MAX_OUTPUT_TOKENS` (по умолчанию `1500`)

## CLI запуск (Агент 1)
```bash
python main.py "Сформируй процедуру решения задачи по классификации текста на 3 категории."
```

На выходе будет JSON-объект в формате Агент 1:
`task_type`, `plan`, `alternatives`, `notes`.

## Запуск связки Agent 1 -> Agent 2
```bash
python main.py --agent2 "Выполни практическое решение задачи: предложи шаги и действия для классификации текстов на 3 категории."
```

Сначала запускается `Агент 1 — Procedural Analyst`, затем его результат передается в `Агент 2 — Solver/Executor`. На выходе будет структурированный JSON:
`solution_steps`, `final_answer`, `adaptation_points`.

## Запуск Agent 1 -> Agent 2 -> Agent 3
Только результат рефлексии Агент 3:
```bash
python main.py --agent3 "Выполни практическое решение задачи и проведи рефлексию по PMR."
```

Через существующий путь `--agent2`, но с расширением до Агент 3:
```bash
python main.py --agent2 --with-agent3 "Выполни практическое решение задачи и проведи рефлексию по PMR."
```

## Запуск по датасету
Все задачи из датасета по умолчанию (`pmr_dataset_30_with_answers.json`):
```bash
python main.py --dataset --all
```

Все задачи из указанного файла:
```bash
python main.py --dataset "pmr_dataset_30_with_answers.json" --all
```

Выбор задач по `id` (через запятую):
```bash
python main.py --dataset "pmr_dataset_30_with_answers.json" --nums S1,S4,S10
```

Важно: поле `answer` из датасета **не передается в модель**.  
После решения для каждой задачи выводятся:
- `ai_final_answer` (итоговый ответ ИИ),
- `dataset_answer` (эталон из датасета)  
для наглядного сравнения.

## Использование из Python-скрипта
```python
import json
from pmr_api import run_pipeline, run_agent1, run_agent2
from pmr_api import run_agent3, run_pipeline_v3

task_text = "Нужно предложить способ классификации текстов на 3 категории."

# Вариант 1: полный пайплайн Agent1 -> Agent2
pipeline_result = run_pipeline(task_text)
print(json.dumps(pipeline_result, ensure_ascii=False, indent=2))

# Вариант 2: вызывать шаги по отдельности
agent1_result = run_agent1(task_text)
agent2_result = run_agent2(task_text, agent1_result)

# Вариант 3: подключить рефлексию Агент 3
agent3_result = run_agent3(task_text, agent1_result, agent2_result)

# Вариант 4: полный цикл Agent1 -> Agent2 -> Agent3
pipeline_v3_result = run_pipeline_v3(task_text)
```

