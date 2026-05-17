# Spec: Article Alignment Contract

Этот файл фиксирует целевую спецификацию проекта для строгого соответствия статье (разделы 3-6).
Если поведение кода расходится с этим документом, приоритет у спецификации.

## 1) Prompt templates (дословные базовые формулировки)

- `direct`: `Ты решаешь задачу. Дай краткий, точный и проверяемый ответ.`
- `cot`: `Ты решаешь задачу. Опиши решение пошагово, затем отдельно укажи итоговый ответ.`
- `ps`: `Ты решаешь задачу. Сначала составь план решения задачи, затем выполни решение по этому плану. В конце отдельно укажи итоговый ответ.`

PMR остаётся JSON-ориентированным конвейером из 3 агентов.

## 2) JSON schema contracts for PMR agents

### Agent 1

```json
{
  "task_type": "string",
  "plan": ["string"],
  "alternatives": [
    {
      "method": "string",
      "rejection_reason": "string"
    }
  ],
  "notes": ["string"]
}
```

### Agent 2

```json
{
  "solution_steps": [
    {
      "step_id": 1,
      "action": "string",
      "plan_alignment": "string",
      "result": "string"
    }
  ],
  "final_answer": "string",
  "adaptation_points": ["string"]
}
```

### Agent 3

```json
{
  "alignment_assessment": {
    "score": 9,
    "explanation": "string"
  },
  "step_reflection": [
    {
      "step_id": 1,
      "assessment": "string"
    }
  ],
  "risks": ["string"],
  "final_verdict": "string"
}
```

## 3) Logging contract (per run)

Обязательные поля:

- `raw_response`, `parsed_response`, `final_answer`, `reference_answer`
- `prompt_tokens`, `completion_tokens`, `total_tokens`, `latency_sec`
- `api_endpoint`, `api_type`, `sdk_name`, `sdk_version`
- `request_model`, `response_model`, `system_fingerprint`
- `response_created`, `response_format`, `finish_reason`

## 4) Metrics contract

- Автоматические: `ROUGE-L`, `total_tokens`, `latency_sec`.
- Воспроизводимость режима: доля задач, где для всех seed попарный `ROUGE-L >= 0.99`.
- Нормализованные токены: относительно `direct`.
- `valid_json_rate`: только для `pmr`, знаменатель включает все PMR-запуски (включая failed).

## 4.1) Results artifacts contract

- Финальный (публикационный) прогон пишет в `results/`.
- Smoke/debug прогоны пишут в `results_smoke/`.
- Обязательные финальные файлы:
  - `summary.json`
  - `table4_results.csv`
  - `table4_full.csv`
  - `expert_report.json`
  - `stats_report.json`
  - `experiment.log.jsonl`

## 5) Expert and statistics contract

- Экспертные шкалы:
  - `content_correctness`: 0-2
  - `choice_justification`: 1-5
  - `alternative_depth`: 1-5
- 3 независимых эксперта, итог по задаче/режиму: медиана.
- Межэкспертная согласованность:
  - Fleiss kappa (unweighted)
  - средняя попарная weighted kappa (линейные веса).
- Статистика по режимам: двухвыборочный t-test и Mann-Whitney U.

## 6) Runtime/environment contract

- Референс окружение: Python `3.11`.
- Таймаут API-запроса: `60` секунд.
- Лицензия репозитория: MIT (`LICENSE`).
- Команда полного воспроизведения:
  `python main.py --dataset --all --runs 3 --seeds 42,43,44`
