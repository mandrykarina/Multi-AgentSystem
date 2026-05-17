from __future__ import annotations

from typing import Any

from agents.agent1 import analyze_task
from agents.agent2 import execute_task
from agents.agent3 import reflect_solution
from debug_utils import debug_print
from modes import run_cot, run_direct, run_ps


def run_agent1(task_text: str, *, seed: int | None = None, debug: bool = False) -> dict[str, Any]:
    """
    Публичный API-вход для запуска Агент 1.
    """
    return analyze_task(task_text, seed=seed, debug=debug)


def run_agent2(
    task_text: str,
    agent1_result: dict[str, Any],
    *,
    seed: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Публичный API-вход для запуска Агент 2.
    """
    return execute_task(task_text, agent1_result, seed=seed, debug=debug)


def run_pipeline(task_text: str, *, seed: int | None = None, debug: bool = False) -> dict[str, Any]:
    """
    Публичный API пайплайна Agent1 -> Agent2.
    """
    agent1_result = run_agent1(task_text, seed=seed, debug=debug)
    agent2_result = run_agent2(task_text, agent1_result, seed=seed, debug=debug)
    return {
        "task_text": task_text,
        "agent1_result": agent1_result,
        "agent2_result": agent2_result,
    }


def run_agent3(
    task_text: str,
    agent1_result: dict[str, Any],
    agent2_result: dict[str, Any],
    *,
    seed: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Публичный API-вход для запуска Агент 3.
    """
    return reflect_solution(task_text, agent1_result, agent2_result, seed=seed, debug=debug)


def run_pipeline_v3(task_text: str, *, seed: int | None = None, debug: bool = False) -> dict[str, Any]:
    """
    Публичный API пайплайна Agent1 -> Agent2 -> Agent3.
    """
    agent1_result = run_agent1(task_text, seed=seed, debug=debug)
    agent2_result = run_agent2(task_text, agent1_result, seed=seed, debug=debug)
    agent3_result = run_agent3(task_text, agent1_result, agent2_result, seed=seed, debug=debug)
    return {
        "task_text": task_text,
        "agent1_result": agent1_result,
        "agent2_result": agent2_result,
        "agent3_result": agent3_result,
    }


def run_pmr(task_text: str, *, seed: int | None = None, debug: bool = False) -> dict[str, Any]:
    pipeline = run_pipeline_v3(task_text, seed=seed, debug=debug)
    agent1_result = pipeline["agent1_result"]
    agent2_result = pipeline["agent2_result"]
    agent3_result = pipeline["agent3_result"]

    agent2_meta = agent2_result.get("_meta", {}) if isinstance(agent2_result, dict) else {}
    retry_count = 0
    parse_status = "ok"
    prompt_tokens_sum = 0
    completion_tokens_sum = 0
    total_tokens_sum = 0
    latency_sec_sum = 0.0
    model_value = ""
    temperature_value: Any = None
    max_tokens_value: Any = None
    seed_value: Any = None
    api_endpoint_value: Any = None
    api_type_value: Any = None
    sdk_name_value: Any = None
    sdk_version_value: Any = None
    request_model_value: Any = None
    response_model_value: Any = None
    system_fingerprint_value: Any = None
    response_created_value: Any = None
    response_format_value: Any = None
    finish_reason_value: Any = None

    for item in (agent1_result, agent2_result, agent3_result):
        if isinstance(item, dict):
            meta = item.get("_meta", {})
            if isinstance(meta, dict):
                retry_count += int(meta.get("retry_count", 0) or 0)
                prompt_tokens_sum += int(meta.get("prompt_tokens", 0) or 0)
                completion_tokens_sum += int(meta.get("completion_tokens", 0) or 0)
                total_tokens_sum += int(meta.get("total_tokens", 0) or 0)
                latency_sec_sum += float(meta.get("latency_sec", 0.0) or 0.0)
                if not model_value:
                    model_value = str(meta.get("model", "") or "")
                if temperature_value is None:
                    temperature_value = meta.get("temperature")
                if max_tokens_value is None:
                    max_tokens_value = meta.get("max_tokens")
                if seed_value is None:
                    seed_value = meta.get("seed")
                if api_endpoint_value is None:
                    api_endpoint_value = meta.get("api_endpoint")
                if api_type_value is None:
                    api_type_value = meta.get("api_type")
                if sdk_name_value is None:
                    sdk_name_value = meta.get("sdk_name")
                if sdk_version_value is None:
                    sdk_version_value = meta.get("sdk_version")
                if request_model_value is None:
                    request_model_value = meta.get("request_model")
                if response_model_value is None:
                    response_model_value = meta.get("response_model")
                if system_fingerprint_value is None:
                    system_fingerprint_value = meta.get("system_fingerprint")
                if response_created_value is None:
                    response_created_value = meta.get("response_created")
                if response_format_value is None:
                    response_format_value = meta.get("response_format")
                if finish_reason_value is None:
                    finish_reason_value = meta.get("finish_reason")
                if meta.get("parse_status") == "retry_success":
                    parse_status = "retry_success"

    result = {
        "final_answer": str(agent2_result.get("final_answer", "")),
        "raw_response": str(agent2_meta.get("raw_response", "")),
        "parsed_response": {
            "agent1_result": agent1_result,
            "agent2_result": agent2_result,
            "agent3_result": agent3_result,
        },
        "prompt_tokens": prompt_tokens_sum,
        "completion_tokens": completion_tokens_sum,
        "total_tokens": total_tokens_sum,
        "latency_sec": latency_sec_sum,
        "model": model_value,
        "seed": seed_value,
        "temperature": temperature_value,
        "max_tokens": max_tokens_value,
        "retry_count": retry_count,
        "parse_status": parse_status,
        "api_endpoint": api_endpoint_value,
        "api_type": api_type_value,
        "sdk_name": sdk_name_value,
        "sdk_version": sdk_version_value,
        "request_model": request_model_value,
        "response_model": response_model_value,
        "system_fingerprint": system_fingerprint_value,
        "response_created": response_created_value,
        "response_format": response_format_value,
        "finish_reason": finish_reason_value,
        "agent1_result": agent1_result,
        "agent2_result": agent2_result,
        "agent3_result": agent3_result,
    }
    debug_print(
        debug,
        "pmr.metrics_summary",
        mode="pmr",
        seed=seed_value if isinstance(seed_value, int) else None,
        payload={
            "prompt_tokens": prompt_tokens_sum,
            "completion_tokens": completion_tokens_sum,
            "total_tokens": total_tokens_sum,
            "latency_sec": latency_sec_sum,
            "retry_count": retry_count,
            "parse_status": parse_status,
        },
    )
    return result


def run_all_modes(task_text: str, *, seed: int | None = None, debug: bool = False) -> dict[str, dict[str, Any]]:
    return {
        "direct": run_direct(task_text, seed=seed, debug=debug),
        "cot": run_cot(task_text, seed=seed, debug=debug),
        "ps": run_ps(task_text, seed=seed, debug=debug),
        "pmr": run_pmr(task_text, seed=seed, debug=debug),
    }

