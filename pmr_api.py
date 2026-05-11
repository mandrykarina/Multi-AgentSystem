from __future__ import annotations

from typing import Any

from agents.agent1 import analyze_task
from agents.agent2 import execute_task
from agents.agent3 import reflect_solution


def run_agent1(task_text: str) -> dict[str, Any]:
    """
    Публичный API-вход для запуска Агент 1.
    """
    return analyze_task(task_text)


def run_agent2(task_text: str, agent1_result: dict[str, Any]) -> dict[str, Any]:
    """
    Публичный API-вход для запуска Агент 2.
    """
    return execute_task(task_text, agent1_result)


def run_pipeline(task_text: str) -> dict[str, Any]:
    """
    Публичный API пайплайна Agent1 -> Agent2.
    """
    agent1_result = run_agent1(task_text)
    agent2_result = run_agent2(task_text, agent1_result)
    return {
        "task_text": task_text,
        "agent1_result": agent1_result,
        "agent2_result": agent2_result,
    }


def run_agent3(
    task_text: str,
    agent1_result: dict[str, Any],
    agent2_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Публичный API-вход для запуска Агент 3.
    """
    return reflect_solution(task_text, agent1_result, agent2_result)


def run_pipeline_v3(task_text: str) -> dict[str, Any]:
    """
    Публичный API пайплайна Agent1 -> Agent2 -> Agent3.
    """
    agent1_result = run_agent1(task_text)
    agent2_result = run_agent2(task_text, agent1_result)
    agent3_result = run_agent3(task_text, agent1_result, agent2_result)
    return {
        "task_text": task_text,
        "agent1_result": agent1_result,
        "agent2_result": agent2_result,
        "agent3_result": agent3_result,
    }

