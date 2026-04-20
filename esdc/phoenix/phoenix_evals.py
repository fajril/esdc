from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phoenix.evals import LLM as PhoenixLLM  # noqa: N811
    from phoenix.evals.metrics import (
        ToolInvocationEvaluator,
        ToolResponseHandlingEvaluator,
        ToolSelectionEvaluator,
    )

logger = logging.getLogger("esdc.phoenix.evals")

_judge_llm: PhoenixLLM | None = None


def _create_judge_llm() -> PhoenixLLM:
    global _judge_llm
    if _judge_llm is not None:
        return _judge_llm

    from phoenix.evals import LLM

    from esdc.configs import Config

    provider_config = Config.get_provider_config()
    if not provider_config:
        raise RuntimeError(
            "Cannot create judge LLM: no provider configured. "
            "Run 'esdc chat --setup' or configure provider first."
        )

    provider_type: str = provider_config["provider_type"]
    model_name: str = provider_config["model"]

    if provider_type == "ollama":
        base_url = provider_config.get("base_url") or "http://localhost:11434/v1"
        _judge_llm = LLM(
            provider="openai",
            model=model_name,
            base_url=base_url,
            api_key="ollama",
        )
    elif provider_type == "openai_compatible":
        base_url = provider_config["base_url"]
        api_key = provider_config.get("api_key") or "not-needed"
        _judge_llm = LLM(
            provider="openai",
            model=model_name,
            base_url=base_url,
            api_key=api_key,
        )
    else:
        logger.warning(
            "Judge model: provider type '%s' is not openai-compatible, "
            "but Phoenix evaluators require openai-compatible LLM. "
            "Falling back to direct model name.",
            provider_type,
        )
        _judge_llm = LLM(provider="openai", model=model_name)

    logger.info(
        "Created judge LLM | provider_type=%s model=%s base_url=%s",
        provider_type,
        model_name,
        provider_config.get("base_url"),
    )
    return _judge_llm


def get_tool_selection_evaluator(
    temperature: float = 0.0,
) -> ToolSelectionEvaluator:
    from phoenix.evals.metrics import ToolSelectionEvaluator

    return ToolSelectionEvaluator(llm=_create_judge_llm(), temperature=temperature)


def get_tool_invocation_evaluator(
    temperature: float = 0.0,
) -> ToolInvocationEvaluator:
    from phoenix.evals.metrics import ToolInvocationEvaluator

    return ToolInvocationEvaluator(llm=_create_judge_llm(), temperature=temperature)


def get_tool_response_handling_evaluator(
    temperature: float = 0.0,
) -> ToolResponseHandlingEvaluator:
    from phoenix.evals.metrics import ToolResponseHandlingEvaluator

    return ToolResponseHandlingEvaluator(
        llm=_create_judge_llm(), temperature=temperature
    )


def reset_judge_llm() -> None:
    global _judge_llm
    _judge_llm = None
