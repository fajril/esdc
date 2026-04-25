from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from phoenix.evals import LLM as PhoenixLLM  # noqa: N811
    from phoenix.evals.metrics import (
        ToolInvocationEvaluator,
        ToolResponseHandlingEvaluator,
        ToolSelectionEvaluator,
    )

logger = logging.getLogger("esdc.phoenix.evals")

_judge_llm: PhoenixLLM | None = None

ESDC_TOOLS_DESCRIPTION = (
    "SQL Executor: Execute a SQL SELECT query against the ESDC database.\n"
    "Schema Inspector: Get the schema (column names and types) "
    "for tables in the ESDC database.\n"
    "Table Lister: List all available tables and views in the ESDC database.\n"
    "Model Checker: List available models for a given provider type.\n"
    "Table Selector: Get the recommended database table or view for a query.\n"
    "Uncertainty Resolver: Resolve uncertainty level to database "
    "filter values and SQL conditions.\n"
    "Timeseries Column Guide: Get the correct column names "
    "for timeseries queries.\n"
    "Resources Column Guide: Get the correct column names "
    "for static resource queries.\n"
    "Problem Cluster Search: Search for problem cluster definitions "
    "when user asks about project issues.\n"
    "Knowledge Traversal: Resolve entities and match query patterns "
    "from the ESDC knowledge graph.\n"
    "Cypher Executor: Execute a Cypher query against the ESDC knowledge graph.\n"
    "Spatial Resolver: Execute spatial queries using DuckDB's "
    "native spatial capabilities.\n"
    "Semantic Search: Search for documents by semantic similarity to the query.\n"
    "Compute Engine: Execute a shell command in a sandboxed environment.\n"
    "File Processing: Write text content to a file in the sandboxed environment.\n"
    "View File: Display a file from the sandbox inline in the chat."
)


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

    if provider_type in ("ollama", "groq", "azure_openai"):
        base_url = provider_config.get("base_url") or "http://localhost:11434/v1"
        if provider_type == "groq":
            base_url = (
                provider_config.get("base_url") or "https://api.groq.com/openai/v1"
            )
        elif provider_type == "azure_openai":
            base_url = provider_config.get("base_url") or ""
        api_key = provider_config.get("api_key") or (
            "ollama" if provider_type == "ollama" else ""
        )
        _judge_llm = LLM(
            provider="openai",
            model=model_name,
            base_url=base_url,
            api_key=api_key,
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
    """Return a tool selection evaluator using the judge LLM."""
    from phoenix.evals.metrics import ToolSelectionEvaluator

    return ToolSelectionEvaluator(llm=_create_judge_llm(), temperature=temperature)


def get_tool_invocation_evaluator(
    temperature: float = 0.0,
) -> ToolInvocationEvaluator:
    """Return a tool invocation evaluator using the judge LLM."""
    from phoenix.evals.metrics import ToolInvocationEvaluator

    return ToolInvocationEvaluator(llm=_create_judge_llm(), temperature=temperature)


def get_tool_response_handling_evaluator(
    temperature: float = 0.0,
) -> ToolResponseHandlingEvaluator:
    """Return a tool response handling evaluator using the judge LLM."""
    from phoenix.evals.metrics import ToolResponseHandlingEvaluator

    return ToolResponseHandlingEvaluator(
        llm=_create_judge_llm(), temperature=temperature
    )


def run_evaluations(
    spans_df: pd.DataFrame,
    evaluators: list[str] | None = None,
    project_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run selected evaluators against a spans DataFrame."""
    from phoenix.evals import evaluate_dataframe
    from phoenix.trace import suppress_tracing

    from esdc.configs import Config

    Config.init_config()

    if evaluators is None:
        evaluators = ["tool_selection", "tool_invocation", "tool_response_handling"]

    eval_map: dict[str, Any] = {
        "tool_selection": get_tool_selection_evaluator(),
        "tool_invocation": get_tool_invocation_evaluator(),
        "tool_response_handling": get_tool_response_handling_evaluator(),
    }

    selected = [eval_map[e] for e in evaluators if e in eval_map]
    if not selected:
        raise ValueError(
            f"No valid evaluators selected from: {evaluators}. "
            f"Available: {list(eval_map.keys())}"
        )

    if "available_tools" not in spans_df.columns:
        spans_df = spans_df.copy()
        spans_df["available_tools"] = ESDC_TOOLS_DESCRIPTION

    if (
        "tool_selection" in evaluators
        and "tool_call" in spans_df.columns
        and "tool_selection" not in spans_df.columns
    ):
        spans_df = spans_df.rename(columns={"tool_call": "tool_selection"})

    results: dict[str, pd.DataFrame] = {}
    with suppress_tracing():
        result_df = evaluate_dataframe(
            dataframe=spans_df,
            evaluators=selected,
        )
        for evaluator_name in evaluators:
            score_col = f"{evaluator_name}_score"
            if score_col in result_df.columns:
                results[evaluator_name] = result_df[[score_col]]

    logger.info("Completed evaluations: %s", list(results.keys()))
    return results


def reset_judge_llm() -> None:
    """Reset the cached judge LLM instance."""
    global _judge_llm
    _judge_llm = None
