from unittest.mock import patch

import pytest

pytestmark = pytest.mark.allow_provider_config


@pytest.fixture(autouse=True)
def clean_judge_llm():
    from esdc.phoenix import phoenix_evals

    phoenix_evals._judge_llm = None
    yield
    phoenix_evals._judge_llm = None


@pytest.fixture
def mock_provider_config_ollama():
    return {
        "provider_type": "ollama",
        "model": "kimi-k2.5:cloud",
        "base_url": "http://localhost:11434/v1",
        "api_key": None,
    }


@pytest.fixture
def mock_provider_config_openai_compatible():
    return {
        "provider_type": "openai_compatible",
        "model": "gpt-4o-mini",
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-test",
    }


class TestCreateJudgeLLMOllama:
    def test_judge_llm_ollama(self, mock_provider_config_ollama):
        from phoenix.evals import LLM

        with patch("esdc.configs.Config.get_provider_config") as mock_get:
            mock_get.return_value = mock_provider_config_ollama

            from esdc.phoenix.phoenix_evals import _create_judge_llm

            llm = _create_judge_llm()
            assert isinstance(llm, LLM)

    def test_judge_llm_ollama_default_base_url(self):
        mock_config = {
            "provider_type": "ollama",
            "model": "kimi-k2.5:cloud",
            "base_url": None,
            "api_key": None,
        }

        with patch("esdc.configs.Config.get_provider_config") as mock_get:
            mock_get.return_value = mock_config

            from esdc.phoenix.phoenix_evals import _create_judge_llm

            llm = _create_judge_llm()
            assert llm.model == "kimi-k2.5:cloud"

    def test_judge_llm_cached(self, mock_provider_config_ollama):
        with patch("esdc.configs.Config.get_provider_config") as mock_get:
            mock_get.return_value = mock_provider_config_ollama

            from esdc.phoenix.phoenix_evals import _create_judge_llm

            llm1 = _create_judge_llm()
            llm2 = _create_judge_llm()
            assert llm1 is llm2


class TestCreateJudgeLLMOpenAICompatible:
    def test_judge_llm_openai_compatible(self, mock_provider_config_openai_compatible):
        from phoenix.evals import LLM

        with patch("esdc.configs.Config.get_provider_config") as mock_get:
            mock_get.return_value = mock_provider_config_openai_compatible

            from esdc.phoenix.phoenix_evals import _create_judge_llm

            llm = _create_judge_llm()
            assert isinstance(llm, LLM)


class TestCreateJudgeLLMErrors:
    def test_judge_llm_no_provider_config(self):
        with patch("esdc.configs.Config.get_provider_config") as mock_get:
            mock_get.return_value = None

            from esdc.phoenix.phoenix_evals import _create_judge_llm

            with pytest.raises(RuntimeError, match="no provider configured"):
                _create_judge_llm()


class TestEvaluators:
    def test_get_tool_selection_evaluator(self, mock_provider_config_ollama):
        with patch("esdc.configs.Config.get_provider_config") as mock_get:
            mock_get.return_value = mock_provider_config_ollama

            from esdc.phoenix.phoenix_evals import get_tool_selection_evaluator

            eval = get_tool_selection_evaluator()
            assert eval.__class__.__name__ == "ToolSelectionEvaluator"

    def test_get_tool_invocation_evaluator(self, mock_provider_config_ollama):
        with patch("esdc.configs.Config.get_provider_config") as mock_get:
            mock_get.return_value = mock_provider_config_ollama

            from esdc.phoenix.phoenix_evals import get_tool_invocation_evaluator

            eval = get_tool_invocation_evaluator()
            assert eval.__class__.__name__ == "ToolInvocationEvaluator"

    def test_get_tool_response_handling_evaluator(self, mock_provider_config_ollama):
        with patch("esdc.configs.Config.get_provider_config") as mock_get:
            mock_get.return_value = mock_provider_config_ollama

            from esdc.phoenix.phoenix_evals import get_tool_response_handling_evaluator

            eval = get_tool_response_handling_evaluator()
            assert eval.__class__.__name__ == "ToolResponseHandlingEvaluator"


class TestResetJudgeLLM:
    def test_reset_judge_llm(self, mock_provider_config_ollama):
        with patch("esdc.configs.Config.get_provider_config") as mock_get:
            mock_get.return_value = mock_provider_config_ollama

            from esdc.phoenix.phoenix_evals import _create_judge_llm, reset_judge_llm

            llm1 = _create_judge_llm()
            reset_judge_llm()
            llm2 = _create_judge_llm()
            assert llm1 is not llm2


class TestESDCToolsDescription:
    def test_tools_description_is_non_empty_string(self):
        from esdc.phoenix.phoenix_evals import ESDC_TOOLS_DESCRIPTION

        assert isinstance(ESDC_TOOLS_DESCRIPTION, str)
        assert len(ESDC_TOOLS_DESCRIPTION) > 0
        assert "SQL Executor" in ESDC_TOOLS_DESCRIPTION
        assert "Knowledge Traversal" in ESDC_TOOLS_DESCRIPTION
        assert "Semantic Search" in ESDC_TOOLS_DESCRIPTION


class TestRunEvaluations:
    def test_run_evaluations_invalid_evaluator(self, mock_provider_config_ollama):
        with patch("esdc.configs.Config.get_provider_config") as mock_get:
            mock_get.return_value = mock_provider_config_ollama
            with patch("esdc.configs.Config.init_config"):
                import pandas as pd

                from esdc.phoenix.phoenix_evals import run_evaluations

                df = pd.DataFrame(
                    {"input": ["test"], "tool_selection": ["SQL Executor"]}
                )
                with pytest.raises(ValueError, match="No valid evaluators"):
                    run_evaluations(df, evaluators=["nonexistent"])
