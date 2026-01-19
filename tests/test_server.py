"""Tests for the MCP server."""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.mlflow_server.server import (
    _runs_to_dataframe,
    list_tools,
    call_tool,
)


class TestRunsToDataframe:
    """Tests for _runs_to_dataframe utility function."""

    def test_converts_runs_to_dataframe(self, mock_runs):
        """Test basic conversion of runs to DataFrame."""
        df = _runs_to_dataframe(mock_runs)

        assert len(df) == 2
        assert "run_id" in df.columns
        assert "param.learning_rate" in df.columns
        assert "metric.val_loss" in df.columns

    def test_empty_runs_returns_empty_dataframe(self):
        """Test that empty runs list returns empty DataFrame."""
        df = _runs_to_dataframe([])
        assert len(df) == 0

    def test_preserves_run_metadata(self, mock_runs):
        """Test that run metadata is preserved."""
        df = _runs_to_dataframe(mock_runs)

        assert df.iloc[0]["run_id"] == "run_001"
        assert df.iloc[0]["run_name"] == "test_run_1"
        assert df.iloc[0]["status"] == "FINISHED"

    def test_prefixes_params_and_metrics(self, mock_runs):
        """Test that params and metrics get correct prefixes."""
        df = _runs_to_dataframe(mock_runs)

        param_cols = [c for c in df.columns if c.startswith("param.")]
        metric_cols = [c for c in df.columns if c.startswith("metric.")]

        assert len(param_cols) == 2
        assert len(metric_cols) == 2


class TestListTools:
    """Tests for the list_tools endpoint."""

    @pytest.mark.asyncio
    async def test_returns_all_tools(self):
        """Test that list_tools returns all expected tools."""
        tools = await list_tools()

        tool_names = [t.name for t in tools]
        expected_names = [
            "list_experiments",
            "get_experiment_runs",
            "get_best_runs",
            "get_metric_history",
            "compute_param_correlations",
            "compute_param_importance",
        ]

        assert len(tools) == 6
        for name in expected_names:
            assert name in tool_names

    @pytest.mark.asyncio
    async def test_tools_have_valid_schemas(self):
        """Test that all tools have valid input schemas."""
        tools = await list_tools()

        for tool in tools:
            assert tool.inputSchema is not None
            assert tool.inputSchema.get("type") == "object"
            assert "properties" in tool.inputSchema
            assert "required" in tool.inputSchema


class TestCallTool:
    """Tests for the call_tool endpoint."""

    @pytest.mark.asyncio
    async def test_list_experiments(self, mock_experiment):
        """Test list_experiments tool."""
        with patch("src.mlflow_server.server.client") as mock_client:
            mock_client.search_experiments.return_value = [mock_experiment]

            result = await call_tool("list_experiments", {})

            assert len(result) == 1
            data = json.loads(result[0].text)
            assert len(data) == 1
            assert data[0]["name"] == "test_experiment"

    @pytest.mark.asyncio
    async def test_get_experiment_runs(self, mock_experiment, mock_runs):
        """Test get_experiment_runs tool."""
        with patch("src.mlflow_server.server.client") as mock_client:
            mock_client.get_experiment_by_name.return_value = mock_experiment
            mock_client.search_runs.return_value = mock_runs

            result = await call_tool(
                "get_experiment_runs",
                {"experiment_name": "test_experiment"},
            )

            assert len(result) == 1
            data = json.loads(result[0].text)
            assert len(data) == 2

    @pytest.mark.asyncio
    async def test_get_experiment_runs_not_found(self):
        """Test get_experiment_runs with non-existent experiment."""
        with patch("src.mlflow_server.server.client") as mock_client:
            mock_client.get_experiment_by_name.return_value = None

            result = await call_tool(
                "get_experiment_runs",
                {"experiment_name": "nonexistent"},
            )

            assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_get_best_runs(self, mock_experiment, mock_runs):
        """Test get_best_runs tool."""
        with patch("src.mlflow_server.server.client") as mock_client:
            mock_client.get_experiment_by_name.return_value = mock_experiment
            mock_client.search_runs.return_value = mock_runs[:1]

            result = await call_tool(
                "get_best_runs",
                {
                    "experiment_name": "test_experiment",
                    "metric": "val_loss",
                    "k": 1,
                },
            )

            assert len(result) == 1
            data = json.loads(result[0].text)
            assert len(data) == 1

    @pytest.mark.asyncio
    async def test_get_metric_history(self, mock_metric_history):
        """Test get_metric_history tool."""
        with patch("src.mlflow_server.server.client") as mock_client:
            mock_client.get_metric_history.return_value = mock_metric_history

            result = await call_tool(
                "get_metric_history",
                {"run_id": "run_001", "metric": "val_loss"},
            )

            assert len(result) == 1
            data = json.loads(result[0].text)
            assert len(data) == 4
            assert data[0]["step"] == 0
            assert data[0]["value"] == 1.0

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """Test calling an unknown tool."""
        result = await call_tool("unknown_tool", {})
        assert "Unknown tool" in result[0].text

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test that errors are caught and returned."""
        with patch("src.mlflow_server.server.client") as mock_client:
            mock_client.search_experiments.side_effect = Exception("Connection failed")

            result = await call_tool("list_experiments", {})

            assert "Error:" in result[0].text
            assert "Connection failed" in result[0].text
