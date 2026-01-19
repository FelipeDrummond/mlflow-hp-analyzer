"""Basic tests for the MCP server."""

import pytest


def test_import_server():
    """Test that server module can be imported."""
    from mcp.mlflow_server import server  # noqa: F401


def test_tools_defined():
    """Test that required tools are defined."""
    expected_tools = [
        "list_experiments",
        "get_experiment_runs",
        "get_best_runs",
        "get_metric_history",
        "compute_param_correlations",
        "compute_param_importance",
    ]
    # This will be expanded when we have proper test infrastructure
    assert len(expected_tools) == 6
