"""Pytest configuration and fixtures for MCP server tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock
from dataclasses import dataclass

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MockRunInfo:
    """Mock MLflow RunInfo object."""
    run_id: str
    run_name: str
    status: str
    start_time: int
    end_time: int


@dataclass
class MockRunData:
    """Mock MLflow RunData object."""
    params: dict
    metrics: dict


@dataclass
class MockRun:
    """Mock MLflow Run object."""
    info: MockRunInfo
    data: MockRunData


@dataclass
class MockExperiment:
    """Mock MLflow Experiment object."""
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str


@dataclass
class MockMetricHistory:
    """Mock MLflow Metric history entry."""
    step: int
    value: float
    timestamp: int


@pytest.fixture
def mock_runs():
    """Create mock MLflow runs for testing."""
    return [
        MockRun(
            info=MockRunInfo(
                run_id="run_001",
                run_name="test_run_1",
                status="FINISHED",
                start_time=1000000,
                end_time=1000100,
            ),
            data=MockRunData(
                params={"learning_rate": "0.001", "batch_size": "32"},
                metrics={"val_loss": 0.5, "val_accuracy": 0.85},
            ),
        ),
        MockRun(
            info=MockRunInfo(
                run_id="run_002",
                run_name="test_run_2",
                status="FINISHED",
                start_time=1000200,
                end_time=1000300,
            ),
            data=MockRunData(
                params={"learning_rate": "0.01", "batch_size": "64"},
                metrics={"val_loss": 0.3, "val_accuracy": 0.90},
            ),
        ),
    ]


@pytest.fixture
def mock_experiment():
    """Create a mock MLflow experiment."""
    return MockExperiment(
        experiment_id="exp_001",
        name="test_experiment",
        artifact_location="/mlflow/artifacts",
        lifecycle_stage="active",
    )


@pytest.fixture
def mock_metric_history():
    """Create mock metric history entries."""
    return [
        MockMetricHistory(step=0, value=1.0, timestamp=1000000),
        MockMetricHistory(step=1, value=0.8, timestamp=1000010),
        MockMetricHistory(step=2, value=0.6, timestamp=1000020),
        MockMetricHistory(step=3, value=0.4, timestamp=1000030),
    ]
