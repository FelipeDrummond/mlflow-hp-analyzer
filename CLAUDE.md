# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Claude Code plugin** for conversational analysis of MLflow experiments and hyperparameter tuning runs. It provides MCP tools to query MLflow, analyze HP importance, and suggest optimal configurations.

## Development Commands

```bash
# Install all dependencies (including dev)
pip install -r requirements-dev.txt

# Install only runtime dependencies
pip install -r src/mlflow_server/requirements.txt

# Run the MCP server directly (for testing)
python src/mlflow_server/server.py

# Install plugin locally
claude plugin install .

# Run tests
pytest tests/ -v

# Run single test
pytest tests/test_server.py::TestCallTool::test_list_experiments -v

# Lint
ruff check .

# Format
ruff format .

# Run pre-commit hooks
pre-commit run --all-files
```

## Environment

Set `MLFLOW_TRACKING_URI` to point to your MLflow server:
```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

## Architecture

This is a Claude Code plugin with three main components:

### 1. MCP Server (`src/mlflow_server/server.py`)
Async Python server using the MCP SDK that exposes tools to Claude:
- `list_experiments` - List all MLflow experiments
- `get_experiment_runs` - Get runs with params/metrics as JSON
- `get_best_runs` - Top-k runs sorted by target metric
- `get_metric_history` - Per-step metric values (learning curves)
- `compute_param_correlations` - Pearson correlation of params vs metric
- `compute_param_importance` - Random Forest feature importance

The server uses `MlflowClient` for all MLflow API calls and converts runs to pandas DataFrames internally via `_runs_to_dataframe()`.

### 2. Skills (`skills/hp-analysis/SKILL.md`)
Encodes HP analysis expertise. Defines the structured response format and analysis workflow that Claude should follow.

### 3. Slash Commands (`commands/`)
- `/analyze-experiment <name> [metric]` - Full experiment analysis
- `/suggest-hps <name> <metric> [n]` - Get suggestions for next runs
- `/compare-runs <run_ids...>` - Compare specific runs

## Plugin Configuration

- `.claude-plugin/plugin.json` - Plugin manifest (name, version, paths)
- `.mcp.json` - MCP server configuration with `${CLAUDE_PLUGIN_ROOT}` variable

## Key Patterns

- All MCP tools return `list[TextContent]` with JSON-formatted results
- Experiment lookups use `client.get_experiment_by_name()` - always check for None
- Numeric param conversion uses `pd.to_numeric(errors="coerce")` to handle mixed types
- Parameter importance requires minimum 10 valid data points and 2+ numeric params

## Assumptions

- All runs in an experiment should have consistent parameter names (sparse DataFrames otherwise)
- Correlation/importance analysis only works with numeric parameters
- MLflow server must be accessible at the configured tracking URI
