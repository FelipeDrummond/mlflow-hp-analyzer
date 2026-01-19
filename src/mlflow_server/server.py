#!/usr/bin/env python3
"""MCP server for MLflow hyperparameter analysis.

Assumptions and Limitations:
- All runs in an experiment are expected to have consistent parameter names.
  Runs with different parameter sets will result in sparse DataFrames where
  missing parameters are represented as NaN.
- Correlation and importance analysis require numeric parameters. String
  parameters are automatically filtered out.
- Parameter importance requires at least 10 data points and 2+ numeric params.
- The server connects to MLflow at startup; connection errors will cause
  tool calls to fail with descriptive error messages.
"""

import json
import os
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor

server = Server("mlflow-hp-analyzer")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def _runs_to_dataframe(runs: list) -> pd.DataFrame:
    """Convert MLflow runs to a DataFrame with params and metrics."""
    records = []
    for run in runs:
        record = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }
        record.update({f"param.{k}": v for k, v in run.data.params.items()})
        record.update({f"metric.{k}": v for k, v in run.data.metrics.items()})
        records.append(record)
    return pd.DataFrame(records)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MLflow analysis tools."""
    return [
        Tool(
            name="list_experiments",
            description="List all MLflow experiments in the tracking server",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="get_experiment_runs",
            description="Get all runs for an experiment with params and metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Name of the MLflow experiment",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of runs to return (default: 100)",
                        "default": 100,
                    },
                },
                "required": ["experiment_name"],
            },
        ),
        Tool(
            name="get_best_runs",
            description="Get top-k runs by a target metric",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Name of the MLflow experiment",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Target metric to optimize",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of top runs to return (default: 5)",
                        "default": 5,
                    },
                    "ascending": {
                        "type": "boolean",
                        "description": "Sort ascending (True for loss metrics, False for accuracy)",
                        "default": True,
                    },
                },
                "required": ["experiment_name", "metric"],
            },
        ),
        Tool(
            name="get_metric_history",
            description="Get step-by-step metric values for a specific run (learning curves)",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "MLflow run ID",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Metric name to retrieve history for",
                    },
                },
                "required": ["run_id", "metric"],
            },
        ),
        Tool(
            name="compute_param_correlations",
            description="Compute correlation between hyperparameters and a target metric",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Name of the MLflow experiment",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Target metric for correlation analysis",
                    },
                },
                "required": ["experiment_name", "metric"],
            },
        ),
        Tool(
            name="compute_param_importance",
            description="Compute feature importance of hyperparameters using Random Forest",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_name": {
                        "type": "string",
                        "description": "Name of the MLflow experiment",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Target metric for importance analysis",
                    },
                },
                "required": ["experiment_name", "metric"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute an MLflow analysis tool."""
    try:
        if name == "list_experiments":
            experiments = client.search_experiments()
            result = [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                }
                for exp in experiments
            ]
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_experiment_runs":
            experiment = client.get_experiment_by_name(arguments["experiment_name"])
            if not experiment:
                return [
                    TextContent(
                        type="text", text=f"Experiment '{arguments['experiment_name']}' not found"
                    )
                ]

            max_results = arguments.get("max_results", 100)
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
            )
            df = _runs_to_dataframe(runs)
            return [TextContent(type="text", text=df.to_json(orient="records", indent=2))]

        elif name == "get_best_runs":
            experiment = client.get_experiment_by_name(arguments["experiment_name"])
            if not experiment:
                return [
                    TextContent(
                        type="text", text=f"Experiment '{arguments['experiment_name']}' not found"
                    )
                ]

            metric = arguments["metric"]
            k = arguments.get("k", 5)
            ascending = arguments.get("ascending", True)
            order = "ASC" if ascending else "DESC"

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} {order}"],
                max_results=k,
            )
            df = _runs_to_dataframe(runs)
            return [TextContent(type="text", text=df.to_json(orient="records", indent=2))]

        elif name == "get_metric_history":
            run_id = arguments["run_id"]
            metric = arguments["metric"]
            history = client.get_metric_history(run_id, metric)
            result = [{"step": h.step, "value": h.value, "timestamp": h.timestamp} for h in history]
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "compute_param_correlations":
            experiment = client.get_experiment_by_name(arguments["experiment_name"])
            if not experiment:
                return [
                    TextContent(
                        type="text", text=f"Experiment '{arguments['experiment_name']}' not found"
                    )
                ]

            runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=500)
            df = _runs_to_dataframe(runs)

            metric_col = f"metric.{arguments['metric']}"
            if metric_col not in df.columns:
                return [
                    TextContent(
                        type="text", text=f"Metric '{arguments['metric']}' not found in runs"
                    )
                ]

            param_cols = [c for c in df.columns if c.startswith("param.")]
            correlations = {}

            for col in param_cols:
                try:
                    numeric_vals = pd.to_numeric(df[col], errors="coerce")
                    if numeric_vals.notna().sum() >= 3:
                        corr = numeric_vals.corr(df[metric_col])
                        if not np.isnan(corr):
                            correlations[col.replace("param.", "")] = round(corr, 4)
                except Exception:
                    continue

            sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
            return [TextContent(type="text", text=json.dumps(sorted_corr, indent=2))]

        elif name == "compute_param_importance":
            experiment = client.get_experiment_by_name(arguments["experiment_name"])
            if not experiment:
                return [
                    TextContent(
                        type="text", text=f"Experiment '{arguments['experiment_name']}' not found"
                    )
                ]

            runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=500)
            df = _runs_to_dataframe(runs)

            metric_col = f"metric.{arguments['metric']}"
            if metric_col not in df.columns:
                return [
                    TextContent(
                        type="text", text=f"Metric '{arguments['metric']}' not found in runs"
                    )
                ]

            param_cols = [c for c in df.columns if c.startswith("param.")]
            X_data = {}
            for col in param_cols:
                try:
                    numeric_vals = pd.to_numeric(df[col], errors="coerce")
                    if numeric_vals.notna().sum() >= 10:
                        X_data[col] = numeric_vals
                except Exception:
                    continue

            if len(X_data) < 2:
                return [
                    TextContent(
                        type="text", text="Not enough numeric parameters for importance analysis"
                    )
                ]

            X = pd.DataFrame(X_data)
            y = df[metric_col]
            mask = X.notna().all(axis=1) & y.notna()
            X, y = X[mask], y[mask]

            if len(X) < 10:
                return [
                    TextContent(
                        type="text", text="Not enough valid data points for importance analysis"
                    )
                ]

            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            importance = dict(
                zip(
                    [c.replace("param.", "") for c in X.columns],
                    [round(imp, 4) for imp in rf.feature_importances_],
                    strict=False,
                )
            )
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            return [TextContent(type="text", text=json.dumps(sorted_importance, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
