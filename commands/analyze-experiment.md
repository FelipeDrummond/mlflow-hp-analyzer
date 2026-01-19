---
description: Analyze an MLflow experiment to find best configurations and parameter importance
---

# Analyze Experiment

Analyze the MLflow experiment named "$1" to understand hyperparameter impact and find optimal configurations.

Use the hp-analysis skill with the target metric "$2" (if not provided, ask the user which metric to optimize).

Steps:
1. Get experiment runs using `get_experiment_runs`
2. Find best runs using `get_best_runs`
3. Compute parameter importance using `compute_param_importance`
4. Compute correlations using `compute_param_correlations`
5. Provide a structured analysis with recommendations
