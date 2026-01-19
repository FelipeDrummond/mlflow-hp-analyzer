---
description: Get data-driven hyperparameter suggestions for the next experiment runs
---

# Suggest Hyperparameters

Based on the MLflow experiment "$1", suggest optimal hyperparameter configurations to try next.

Target metric: "$2" (if not provided, ask the user)
Number of suggestions: "$3" (default: 5)

Analysis approach:
1. Get all runs and find the best performers
2. Analyze parameter importance to focus on what matters
3. Identify unexplored regions of the hyperparameter space
4. Suggest configurations that:
   - Refine around the best-found configurations
   - Explore promising but under-sampled regions
   - Test interactions between important parameters

Format suggestions as a table with specific values to try.
