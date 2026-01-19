# MLflow HP Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-blue)](https://code.claude.com)

A Claude Code plugin for conversational analysis of MLflow experiments and hyperparameter tuning runs. Query your experiments with natural language, identify optimal configurations, and get intelligent suggestions for your next experiments.

## Features

- **Conversational Analysis**: Ask questions about your experiments in natural language
- **HP Importance**: Identify which hyperparameters most impact your metrics
- **Best Config Discovery**: Quickly find top-performing configurations
- **Smart Suggestions**: Get data-driven recommendations for next experiments
- **Domain Patterns**: Built-in expertise for Vision, Multimodal, and general ML workflows

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mlflow-hp-analyzer.git
cd mlflow-hp-analyzer

# Install the plugin
claude plugin install .
```

### Configuration

Set your MLflow tracking URI:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"  # or your MLflow server
```

### Usage

**Slash Commands:**

```bash
# Analyze an experiment
/analyze-experiment my_experiment --metric val_loss

# Get HP suggestions
/suggest-hps my_experiment --metric mape --n 5

# Compare runs
/compare-runs run1_id run2_id
```

**Conversational:**

```
> Look at my soc_vit_fusion experiment. Which learning rate worked best?

> Why did runs with batch_size > 64 perform worse?

> Based on my experiments, what should I try next?
```

## Architecture

```
mlflow-hp-analyzer/
├── plugin.json              # Plugin manifest
├── mcp/
│   └── mlflow-server/       # MCP server for MLflow access
│       ├── server.py        # Tool implementations
│       └── requirements.txt
├── skills/
│   └── hp-analysis/         # Analysis expertise
│       ├── SKILL.md         # Core methodology
│       ├── resources/       # Domain patterns
│       └── scripts/         # Analysis utilities
├── commands/                # Slash commands
│   ├── analyze-experiment.md
│   ├── suggest-hps.md
│   └── compare-runs.md
└── tests/
```

### Components

| Component | Purpose |
|-----------|---------|
| **MCP Server** | Provides tools to query MLflow (list experiments, get runs, compute metrics) |
| **Skills** | Encodes HP analysis expertise and domain-specific patterns |
| **Commands** | Quick access to common analysis workflows |

## MCP Tools

| Tool | Description |
|------|-------------|
| `list_experiments` | List all available MLflow experiments |
| `get_experiment_runs` | Fetch runs with params, metrics, tags |
| `get_best_runs` | Get top-k runs by target metric |
| `get_metric_history` | Step-by-step metric values for learning curves |
| `compute_param_correlations` | Quick correlation analysis |
| `compute_param_importance` | RF-based feature importance |

## Requirements

- Python 3.10+
- Claude Code CLI
- MLflow 2.10+ (tracking server accessible)

### Dependencies

```
mcp>=1.0.0
mlflow>=2.10.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r mcp/mlflow-server/requirements.txt

# Install dev dependencies
pip install pytest ruff

# Run tests
pytest tests/
```

### Testing the MCP Server

```bash
# Start the server manually
python mcp/mlflow-server/server.py

# Or register with Claude Code
claude mcp add mlflow-hp-analyzer python mcp/mlflow-server/server.py
```

### Project Structure

```bash
# Create a new command
touch commands/my-command.md

# Add domain patterns
touch skills/hp-analysis/resources/my-domain.md
```

## Example Output

```
## Experiment: soc_vit_fusion
**Runs analyzed**: 127 | **Target**: mape (↓)

### Best Configuration
| Parameter | Value |
|-----------|-------|
| learning_rate | 0.0003 |
| weight_decay | 0.1 |
| embed_dim | 256 |
| num_heads | 8 |

**Result**: mape = 0.138

### Parameter Importance
1. learning_rate: 34% — strong negative correlation (lower is better)
2. weight_decay: 22% — optimal around 0.1
3. embed_dim: 18% — 256 outperforms 128/512

### Recommendations
- Focus search on lr ∈ [1e-4, 5e-4]
- Try weight_decay ∈ [0.05, 0.15]
- embed_dim=256 appears optimal, consider fixing
```

## Roadmap

- [ ] v0.1: Core MCP tools + basic analysis skill
- [ ] v0.2: Slash commands + suggestion engine
- [ ] v0.3: Visualization support (learning curves, parallel coords)
- [ ] v0.4: Integration with Optuna/Ray Tune metadata

## Contributing

Contributions welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MLflow](https://mlflow.org/) for experiment tracking
- [Claude Code](https://code.claude.com/) for the plugin framework
- [MCP](https://modelcontextprotocol.io/) for the tool protocol
