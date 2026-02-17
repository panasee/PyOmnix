# PyOmnix

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

PyOmnix is a Python toolkit for scientific data workflows, plotting, and AI-oriented utilities.
It is built as a modular library so you can use only the components you need.

## Overview

PyOmnix currently provides:

- Data processing helpers for experimental/scientific tabular data.
- Plotting utilities built around Matplotlib and Plotly.
- GUI tools for fast data inspection and manipulation.
- Agent-related modules for model/tool orchestration.
- Logging, environment, file, and math utility helpers.

## Installation

### Base

```bash
pip install pyomnix
```

### Optional extras

```bash
# GUI support (PyQt6 + WebEngine)
pip install "pyomnix[gui]"

# Web-related tools
pip install "pyomnix[web]"

# Development toolchain
pip install -e ".[dev]"
```

## Quick Start

### Logging

```python
from pyomnix import setup_logger, get_logger

setup_logger()
logger = get_logger(__name__)
logger.info("PyOmnix is ready.")
```

### Data Manipulator

```python
from pyomnix.data_process import DataManipulator

dm = DataManipulator(1)
# Example:
# dm.load_dfs(loc=0, data_in="data.csv")
# dfs = dm.get_datas(loc=0)
```

## CLI Entry Points

After installation, these commands are available:

- `gui_pan_color`: Launch the color palette selector.
- `gui_easy_data`: Launch the PyQt-based data GUI.

## Project Structure

- `src/pyomnix/data_process/`: data loading, splitting, plotting, GUI tools.
- `src/pyomnix/utils/`: reusable data/math/plot/environment/file utilities.
- `src/pyomnix/agents/`: graph, node, tool, prompt, storage, and settings modules.
- `src/pyomnix/omnix_logger.py`: logging framework used across the project.
- `src/pyomnix/pltconfig/`: plotting and color configuration assets.

## Development

```bash
# Lint
ruff check .

# Format (if you use black in your workflow)
black .

# Run tests
pytest
```

## Requirements

- Python `>=3.12`

See `pyproject.toml` for the complete dependency list.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
