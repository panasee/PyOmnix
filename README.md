# 🔬 PyOmnix

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**PyOmnix** is an integrated package designed for scientific computing, data analysis, and AI development assistance. It provides a comprehensive suite of tools for data processing, visualization, and AI model integration.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [License](#license)

## 🌟 Overview

PyOmnix is a versatile Python package that combines various tools and utilities for:

- 📊 **Data Processing**: Efficient data manipulation and analysis tools
- 📈 **Scientific Visualization**: Advanced plotting capabilities
- 🤖 **AI Integration**: Seamless AI model integration and management
- ⚙️ **Workflow Automation**: Streamlined workflow management
- 📝 **Logging & Monitoring**: Comprehensive logging and monitoring solutions

The package is designed to be modular and extensible, allowing users to integrate different components as needed.

## 💻 Installation

### Installation
```bash
pip install pyomnix
```

### For Development
```bash
# git clone and cd to dir
pip install -e ".[dev]"
```

### For GUI Support
```bash
pip install "pyomnix[gui]"
```

## 🚀 Usage

### Logger
```python
from pyomnix import setup_logger, get_logger

# Setup logging with default configuration
logger = setup_logger()

# Get a logger instance
logger = get_logger(__name__)
```

### GUI Application
```bash
# Launch the GUI application
gui_pan_color
```

## ✨ Features

### Core Features
- **Data Processing**: Tools for data manipulation and analysis
- **Visualization**: Plotting capabilities with matplotlib and plotly
- **AI Integration**: Support for AI models and frameworks
- **Workflow Management**: Prefect-based workflow automation

### Key Components
- 📁 `data_process/`: Data processing and analysis tools
- 🤖 `model_interface/`: AI model integration
- 🛠️ `utils/`: Utility functions and helpers
- 📝 `omnix_logger.py`: Advanced logging system

## 📦 Dependencies

### Core Dependencies
| Package | Purpose |
|---------|---------|
| numpy | Numerical computing |
| pandas | Data manipulation |
| matplotlib | Basic plotting |
| plotly | Interactive visualization |
| jupyter | Notebook support |
| prefect | Workflow support |
| pydantic | Data validation |
| langchain | AI framework integration |
| langgraph | Graph-based AI workflows |

### Optional Dependencies
- PyQt6: GUI support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
