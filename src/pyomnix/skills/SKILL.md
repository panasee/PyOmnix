---
name: pyomnix
description: Use PyOmnix for scientific data workflows, LangGraph agents, plotting, and AI utilities. Use when working with PyOmnix, building LangGraph agents, processing scientific tabular data, configuring LLM providers, or using DataManipulator, ModelConfig, or GraphSession.
---

# PyOmnix Skill

PyOmnix is a Python toolkit for scientific data workflows, plotting, and AI-oriented utilities. Use this skill when working with PyOmnix code, building agents, or processing scientific data.

## Quick Start

```python
from pyomnix import setup_logger, get_logger, DataManipulator, set_envs
from pyomnix.consts import set_paths, OMNIX_PATH

# Paths: set OMNIX_PATH env var or call set_paths(omnix_path="/path/to/data")
set_paths()
setup_logger()
logger = get_logger(__name__)
```

**Environment**: Set `OMNIX_PATH` or `PYLAB_DB_LOCAL` for config/log paths. API keys live in `{OMNIX_PATH}/api_config.json`.

---

## 1. Logging

```python
import logging
from pyomnix import setup_logger, get_logger

setup_logger(name="PyOmnix", log_level=logging.INFO, log_file="log/app.log")
logger = get_logger(__name__)
logger.info("Ready")
logger.trace("Trace-level")  # Custom TRACE level
```

- `setup_logger()`: Configure logger; resets existing handlers.
- `get_logger(name)`: Get or create logger. `get_logger(None)` returns default.
- `TempLogLevel(logger, level)`: Context manager for temporary log level.
- `close_logger(name)`: Close handlers (important on Windows).

---

## 2. Data Processing

### DataManipulator

Load, manipulate, and plot tabular data (DataFrames). Supports Matplotlib (TeX/PGF) and Plotly/Dash.

```python
from pyomnix import DataManipulator

dm = DataManipulator(1, plot_params=(2, 2), usetex=False)
dm.load_dfs(loc=0, data_in="data.csv")
dfs = dm.get_datas(loc=0)
```

- **Plotting**: `live_plot_init()` + `live_plot_update()` for live Plotly; `create_plotly_figure()` for custom figures.
- **Dash**: `create_dash()` to embed Plotly in a Dash app.
- **CLI**: `gui_easy_data` launches PyQt data GUI; `gui_pan_color` launches color palette selector.

### DataSplitter

Split field-sweep datasets into segments by direction.

```python
from pyomnix.data_process.data_splitter import DataSplitter, SweepSegment

splitter = DataSplitter()
segments = splitter.detect_sweep_segments(df, field_col="B", min_points=10)
split_dfs = splitter.split_dataframe_by_segments(df, segments)
```

---

## 3. LangGraph Agents

### Model Configuration

```python
from pyomnix.agents.models_settings import ModelConfig

config = ModelConfig()
models = config.setup_model_factory("deepseek")  # or "openai", "google_genai", etc.
model = models["deepseek"].with_config(llm_model="deepseek-chat", llm_temperature=0.7)
```

**Provider aliases**: `openai`, `anthropic`, `google_vertexai`, `google_genai`, `deepseek`, `groq`, `huggingface`, `ollama`, etc. Config from `{OMNIX_PATH}/api_config.json`.

### Graph Builders

```python
from pyomnix.agents.graphs import build_chat_graph, build_tool_agent_graph, build_self_correction_graph
from pyomnix.agents.storage import get_checkpointer

# Chat with auto-summarization
workflow = build_chat_graph(model)

# ReAct tool-calling agent
workflow = build_tool_agent_graph(model, tools=[tavily_search, read_file, ...])

# Self-correction with human-in-the-loop
workflow = build_self_correction_graph(model)
```

### GraphSession

```python
from pyomnix.agents.graphs import GraphSession

graph = workflow.compile(checkpointer=checkpointer)
session = GraphSession(graph, thread_id="my-thread", config_dict={"configurable": {"max_history_messages": 10}})

# Invoke
result = await session.ainvoke({"messages": [HumanMessage(content="Hello")]})

# Stream
async for event in session.astream(input_state):
    ...

# Inspect
await session.inspect_state()
await session.export_history(Path("./exports"))
```

### Checkpointing

```python
from pyomnix.agents.storage import get_checkpointer

async with get_checkpointer() as checkpointer:
    graph = workflow.compile(checkpointer=checkpointer)
    # Use graph...
```

Uses PostgreSQL (Supabase) or SQLite from `api_config.json`.

---

## 4. Agent Tools

Built-in LangChain tools in `pyomnix.agents.tools`:

| Tool | Purpose |
|------|---------|
| `tavily_search` | Web search (Tavily API) |
| `google_search` | Backward-compatible alias for `tavily_search` |
| `get_current_time` | Current date and time in ISO format |
| `read_file`, `write_file` | File I/O (sandboxed under `PYOMNIX_TOOL_ROOT`) |
| `list_directory`, `file_search` | Directory listing, recursive search |
| `execute_python_code` | Safe Python execution (blocked: os, subprocess, open, etc.) |
| `calculator` | Math expression evaluation |
| `fetch_url`, `extract_text_from_url` | HTTP fetch, HTML text extraction |
| `read_csv`, `write_csv`, `read_json`, `write_json` | Tabular/JSON I/O |
| `translate_text` | Translation |
| `get_weather` | Weather (wttr.in) |
| `base64_encode`, `base64_decode`, `hash_text` | Encoding utilities |

Use `create_standard_tools_node(tools)` or `create_tavily_search_tools_node()` in nodes.

---

## 5. Utils

### Data (`pyomnix.utils.data`)

- `ObjectArray(*dims, fill_value)`: Multi-dimensional object storage.
- `CacheArray`: Cached array with invalidation.
- `difference()`, `loop_diff()`, `match_with_tolerance()`, `rename_duplicates()`, `symmetrize()`, `identify_direction()`.
- `extract_longest_monotonic_segment()`, `aggregate_by_distance()`, `smooth_irregular_series()`.
- `sph_to_cart()`, `scatter_to_candle()`.

### Math (`pyomnix.utils.math`)

- Constants: `HBAR`, `HPLANCK`, `KB`, `CM_TO_INCH`, `UNIT_FACTOR_FROM_SI`, `UNIT_FACTOR_TO_SI`.
- `factor(unit, mode)`: SI unit conversion.
- `convert_unit()`, `get_unit_factor_and_texname()`.
- `gen_seq()`, `time_generator()`, `timestr_convert()`.

### Plot (`pyomnix.utils.plot`)

- `PlotParam(*dims)`: Plot parameters (extends ObjectArray).
- `hex_to_rgb()`, `combine_cmap()`, `truncate_cmap()`, `hsv_analyze()`.
- `print_progress_bar()`, `add_watermark()`.

### Env & Files

- `is_notebook()`: Detect Jupyter environment.
- `set_envs()`: Set environment defaults.
- `ShPath`: Shell-like Path class (`cd`, `ls`, `cat`, `grep`, `cp`, `mv`, `rm`, `find`, `du`, etc.).

---

## 6. Configuration

**api_config.json** (at `{OMNIX_PATH}/api_config.json`):

```json
{
  "langsmith": {"api_key": "...", "base_url": "..."},
  "openai": {"api_key": "...", "base_url": "...", "provider_kwargs": {}},
  "deepseek": {"api_key": "...", "base_url": "..."}
}
```

**Optional extras**:

```bash
pip install "pyomnix[gui]"   # PyQt6 + WebEngine
pip install "pyomnix[web]"   # Reflex
pip install -e ".[dev]"      # ruff, mypy, black, pre-commit
```

---

## Common Patterns

### Minimal ReAct Agent

```python
from pyomnix.agents.graphs import GraphSession, build_tool_agent_graph
from pyomnix.agents.models_settings import ModelConfig
from pyomnix.agents.tools import tavily_search, read_file
from pyomnix.agents.storage import get_checkpointer

config = ModelConfig()
models = config.setup_model_factory("deepseek")
model = models["deepseek"].with_config(llm_model="deepseek-chat")

workflow = build_tool_agent_graph(model, tools=[tavily_search, read_file])
async with get_checkpointer() as cp:
    graph = workflow.compile(checkpointer=cp)
    session = GraphSession(graph, thread_id="agent-1")
    result = await session.ainvoke({"messages": [HumanMessage(content="Search for X")]})
```

### DataManipulator with Plotly

```python
dm = DataManipulator(1)
dm.load_dfs(loc=0, data_in="sweep.csv")
dm.live_plot_init(loc=0)
# In loop: dm.live_plot_update(loc=0, idx=...)
```

---

## Additional Resources

- For detailed API reference, see [reference.md](reference.md).
- For usage examples, see [examples.md](examples.md).
- Source: `src/pyomnix/` — `agents/`, `data_process/`, `utils/`, `pltconfig/`.
