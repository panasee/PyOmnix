# PyOmnix Usage Examples

Concrete examples for common workflows.

---

## Example 1: Simple Chat Agent with Persistence

```python
import asyncio
from langchain_core.messages import HumanMessage

from pyomnix.agents.graphs import GraphSession, build_chat_graph
from pyomnix.agents.models_settings import ModelConfig
from pyomnix.agents.storage import get_checkpointer

async def main():
    config = ModelConfig()
    models = config.setup_model_factory("deepseek")
    model = models["deepseek"].with_config(llm_model="deepseek-chat", llm_temperature=0.7)

    workflow = build_chat_graph(model)
    async with get_checkpointer() as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        session = GraphSession(
            graph,
            thread_id="chat-001",
            config_dict={"configurable": {"max_history_messages": 10}},
        )
        result = await session.ainvoke({
            "messages": [HumanMessage(content="What is PyOmnix?")],
        })
        print(result["messages"][-1].content)

asyncio.run(main())
```

---

## Example 2: Tool Agent with Tavily Search

```python
import asyncio
from langchain_core.messages import HumanMessage

from pyomnix.agents.graphs import GraphSession, build_tool_agent_graph
from pyomnix.agents.models_settings import ModelConfig
from pyomnix.agents.tools import tavily_search, read_file, get_current_time

async def main():
    config = ModelConfig()
    models = config.setup_model_factory("deepseek")
    model = models["deepseek"].with_config(llm_model="deepseek-chat")

    tools = [tavily_search, read_file, get_current_time]
    workflow = build_tool_agent_graph(model, tools=tools)
    graph = workflow.compile()

    session = GraphSession(graph, thread_id="tool-agent-1")
    # Invoke with a query that requires search
    result = await session.ainvoke({
        "messages": [HumanMessage(content="Search for latest Python 3.12 release notes")],
    })
    print(result["messages"][-1].content)

asyncio.run(main())
```

---

## Example 3: DataManipulator Load and Plot

```python
from pyomnix import DataManipulator

dm = DataManipulator(1, plot_params=(2, 2))
dm.load_dfs(loc=0, data_in="data/sweep.csv")
dfs = dm.get_datas(loc=0)

# Static matplotlib (if configured)
# dm.plot(...)

# Or Plotly live
dm.live_plot_init(loc=0)
# In a loop: dm.live_plot_update(loc=0, idx=i)
```

---

## Example 4: DataSplitter for Field Sweeps

```python
import pandas as pd
from pyomnix.data_process.data_splitter import DataSplitter

df = pd.read_csv("field_sweep.csv")
splitter = DataSplitter()
segments = splitter.detect_sweep_segments(df, field_col="B", min_points=10)
split_dfs = splitter.split_dataframe_by_segments(df, segments)

for i, seg_df in enumerate(split_dfs):
    direction = segments[i].direction  # "up" or "down"
    print(f"Segment {i}: {direction}, {len(seg_df)} points")
```

---

## Example 5: Custom Model Provider (e.g. SiliconFlow)

```python
# In api_config.json, add:
# "siliconflow": {"api_key": "...", "base_url": "https://api.siliconflow.cn/...", "provider_kwargs": {}}

from pyomnix.agents.models_settings import ModelConfig

config = ModelConfig()
models = config.setup_model_factory("siliconflow-deepseek")
model = models["siliconflow-deepseek"].with_config(llm_model="deepseek-chat")
```

---

## Example 6: Export Conversation to File or Drive

```python
from pathlib import Path
from pyomnix.agents.storage import DriveManager

# Local export
await session.export_history(Path("./exports"))

# Google Drive export
drive = DriveManager()
await drive.initialize()
metadata = await session.export_history(drive, filename="chat_backup.json")
print(metadata["web_link"])
```

---

## Example 7: Self-Correction Graph with Human Review

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from pyomnix.agents.graphs import build_self_correction_graph

workflow = build_self_correction_graph(model)
compiled = workflow.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "critic-1"}}

# First run - pauses at human_review
result = compiled.invoke({"messages": [HumanMessage(content="Write a haiku")]}, config=config)

# Resume with decision
result = compiled.invoke(Command(resume="continue"), config=config)  # or "end"
```

---

## Example 8: Logging with File Rotation

```python
import logging
from pyomnix import setup_logger

logger = setup_logger(
    name="my_app",
    log_level=logging.DEBUG,
    log_file="log/app.log",
    rotation="size",
    max_size=10 * 1024 * 1024,  # 10 MB
    backup_count=5,
)
logger.info("Application started")
```
