"""
Reusable graph builders for LangGraph agents.

This module provides factory functions for building various agent graphs:
- build_chat_graph: Chat graph with automatic summarization
- build_tool_agent_graph: Agent graph with tool calling capability
"""

import uuid
from typing import Any

from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pyomnix.agents.edges import should_summarize_edge
from pyomnix.agents.nodes import create_chat_node, create_summarize_node
from pyomnix.agents.runnables import create_chat_chain
from pyomnix.agents.schemas import ConversationState, GraphContext
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


class GraphSession:
    """
    GraphSession is a wrapper for the graph, it can be used to run the graph and get the history.
    Args:
        graph: the compiled graph to wrap
        thread_id: the thread id to use for the graph
        config_dict: the config to update the graph with, should be a dict, with configurable key
    Returns:
        GraphSession: the GraphSession object
    """

    def __init__(
        self,
        graph: CompiledStateGraph[Any, Any, Any, Any],
        thread_id: str | None = None,
        config_dict: dict[str, Any] | None = None,
    ):
        self.graph = graph
        # if not thread_id, generate a UUID
        self.thread_id = thread_id or str(uuid.uuid4())
        self.configurable: dict = {"thread_id": self.thread_id}
        self.other_config: dict = {}
        if config_dict:
            self.config = config_dict

    @property
    def config(self) -> RunnableConfig:
        """get the config"""
        return RunnableConfig(configurable=self.configurable, **self.other_config)

    @config.setter
    def config(self, value: dict[str, Any]):
        """set the config"""
        if "configurable" in value:
            new_configurable = value.pop("configurable")
            if isinstance(new_configurable, dict):
                if "thread_id" in new_configurable:
                    logger.warning(
                        "thread_id detected, please set thread_id by contructing a new instance"
                    )
                    del new_configurable["thread_id"]
                self.configurable.update(**new_configurable)
            else:
                raise ValueError("configurable value must be a dict")
        self.other_config.update(value)

    def update_config(self, **updates: Any):
        """update the config"""
        self.config = updates

    async def ainvoke(self, input_data: Any):
        """wrap invoke, update config if needed, here config_updates must be a dict, for clarity"""
        return await self.graph.ainvoke(input_data, config=self.config)

    def astream(self, input_data: Any):
        """wrap astream"""
        return self.graph.astream(input_data, config=self.config)

    async def get_history(self):
        """wrap aget_state"""
        snapshot = await self.get_state()
        return snapshot.values.get("messages", [])

    async def get_state(self):
        """wrap aget_state"""
        return await self.graph.aget_state(self.config)

    async def update_state(self, values: dict, as_node: str):
        """wrap aupdate_state"""
        return await self.graph.aupdate_state(self.config, values, as_node=as_node)


def update_dict(old_dict: dict[str, Any], new_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Mechanism: Patch update instead of overwrite.
    Similar to Python's dict.update().
    """
    if not old_dict:
        return new_dict
    return {**old_dict, **new_dict}


def build_chat_graph(model: BaseChatModel | _ConfigurableModel):
    """
    Build a chat graph with automatic memory summarization.

    Args:
        model: The base LLM object (e.g., ChatOpenAI)

    Returns:
        Compiled StateGraph with memory checkpointer.
    """
    chat_chain = create_chat_chain(model)
    chat_node = create_chat_node(chat_chain)
    summarize_node = create_summarize_node(model)

    # Build graph
    workflow = StateGraph(ConversationState, GraphContext)

    workflow.add_node("conversation", chat_node)
    workflow.add_node("summarize", summarize_node)

    workflow.set_entry_point("conversation")

    workflow.add_conditional_edges(
        "conversation",
        should_summarize_edge,
        {
            "summarize": "summarize",
            "no_summarize": END,
        },
    )

    workflow.add_edge("summarize", END)

    return workflow
