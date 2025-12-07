"""
Reusable graph builders for LangGraph agents.

This module provides factory functions for building various agent graphs:
- build_chat_graph: Chat graph with automatic summarization
- build_tool_agent_graph: Agent graph with tool calling capability
"""

import asyncio
from functools import partial

from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from pyomnix.agents.edges import should_summarize_edge, tools_condition
from pyomnix.agents.models_settings import ModelConfig
from pyomnix.agents.nodes import (
    create_agent_node,
    create_tools_node,
    summarize_conversation,
)
from pyomnix.agents.tools import TEST_TOOLS

DEFAULT_TOOLS = TEST_TOOLS
from pyomnix.agents.schemas import ConversationState
from pyomnix.agents.storage import get_checkpointer
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Chat Graph (with Summarization)
# =============================================================================


def build_chat_graph(model: BaseChatModel | _ConfigurableModel):
    """
    Build a chat graph with automatic memory summarization.

    Args:
        model: The base LLM object (e.g., ChatOpenAI)

    Returns:
        Compiled StateGraph with memory checkpointer.
    """
    bound_call_model = partial(call_model, model)
    bound_summarize = partial(summarize_conversation, model)

    # Build graph
    workflow = StateGraph(ConversationState)

    workflow.add_node("conversation", bound_call_model)
    workflow.add_node("summarize_conversation", bound_summarize)

    workflow.set_entry_point("conversation")

    workflow.add_conditional_edges(
        "conversation",
        should_summarize_edge,
    )

    workflow.add_edge("summarize_conversation", END)

    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)


# =============================================================================
# Tool Agent Graph
# =============================================================================


def build_tool_agent_graph(
    model: BaseChatModel | _ConfigurableModel,
    tools: list | None = None,
) -> StateGraph:
    """
    Build a StateGraph for a tool-calling agent.

    This creates an uncompiled graph. Use compile_tool_agent_graph() or
    compile_tool_agent_graph_async() to compile with a checkpointer.

    Args:
        model: The chat model to use.
        tools: List of tools to bind to the agent. Defaults to DEFAULT_TOOLS.

    Returns:
        StateGraph: The uncompiled workflow graph.
    """
    tools = tools or DEFAULT_TOOLS

    # Create nodes
    agent_node = create_agent_node(model, tools)
    tools_node = create_tools_node(tools)

    # Initialize StateGraph
    workflow = StateGraph(ConversationState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add edges
    workflow.add_edge("tools", "agent")  # After tool runs, go back to agent

    # Conditional edge: from "agent", decide to go to "tools" or END
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    return workflow


def compile_tool_agent_graph_sync(
    model: BaseChatModel | _ConfigurableModel,
    tools: list | None = None,
):
    """
    Build and compile a tool agent graph with in-memory checkpointer.

    For development/testing. Use async version with PostgreSQL for production.

    Args:
        model: The chat model to use.
        tools: List of tools to bind to the agent.

    Returns:
        Compiled graph with MemorySaver checkpointer.
    """
    workflow = build_tool_agent_graph(model, tools)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


async def compile_tool_agent_graph_async(
    model: BaseChatModel | _ConfigurableModel,
    tools: list | None = None,
):
    """
    Build and compile a tool agent graph with async PostgreSQL checkpointer.

    This function returns an async context manager that yields the compiled graph.

    Args:
        model: The chat model to use.
        tools: List of tools to bind to the agent.

    Yields:
        Compiled graph with AsyncPostgresSaver checkpointer.

    Usage:
        async with compile_tool_agent_graph_async(model) as graph:
            result = await graph.ainvoke(state, config)
    """
    workflow = build_tool_agent_graph(model, tools)

    async with get_checkpointer() as checkpointer:
        compiled_graph = workflow.compile(checkpointer=checkpointer)
        yield compiled_graph


# =============================================================================
# Main Execution Block
# =============================================================================


async def run_tool_agent_demo():
    """
    Run a simple demo of the tool-calling agent.

    This function demonstrates the agent's ability to:
    1. Receive a user message
    2. Detect that a tool is needed
    3. Call the get_current_time tool
    4. Return the final answer
    """
    # Setup model using ModelConfig
    model_factory = ModelConfig()
    models = model_factory.setup_model_factory("deepseek")
    model = models["deepseek"].with_config(model="deepseek-chat", temperature=0.7)

    logger.info("Starting tool agent demo...")

    # Build the graph
    workflow = build_tool_agent_graph(model)

    # Use async checkpointer for production-ready persistence
    async with get_checkpointer() as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)

        # Create config with thread_id for conversation persistence
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "test_thread_1",
            }
        }

        # Input message
        input_state: dict = {
            "messages": [{"role": "user", "content": "What time is it?"}],
            "topic": "",
            "runtime_context": "",
            "user_profile": "",
            "private_memories": {},
            "canvas_title": "",
            "canvas_content": "",
            "canvas_language": "",
            "is_canvas_active": False,
        }

        logger.info("Sending message: 'What time is it?'")

        # Stream the output
        async for event in graph.astream(input_state, config=config):  # type: ignore[arg-type]
            for node_name, node_output in event.items():
                logger.info("Node '%s' output:", node_name)
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, "content"):
                            logger.info("  Content: %s", msg.content)
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            logger.info("  Tool calls: %s", msg.tool_calls)

        logger.info("Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_tool_agent_demo())
