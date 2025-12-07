"""
Minimal ReAct Loop Demo with PostgreSQL Persistence.

This script demonstrates:
1. A minimal ReAct agent with Agent <-> Tools loop
2. Async execution with PostgreSQL checkpointer for persistence
3. Conversation memory that survives script restarts

Usage:
    # First run - start a conversation
    python -m pyomnix.agents.main_react

    # Second run - the agent remembers your previous conversation!
    python -m pyomnix.agents.main_react
"""

import asyncio
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from pyomnix.agents.edges import should_tools_edge
from pyomnix.agents.models_settings import ModelConfig
from pyomnix.agents.nodes import create_chat_node, create_tools_node
from pyomnix.agents.schemas import ConversationState, GraphContext
from pyomnix.agents.storage import get_checkpointer
from pyomnix.agents.tools import dummy_weather_tool
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)
# Thread ID for conversation persistence
# Use a fixed ID to test memory persistence across runs
PERSISTENT_THREAD_ID = "react-demo-thread-001"


# ============================================================================
# Graph Building
# ============================================================================


def build_react_graph(model):
    """
    Build a minimal ReAct graph: Agent <-> Tools.

    The graph structure:
        [START] -> chat -> (tools_condition) -> tools -> chat -> ... -> [END]

    Args:
        model: The chat model with tools bound.

    Returns:
        StateGraph: The uncompiled workflow graph.
    """
    # Create nodes
    chat_node = create_chat_node(model)
    tools_node = create_tools_node([dummy_weather_tool])

    # Initialize StateGraph with minimal state
    workflow = StateGraph(state_schema=ConversationState, context_schema=GraphContext)

    # Add nodes
    workflow.add_node("chat", chat_node)
    workflow.add_node("tools", tools_node)

    # Set entry point
    workflow.set_entry_point("chat")

    # Add edges: tools -> chat (loop back)
    workflow.add_edge("tools", "chat")

    # Conditional edge: chat -> tools OR END
    workflow.add_conditional_edges(
        "chat",
        should_tools_edge,
        {
            "tools": "tools",
            "no_tools": END,
        },
    )

    return workflow


def create_initial_state(user_message: str) -> dict[str, Any]:
    """
    Create an initial state for the conversation.

    Args:
        user_message: The user's input message.

    Returns:
        dict: Initial state with the user message.
    """
    return {
        "messages": [HumanMessage(content=user_message)],
        "topic": "",
        "runtime_context": "",
        "user_profile": "",
        "private_memories": {},
    }


def create_continuation_state(user_message: str) -> dict[str, Any]:
    """
    Create a state for continuing an existing conversation.

    Args:
        user_message: The user's new input message.

    Returns:
        dict: State with just the new message (history loaded from checkpoint).
    """
    return {
        "messages": [HumanMessage(content=user_message)],
    }


async def run_react_agent():
    """
    Run the ReAct agent with interactive conversation and persistence.

    This function:
    1. Sets up the model and graph
    2. Connects to PostgreSQL for checkpoint storage
    3. Runs an interactive conversation loop
    4. Persists state to PostgreSQL for memory across restarts
    """
    # Setup model
    logger.info("Initializing model...")
    model_factory = ModelConfig()
    models = model_factory.setup_model_factory("deepseek")
    model = models["deepseek"].with_config(model="deepseek-chat", temperature=0.7)

    # Build graph
    logger.info("Building ReAct graph...")
    workflow = build_react_graph(model)

    # Use PostgreSQL checkpointer for persistence
    logger.info("Connecting to PostgreSQL for persistence...")
    async with get_checkpointer() as checkpointer:
        # Compile graph with checkpointer
        graph = workflow.compile(checkpointer=checkpointer)

        # Config with persistent thread_id
        config: RunnableConfig = {
            "configurable": {
                "thread_id": PERSISTENT_THREAD_ID,
            }
        }

        # Check if we have existing conversation history
        existing_state = await graph.aget_state(config)
        if existing_state.values and existing_state.values.get("messages"):
            msg_count = len(existing_state.values["messages"])
            logger.info(
                "Found existing conversation with %d messages. Memory restored!",
                msg_count,
            )
            print(f"\n✅ Memory restored! Found {msg_count} messages from previous session.")
            print("=" * 60)
            print("Previous conversation:")
            for msg in existing_state.values["messages"][-6:]:  # Show last 6 messages
                role = msg.__class__.__name__.replace("Message", "")
                content = msg.content if hasattr(msg, "content") else str(msg)
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"  [{role}]: {content}")
            print("=" * 60)
        else:
            logger.info("Starting fresh conversation.")
            print("\n🆕 Starting new conversation (no previous history found).")

        print("\n🤖 ReAct Agent Ready!")
        print("Available tools: dummy_weather_tool (try asking about weather)")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("Type 'clear' to start a new conversation thread.")
        print("-" * 60)

        # Interactive loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
            except EOFError:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("\n💾 Conversation saved. Goodbye!")
                logger.info("Conversation ended by user. State persisted.")
                break

            if user_input.lower() == "clear":
                # Create new thread_id to start fresh
                new_thread_id = f"react-demo-{uuid.uuid4().hex[:8]}"
                config["configurable"]["thread_id"] = new_thread_id
                print(f"\n🧹 Started new conversation thread: {new_thread_id}")
                continue

            # Create state for new message
            input_state = create_continuation_state(user_input)

            print("\n🤖 Agent: ", end="", flush=True)

            # Stream the response
            try:
                async for event in graph.astream(input_state, config=config):
                    for node_name, node_output in event.items():
                        if node_name == "agent" and "messages" in node_output:
                            for msg in node_output["messages"]:
                                # Check for tool calls
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    tool_names = [tc["name"] for tc in msg.tool_calls]
                                    print(f"\n   🔧 Calling tools: {tool_names}")
                                # Print final content
                                elif hasattr(msg, "content") and msg.content:
                                    print(msg.content)
                        elif node_name == "tools" and "messages" in node_output:
                            for msg in node_output["messages"]:
                                tool_result = msg.content if hasattr(msg, "content") else str(msg)
                                if len(tool_result) > 100:
                                    tool_result = tool_result[:100] + "..."
                                print(f"   📋 Tool result: {tool_result}")

            except Exception as e:
                logger.error("Error during agent execution: %s", e)
                print(f"\n❌ Error: {e}")
                continue


async def run_single_query(query: str):
    """
    Run a single query through the ReAct agent (for testing).

    Args:
        query: The query to send to the agent.
    """
    # Setup model
    model_factory = ModelConfig()
    models = model_factory.setup_model_factory("deepseek")
    model = models["deepseek"].with_config(model="deepseek-chat", temperature=0.7)

    # Build and compile graph
    workflow = build_react_graph(model)

    async with get_checkpointer() as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": f"test-{uuid.uuid4().hex[:8]}",
            }
        }

        input_state = create_initial_state(query)

        print(f"\n📤 Query: {query}")
        print("-" * 40)

        async for event in graph.astream(input_state, config=config):
            for node_name, node_output in event.items():
                print(f"[{node_name}]")
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            print(f"  Tool calls: {msg.tool_calls}")
                        if hasattr(msg, "content") and msg.content:
                            print(f"  Content: {msg.content}")


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Minimal ReAct Loop with PostgreSQL Persistence")
    print("=" * 60)

    asyncio.run(run_react_agent())
