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
from typing import Any

from langchain_core.messages import HumanMessage

from pyomnix.agents.graphs import GraphSession, build_chat_graph
from pyomnix.agents.models_settings import ModelConfig
from pyomnix.agents.storage import get_checkpointer
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)
# Thread ID for conversation persistence
# Use a fixed ID to test memory persistence across runs
PERSISTENT_THREAD_ID = "react-demo-thread-001"


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


async def run_react_agent(thread_id: str, summary_threshold: int = 10):
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
    model = models["deepseek"].with_config(llm_model="deepseek-chat", llm_temperature=0.7)

    # Build graph
    logger.info("Building ReAct graph...")
    workflow = build_chat_graph(model)

    # Use PostgreSQL checkpointer for persistence
    logger.info("Connecting to PostgreSQL for persistence...")
    async with get_checkpointer() as checkpointer:
        # Compile graph with checkpointer
        graph = workflow.compile(checkpointer=checkpointer)
        graph_session = GraphSession(
            graph,
            thread_id=thread_id,
            config_dict={"configurable": {"max_history_messages": summary_threshold}},
        )

        # Check if we have existing conversation history
        existing_state = await graph_session.get_state()
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

            # Create state for new message
            input_state = create_continuation_state(user_input)

            print("\n🤖 Agent: ", end="", flush=True)

            # Stream the response
            try:
                async for event in graph_session.astream(input_state):
                    for node_name, node_output in event.items():
                        print(f"=====Node: {node_name}======")
                        for msg in node_output["messages"]:
                            # Check for tool calls
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                tool_names = [tc["name"] for tc in msg.tool_calls]
                                print(f"\n   🔧 Calling tools: {tool_names}")
                            # Print final content
                            elif hasattr(msg, "content") and msg.content:
                                print(f"   📋Result: {msg.content}")

            except Exception as e:
                logger.error("Error during agent execution: %s", e)
                print(f"\n❌ Error: {e}")
                continue


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    model_factory = ModelConfig()
    model_factory.setup_langsmith()
    models = model_factory.setup_model_factory("deepseek")
    model = models["deepseek"].with_config(llm_model="deepseek-chat", llm_temperature=0.7)
    workflow = build_chat_graph(model)
    graph = workflow.compile()
    graph_session = GraphSession(graph, thread_id="test_react_agent", config_dict={"configurable": {"max_history_messages": 3}})
    print(asyncio.run(graph_session.ainvoke({"messages": [HumanMessage(content="Give me your brief model specification")]})))