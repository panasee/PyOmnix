"""
Nodes for LangGraph agent workflows.

This module provides node functions for building LangGraph agents including:
- agent_node: Main agent node that invokes the LLM with messages
- tools_node: Tool execution node using LangGraph's prebuilt ToolNode
- call_model: General chat node with conversation summary support
- summarize_conversation: Summarization node for long conversations
"""

from typing import Any, cast
from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import RemoveMessage
from langgraph.prebuilt import ToolNode

from pyomnix.agents.prompts import PROMPTS
from pyomnix.agents.runnables import create_chat_chain, empty_string_placeholder
from pyomnix.agents.schemas import ConversationState
from pyomnix.agents.tools import handle_tool_error
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)

def create_chat_node(chain: Runnable):
    """
    Create an agent node function with the model bound to tools.

    Args:
        chain: The LCEL chain to use for the agent.
    """

    async def chat_node(state: ConversationState, config: RunnableConfig) -> dict[str, Any]:
        response = await chain.ainvoke(state, config=config)
        return {"messages": [response]}

    return chat_node


def create_tools_node(tools: list | None = None) -> ToolNode:
    """
    Create a tools node using LangGraph's prebuilt ToolNode.

    Args:
        tools: List of tools to include in the node. Defaults to DEFAULT_TOOLS.

    Returns:
        ToolNode: A prebuilt tool execution node.
    """
    tools = tools or []
    return ToolNode(tools, handle_tool_errors=handle_tool_error)


def create_summarize_node(
    model: BaseChatModel | _ConfigurableModel,
    temperature: float = 0.3,
    summary_max_length: int = 1000,
):
    """
    Factory function to create a summarize node.
    note the config of summary_node is not the same as the config of other nodes, so we need to specify alone.

    Args:
        model: BaseChatModel or _ConfigurableModel
        temperature: Temperature for the model
        summary_max_length(tokens): Maximum length of the summary
    """
    try:
        model = model.with_config(temperature=temperature)
    except:
        logger.warning("Failed to set temperature for the model")
    try:
        model = model.with_config(max_tokens=summary_max_length)
    except:
        logger.warning("Failed to set summary_max_length for the model")

    model = cast(BaseChatModel, model)

    async def summarize_node(state: ConversationState, config: RunnableConfig) -> dict[str, Any]:
        messages_to_summarize = state["messages"][:-2]  # keep the last 2 messages
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
        if not messages_to_summarize:
            return {"summary": empty_string_placeholder("summary")}

        if state.get("summary") != empty_string_placeholder("summary"):
            system_prompt = PROMPTS.get("summary_update")
        else:
            system_prompt = PROMPTS.get("summary")

        summary_chain = create_chat_chain(model, system_prompt=system_prompt) | StrOutputParser()
        response = await summary_chain.ainvoke(
            {
                "messages": messages_to_summarize,
                "summary": state.get("summary", empty_string_placeholder("summary")),
                "user_profile": state.get("user_profile", empty_string_placeholder("user profile")),
                "structured_memory": state.get(
                    "structured_memory", empty_string_placeholder("structured memory")
                ),
                "current_intent": state.get(
                    "current_intent", empty_string_placeholder("current intent")
                ),
            },
            config=filter_config(config),
        )
        return {"summary": response, "messages": delete_messages}

    return summarize_node


# TODO: reserved for future customization
# def custom_tools_execution_node(state: ConversationState):
#    """
#    Tool execution node implemented by the underlying mechanism.
#    Not using the prebuilt node, manually handle the tool call loop.
#    """
#    messages = state["messages"]
#    last_message = messages[-1]
#
#    # 1. Check if there are actual tool calls
#    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
#        # Theoretically, this should be controlled by the Conditional Edge of the Graph
#        # But here is defensive programming
#        return {"messages": []}
#
#    # 2. Get tool mapping (usually from global or Config)
#    # Assume tools_map is { "tool_name": tool_instance }
#    # Here is for demonstration, assume you already have tools_map
#    # tools_map = {t.name: t for t in tools}
#
#    results = []
#
#    # 3. Iterate and execute tools (LangGraph allows multiple tool_calls in one output)
#    for tool_call in last_message.tool_calls:
#        tool_name = tool_call["name"]
#        tool_args = tool_call["args"]
#        tool_call_id = tool_call["id"]
#
#        try:
#            # --- pre-execution hook (e.g. log) ---
#            print(f"Executing {tool_name} with {tool_args}...")
#
#            # --- Execute tool ---
#            ##TODO: call the tool function directly
#            tool_instance = tools_map[tool_name]
#            response = tool_instance.invoke(tool_args)
#
#        except Exception as e:
#            # Custom error handling
#            response_content = handle_tool_error(e)
#
#        # 4. Build ToolMessage
#        # This is the standard format that LLM can understand "tool execution completed"
#        tool_message = ToolMessage(
#            tool_call_id=tool_call_id,
#            content=response_content,
#            name=tool_name
#        )
#        results.append(tool_message)
#
#    return {"messages": results}
