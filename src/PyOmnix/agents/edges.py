"""
Edge functions for LangGraph agent workflows.

This module provides edge (conditional routing) functions for LangGraph agents:
- tools_condition: Route to tools node or END based on tool calls
- should_summarize_edge: Route to summarization based on message count
"""

from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from pyomnix.agents.schemas import ConversationState
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


def should_tools_edge(state: ConversationState) -> Literal["tools", "no_tools"]:
    """
    Determine if tools are needed based on the last message.
    """
    messages = state["messages"]
    if not messages:
        return "no_tools"
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            return "tools"
    return "no_tools"


def should_summarize_edge(
    state: ConversationState, config: RunnableConfig
) -> Literal["summarize", "no_summarize"]:
    """
    Determine if summarization is needed based on message count.
    """
    messages = state["messages"]
    # Read parameters from config, default keep 100 messages
    params = config.get("configurable", {})
    max_len = params.get("max_history_messages", 100)

    if len(messages) > max_len:
        logger.debug(
            "Routing to summarize_conversation: %d messages exceed max %d", len(messages), max_len
        )
        return "summarize"

    return "no_summarize"


def should_multimodal_ingest_edge(
    state: ConversationState,
) -> Literal["ingest", "skip_ingest"]:
    """
    Route to multimodal ingest only when at least one modality is present.
    """
    modality_fields = ("images", "audio_files", "video_files", "pdf_files")
    for field in modality_fields:
        value = state.get(field, [])
        if isinstance(value, list) and len(value) > 0:
            return "ingest"
    return "skip_ingest"
