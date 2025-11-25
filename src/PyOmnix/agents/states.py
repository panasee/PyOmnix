"""
State definitions for agent blocks.
"""

from typing import Any

from langgraph.graph.message import MessagesState


class ConversationState(MessagesState):
    """
    Standard state for agents.

    Attributes:
        messages: List of messages in the conversation.
        next: The next node to execute (optional).
    """

    topic: str
    summary: str
    private_memories: dict[str, Any]
