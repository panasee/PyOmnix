"""
Schemas for the agents.
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import MessagesState


def update_dict(old_dict: dict[str, Any], new_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Mechanism: Patch update instead of overwrite.
    Similar to Python's dict.update().
    """
    if not old_dict:
        return new_dict
    return {**old_dict, **new_dict}


class GraphContext(TypedDict):
    """
    Settings and Context for the graph.
    """

    max_history_messages: int


class ConversationState(MessagesState):
    """
    Standard state for agents.

    Attributes:
        messages: List of messages in the conversation.
        next: The next node to execute (optional).
    """

    # Summary of early conversation
    summary: str

    # User profile, almost read-only and write-only, only updated when user profile is changed
    user_profile: str

    # Store intermediate variables, e.g. {"current_ticker": "AAPL", "risk_free_rate": 0.04}
    structured_memory: Annotated[dict[str, Any], update_dict]
    retrieved_docs: list[str]

    # Conversation turn counter, used to trigger Summary Node or force termination
    dialogue_turn_count: int

    # Current intent/task (alternative to fuzzy topic)
    current_intent: str


class CanvasState(ConversationState):
    """
    State for the canvas agent.
    """

    canvas_title: str
    canvas_content: str
    canvas_language: str
    is_canvas_active: bool
