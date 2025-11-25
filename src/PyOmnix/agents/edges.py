from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from typing import Literal
from pyomnix.agents.states import ConversationState
def should_summarize_edge(state: ConversationState, config: RunnableConfig) -> Literal["summarize_conversation", END]:
    """
    edge to determine if summarization is needed
    Args:
        state: ConversationState
        config: RunnableConfig
    Returns:
        Literal["summarize_conversation", END]
    """
    messages = state["messages"]
    # read parameters from config, default keep 6 messages
    params = config.get("configurable", {})
    max_len = params.get("max_history_messages", 100)

    if len(messages) > max_len:
        return "summarize_conversation"
    return END