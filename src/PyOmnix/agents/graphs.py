"""
Reusable nodes for building agents.
"""

from functools import partial
from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from pyomnix.agents.states import ConversationState
from pyomnix.agents.nodes import call_model, summarize_conversation
from pyomnix.agents.edges import should_summarize_edge

from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


def build_chat_graph(model: BaseChatModel | _ConfigurableModel):
    """
    构建一个带有自动记忆总结功能的对话图。
    
    Args:
        base_llm: 基础 LLM 对象 (例如 ChatOpenAI)
    """
    bound_call_model = partial(call_model, model)
    bound_summarize = partial(summarize_conversation, model)

    # build graph ---
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