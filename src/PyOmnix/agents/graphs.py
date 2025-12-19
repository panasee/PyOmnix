"""
Reusable graph builders for LangGraph agents.

This module provides factory functions for building various agent graphs:
- build_chat_graph: Chat graph with automatic summarization
- build_tool_agent_graph: Agent graph with tool calling capability
"""

from functools import wraps
from typing import Any, Literal

from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, interrupt
from langsmith import uuid7
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from pyomnix.agents.edges import should_summarize_edge
from pyomnix.agents.nodes import (
    create_chat_node,
    create_named_chat_node,
    create_summarize_node,
)
from pyomnix.agents.prompts import PROMPTS
from pyomnix.agents.runnables import create_chat_chain
from pyomnix.agents.schemas import ConversationState, GraphContext
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


def rich_format_history(func):
    """Decorator to print formatted message history using rich."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        messages = await func(*args, **kwargs)
        if messages:
            console = Console()
            table = Table(
                title="[bold blue]Conversation History[/bold blue]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
                expand=True,
            )
            table.add_column("Role", style="bold", width=15)
            table.add_column("Message", style="white")

            for msg in messages:
                role = getattr(msg, "type", "unknown")
                name = getattr(msg, "name", None)
                role_display = role.upper()
                if name:
                    role_display = f"{role_display} ({name})"

                content = msg.content
                if not isinstance(content, str):
                    content = str(content)

                # Assign colors based on role
                color = "cyan"
                if role == "human":
                    color = "green"
                elif role == "ai":
                    color = "magenta"
                elif role == "system":
                    color = "yellow"
                elif role == "tool":
                    color = "blue"

                table.add_row(
                    f"[{color}]{role_display}[/{color}]",
                    Markdown(content)
                    if content
                    else "[italic grey]Empty content[/italic grey]",
                )

            console.print(table)
        return messages

    return wrapper


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
        self.thread_id = thread_id or str(uuid7())
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

    @rich_format_history
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

    async def astream_pretty(self, input_data: Any):
        """stream the output pretty"""
        console = Console()
        full_content = ""
        last_node = ""

        with Live(console=console, refresh_per_second=12, auto_refresh=False) as live:
            async for chunk in self.astream(input_data):
                for node_name, state_update in chunk.items():
                    if node_name != last_node:
                        live.console.print(
                            f"\n[bold magenta]Node:[/bold magenta] [cyan]{node_name}[/cyan]"
                        )
                        last_node = node_name

                    if isinstance(state_update, dict):
                        messages = state_update.get("messages", [])
                        if not messages:
                            continue
                    elif isinstance(state_update, tuple) and isinstance(state_update[0], Interrupt):
                        messages = [state_update[0]]
                    else:
                        continue

                    for msg in messages:
                        if isinstance(msg, AIMessageChunk | BaseMessage):
                            content = msg.content
                            if isinstance(content, str):
                                full_content += content
                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, str):
                                        full_content += block
                                    elif (
                                        isinstance(block, dict)
                                        and block.get("type") == "text"
                                    ):
                                        # content is a dict with "type" and "text"
                                        full_content += block.get("text", "")
                            live.update(
                                Panel(
                                    Markdown(full_content),
                                    title="AI:",
                                    title_align="left",
                                    border_style="green",
                                ),
                                refresh=True,
                            )
                        elif isinstance(msg, Interrupt):
                            interrupt_data = msg.value
                            live.update(
                                Panel(
                                    str(interrupt_data["message"]),
                                    title="Interrupt:",
                                    title_align="left",
                                    border_style="red",
                                ),
                                refresh=True,
                            )
            return full_content


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

def build_self_correction_graph(model: BaseChatModel | _ConfigurableModel):
    """
    Build a self-correction graph where chat and critic nodes loop infinitely.

    The graph flow is: chat → critic → human_review → (continue: chat, end: END)

    After each critic response, execution pauses for human intervention.
    Human can decide to:
    - "continue": Continue the loop (go back to chat node)
    - "end": End the conversation

    Args:
        model: The base LLM object (e.g., ChatOpenAI)

    Returns:
        StateGraph (not compiled). Compile with a checkpointer to enable
        human-in-the-loop interrupts.

    Usage:
        ```python
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command

        graph = build_self_correction_graph(model)
        compiled = graph.compile(checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "my-thread"}}

        # First invocation - runs until interrupt
        result = compiled.invoke({"messages": [("user", "Hello")]}, config=config)
        # result["__interrupt__"] contains the critic's response

        # Resume with decision to continue or end
        result = compiled.invoke(
            Command(resume="continue"),  # or "end"
            config=config
        )
        ```
    """
    chat_chain = create_chat_chain(model)
    critic_chain = create_chat_chain(model, system_prompt=PROMPTS.get("critic"))
    critic_node = create_named_chat_node("critic", critic_chain)
    chat_node = create_named_chat_node("chat", chat_chain)

    workflow = StateGraph(ConversationState, GraphContext)
    workflow.add_node("chat", chat_node)
    workflow.add_node("critic", critic_node)
    async def human_review_node(
        state: ConversationState, config: RunnableConfig
    ) -> Command[Literal["chat", "__end__"]]:
        """
        Human review node that pauses execution after critic response.

        Uses interrupt() to pause and wait for human decision.
        Returns a Command to route to the next node based on human input.
        """
        # Get the last message (critic's response) to show to human
        last_message = state["messages"][-1] if state["messages"] else None
        critic_response = last_message.content if last_message else "No response"

        # Interrupt and wait for human decision
        # The human should respond with "continue" or "end"
        human_decision = interrupt({
            "message": "Critic has provided feedback. Continue the loop or end?",
            "critic_response": critic_response,
            "options": ["continue", "end"],
        })

        # Route based on human decision
        if human_decision == "continue":
            return Command(goto="chat")
        else:
            # Any other response (including "end") terminates the loop
            return Command(goto="__end__")

    workflow.add_node("human_review", human_review_node,destinations=("chat", "__end__"))
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", "critic")
    workflow.add_edge("critic", "human_review")
    return workflow