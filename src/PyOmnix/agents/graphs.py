"""
Reusable graph builders for LangGraph agents.

This module provides factory functions for building various agent graphs:
- build_chat_graph: Chat graph with automatic summarization
- build_tool_agent_graph: Agent graph with tool calling capability
"""

import json as _json
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyomnix.agents.storage import DriveManager

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
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
    create_human_review_node,
    create_named_chat_node,
    create_summarize_node,
)
from pyomnix.agents.prompts import PROMPTS
from pyomnix.agents.runnables import create_chat_chain
from pyomnix.agents.schemas import ConversationState, GraphContext
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


def _extract_text_content(msg) -> str:
    """Extract text content from a message using standard content_blocks if available.

    Falls back to raw ``msg.content`` for messages that lack the v1 property.
    """
    if hasattr(msg, "content_blocks"):
        parts: list[str] = []
        for block in msg.content_blocks:
            block_type = block.get("type", "")
            if block_type == "text":
                parts.append(block.get("text", ""))
            elif block_type == "reasoning":
                parts.append(f"[thinking] {block.get('reasoning', '')}")
        if parts:
            return "\n".join(parts)

    # Fallback for pre-v1 or simple string content
    content = msg.content
    if isinstance(content, str):
        return content
    return str(content)


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

                content = _extract_text_content(msg)

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

    def astream(self, input_data: Any, *, stream_mode: str = "updates"):  # type: ignore[arg-type]
        """wrap astream with configurable stream_mode"""
        return self.graph.astream(input_data, config=self.config, stream_mode=stream_mode)  # type: ignore[arg-type]

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

    async def inspect_state(self) -> None:
        """Pretty-print the full StateSnapshot for debugging.

        Shows: current state values, which node runs next, pending tasks/interrupts,
        checkpoint metadata, and step number.
        """
        snapshot = await self.get_state()
        console = Console()

        # --- Header ---
        console.print(
            Panel(
                f"[bold]Thread:[/bold] {self.thread_id}",
                title="[bold blue]State Inspector[/bold blue]",
                border_style="blue",
            )
        )

        # --- Next nodes ---
        next_nodes = snapshot.next
        if next_nodes:
            console.print(f"  [bold cyan]Next node(s):[/bold cyan] {', '.join(next_nodes)}")
        else:
            console.print("  [bold cyan]Next node(s):[/bold cyan] [dim](graph finished)[/dim]")

        # --- Metadata ---
        meta = snapshot.metadata or {}
        console.print(
            f"  [bold cyan]Step:[/bold cyan] {meta.get('step', '?')}    "
            f"[bold cyan]Written by:[/bold cyan] {meta.get('source', '?')}"
        )

        # --- Pending tasks / interrupts ---
        if snapshot.tasks:
            for task in snapshot.tasks:
                interrupts = getattr(task, "interrupts", None)
                if interrupts:
                    for intr in interrupts:
                        console.print(
                            Panel(
                                str(intr.value),
                                title="[bold red]Pending Interrupt[/bold red]",
                                border_style="red",
                            )
                        )
                else:
                    console.print(f"  [dim]Task: {task}[/dim]")

        # --- State values (excluding messages for brevity) ---
        values = snapshot.values or {}
        state_table = Table(
            title="[bold]State Fields[/bold]",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
        )
        state_table.add_column("Field", style="cyan", width=22)
        state_table.add_column("Value", style="white")

        for key, val in values.items():
            if key == "messages":
                state_table.add_row("messages", f"[dim]{len(val)} message(s)[/dim]")
            else:
                display = str(val)
                if len(display) > 120:
                    display = display[:120] + "..."
                state_table.add_row(key, display)
        console.print(state_table)

        # --- Last 3 messages preview ---
        messages = values.get("messages", [])
        if messages:
            msg_table = Table(
                title=f"[bold]Messages (last {min(3, len(messages))} of {len(messages)})[/bold]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )
            msg_table.add_column("Role", width=18)
            msg_table.add_column("Content (truncated)")

            for msg in messages[-3:]:
                role = getattr(msg, "type", "?")
                name = getattr(msg, "name", None)
                label = f"{role}" + (f" ({name})" if name else "")
                content = _extract_text_content(msg)
                if len(content) > 200:
                    content = content[:200] + "..."
                msg_table.add_row(label, content)
            console.print(msg_table)

    async def get_state_history(self, limit: int = 10) -> list:
        """Get the checkpoint history for this thread (time travel).

        Returns a list of StateSnapshot objects, newest first.
        Each snapshot represents the state at a specific point in the execution.

        Args:
            limit: Maximum number of snapshots to return.
        """
        history = []
        async for snapshot in self.graph.aget_state_history(self.config):
            history.append(snapshot)
            if len(history) >= limit:
                break
        return history

    async def print_state_history(self, limit: int = 10) -> None:
        """Pretty-print the checkpoint history for debugging.

        Shows step number, which node wrote the checkpoint, next node(s),
        and message count at each point.
        """
        history = await self.get_state_history(limit)
        console = Console()
        table = Table(
            title=f"[bold blue]State History (last {len(history)} checkpoints)[/bold blue]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("#", style="bold", width=5)
        table.add_column("Step", width=6)
        table.add_column("Written by", width=16)
        table.add_column("Next node(s)", width=20)
        table.add_column("Messages", width=10)
        table.add_column("Interrupt?", width=10)

        for i, snap in enumerate(history):
            meta = snap.metadata or {}
            next_nodes = ", ".join(snap.next) if snap.next else "(end)"
            msg_count = str(len(snap.values.get("messages", [])))
            has_interrupt = any(
                getattr(t, "interrupts", None) for t in (snap.tasks or [])
            )
            table.add_row(
                str(i),
                str(meta.get("step", "?")),
                str(meta.get("source", "?")),
                next_nodes,
                msg_count,
                "[red]Yes[/red]" if has_interrupt else "[dim]No[/dim]",
            )
        console.print(table)

    async def export_history(
        self,
        target: "Path | DriveManager",
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Export the conversation history to a local file or Google Drive.

        Args:
            target: Either a ``pathlib.Path`` (directory for local save) or a
                ``DriveManager`` instance (uploads to Google Drive).
            filename: Optional filename override. Defaults to
                ``"crosstalk_{thread_id}_{timestamp}.json"``.

        Returns:
            dict with export metadata:
              - For local: ``{"path": "<full_path>", "message_count": N}``
              - For Drive: ``{"file_id": "...", "web_link": "...", ...}``
        """
        snapshot = await self.get_state()
        messages = snapshot.values.get("messages", [])

        # Serialize messages to a list of dicts
        serialized: list[dict[str, Any]] = []
        for msg in messages:
            entry: dict[str, Any] = {
                "role": getattr(msg, "type", "unknown"),
                "content": _extract_text_content(msg),
            }
            name = getattr(msg, "name", None)
            if name:
                entry["name"] = name
            msg_id = getattr(msg, "id", None)
            if msg_id:
                entry["id"] = msg_id
            serialized.append(entry)

        export_data = {
            "thread_id": self.thread_id,
            "exported_at": datetime.now().isoformat(),
            "message_count": len(serialized),
            "state_fields": {
                k: (str(v) if not isinstance(v, (str, int, float, bool, dict, list)) else v)
                for k, v in snapshot.values.items()
                if k != "messages"
            },
            "messages": serialized,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = filename or f"conversation_{self.thread_id}_{ts}.json"
        json_bytes = _json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8")

        # Dispatch based on target type
        # If target is a string, convert to Path
        if isinstance(target, str):
            target = Path(target)
        if isinstance(target, Path):
            # Local file export
            target.mkdir(parents=True, exist_ok=True)
            file_path = target / fname
            file_path.write_bytes(json_bytes)
            logger.info("Exported %d messages to %s", len(serialized), file_path)
            return {"path": str(file_path), "message_count": len(serialized)}
        else:
            # Google Drive export via DriveManager
            from pyomnix.agents.storage import DriveManager

            if not isinstance(target, DriveManager):
                raise TypeError(
                    f"target must be a Path or DriveManager, got {type(target).__name__}"
                )
            await target.initialize()
            metadata = await target.upload_asset(json_bytes, fname)
            logger.info(
                "Exported %d messages to Google Drive: %s",
                len(serialized),
                metadata.get("web_link", metadata.get("file_id", "?")),
            )
            return {**metadata, "message_count": len(serialized)}

    async def astream_pretty(self, input_data: Any):
        """Stream the output with rich formatting using LangGraph's messages stream mode.

        Uses ``stream_mode="messages"`` for true token-by-token streaming.
        Each streamed item is a ``(AIMessageChunk, metadata)`` tuple where
        ``metadata`` contains the originating node name and LLM invocation details.
        """
        console = Console()
        full_content = ""
        last_node = ""

        with Live(console=console, refresh_per_second=12, auto_refresh=False) as live:
            async for msg_chunk, metadata in self.astream(
                input_data, stream_mode="messages"
            ):
                # Extract the node that produced this chunk
                node_name = metadata.get("langgraph_node", "")
                if node_name != last_node and node_name:
                    if last_node:
                        live.console.print()  # blank line between nodes
                    live.console.print(
                        f"[bold magenta]Node:[/bold magenta] [cyan]{node_name}[/cyan]"
                    )
                    last_node = node_name

                # Use standard content_blocks for provider-agnostic parsing
                if hasattr(msg_chunk, "content_blocks"):
                    for block in msg_chunk.content_blocks:
                        block_type = block.get("type", "")
                        if block_type == "text":
                            full_content += block.get("text", "")
                        elif block_type == "reasoning":
                            # Skip reasoning/thinking blocks in display
                            pass
                else:
                    # Fallback: extract text from raw content
                    content = msg_chunk.content
                    if isinstance(content, str):
                        full_content += content

                live.update(
                    Panel(
                        Markdown(full_content),
                        title=f"AI ({node_name}):" if node_name else "AI:",
                        title_align="left",
                        border_style="green",
                    ),
                    refresh=True,
                )

            return full_content


def build_chat_graph(model: BaseChatModel):
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

def build_self_correction_graph(model: BaseChatModel):
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
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        graph = build_self_correction_graph(model)
        compiled = graph.compile(checkpointer=InMemorySaver())

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

    human_review = create_human_review_node(
        destinations=("chat", "__end__"),
        continue_to="chat",
    )

    workflow = StateGraph(ConversationState, GraphContext)
    workflow.add_node("chat", chat_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("human_review", human_review, destinations=human_review.destinations)
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", "critic")
    workflow.add_edge("critic", "human_review")
    return workflow
