"""
Nodes for LangGraph agent workflows.

This module provides node functions for building LangGraph agents including:
- agent_node: Main agent node that invokes the LLM with messages
- tools_node: Tool execution node using LangGraph's prebuilt ToolNode
- call_model: General chat node with conversation summary support
- summarize_conversation: Summarization node for long conversations
"""

import base64
import mimetypes
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import RemoveMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from pyomnix.agents.prompts import PROMPTS
from pyomnix.agents.runnables import create_chat_chain, empty_string_placeholder
from pyomnix.agents.schemas import ConversationState
from pyomnix.agents.tools import google_search, handle_tool_error, tavily_search
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


def _empty_multimodal_payload() -> dict[str, list[str]]:
    """Default empty multimodal fields for text-only compatibility."""
    return {
        "images": [],
        "audio_files": [],
        "video_files": [],
        "pdf_files": [],
        "image_context": [],
        "audio_transcripts": [],
        "video_context": [],
        "pdf_context": [],
    }


def _normalize_multimodal_fields(state: ConversationState) -> dict[str, list[str]]:
    """Ensure multimodal keys always exist and are list[str]-like."""
    defaults = _empty_multimodal_payload()
    normalized: dict[str, list[str]] = {}
    for key, fallback in defaults.items():
        value = state.get(key, fallback)
        if isinstance(value, list):
            normalized[key] = [str(v) for v in value]
        elif value is None:
            normalized[key] = []
        else:
            normalized[key] = [str(value)]
    return normalized


def _ensure_message_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalize message content into block format."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                blocks.append(item)
            elif isinstance(item, str):
                blocks.append({"type": "text", "text": item})
            else:
                blocks.append({"type": "text", "text": str(item)})
        return blocks
    return [{"type": "text", "text": str(content)}]


def _inject_multimodal_into_messages(
    messages: list[BaseMessage],
    model_parts: list[dict[str, Any]],
    notes: list[str],
) -> list[BaseMessage]:
    """Attach multimodal parts to the latest human message."""
    if not model_parts and not notes:
        return messages

    notes_blocks = [{"type": "text", "text": f"[Attachment] {n}"} for n in notes]
    combined_parts = model_parts + notes_blocks
    patched = list(messages)
    target_idx = next(
        (idx for idx in range(len(patched) - 1, -1, -1) if isinstance(patched[idx], HumanMessage)),
        None,
    )

    if target_idx is None:
        patched.append(
            HumanMessage(
                content=[{"type": "text", "text": "Multimodal attachments included."}]
                + combined_parts
            )
        )
        return patched

    target = patched[target_idx]
    blocks = _ensure_message_blocks(target.content) + combined_parts
    patched[target_idx] = HumanMessage(content=blocks)
    return patched


def _prepare_state_for_model(state: ConversationState) -> dict[str, Any]:
    """Create a model-input state with optional multimodal payload injection."""
    messages = list(state.get("messages", []))
    model_parts = state.get("multimodal_model_parts", [])
    notes = state.get("multimodal_notes", [])

    if not isinstance(model_parts, list):
        model_parts = []
    if not isinstance(notes, list):
        notes = []

    prepared = dict(state)
    prepared["messages"] = _inject_multimodal_into_messages(messages, model_parts, notes)
    return prepared


def _to_data_url(mime_type: str, payload_base64: str) -> str:
    return f"data:{mime_type};base64,{payload_base64}"


def _audio_format_from_mime_or_path(mime_type: str, path: Path) -> str:
    if "/" in mime_type:
        return mime_type.split("/", 1)[1].lower()
    return path.suffix.replace(".", "").lower() or "wav"


def _transport_entry_to_model_parts(entry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert internal transport entries into message content blocks."""
    modality = entry.get("modality", "")
    filename = entry.get("filename", "file")
    mime_type = entry.get("mime_type", "application/octet-stream")
    payload_base64 = entry.get("data_base64")
    remote_url = entry.get("remote_url")
    path = Path(str(entry.get("path", filename)))
    notes: list[str] = []

    if not payload_base64 and not remote_url:
        notes.append(f"{filename}: no transferable payload generated.")
        return [], notes

    if modality == "images":
        if payload_base64:
            return (
                [{"type": "image_url", "image_url": {"url": _to_data_url(mime_type, payload_base64)}}],
                notes,
            )
        notes.append(f"{filename}: image payload is remote ({remote_url}).")
        return [], notes

    if modality == "audio_files":
        if payload_base64:
            return (
                [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": payload_base64,
                            "format": _audio_format_from_mime_or_path(mime_type, path),
                        },
                    }
                ],
                notes,
            )
        notes.append(f"{filename}: audio payload is remote ({remote_url}).")
        return [], notes

    # Generic file block for video/pdf and other binary attachments.
    file_data = _to_data_url(mime_type, payload_base64) if payload_base64 else remote_url
    if file_data:
        return (
            [
                {
                    "type": "input_file",
                    "filename": filename,
                    "file_data": file_data,
                }
            ],
            notes,
        )
    return [], notes


async def _build_transport_entry(
    path_str: str,
    modality: str,
    inline_max_bytes: int,
    use_gdrive_for_large: bool,
    gdrive_uploader: Callable[[bytes, str], Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Build transferable payload for one media file."""
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        return None, f"{modality}: file not found: {path}"
    if not path.is_file():
        return None, f"{modality}: not a file: {path}"

    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    payload = path.read_bytes()
    entry: dict[str, Any] = {
        "modality": modality,
        "path": str(path),
        "filename": path.name,
        "mime_type": mime_type,
        "size_bytes": len(payload),
    }

    if len(payload) <= inline_max_bytes:
        entry["data_base64"] = base64.b64encode(payload).decode("ascii")
        return entry, None

    if use_gdrive_for_large and gdrive_uploader is not None:
        uploaded = gdrive_uploader(payload, path.name)
        if hasattr(uploaded, "__await__"):
            uploaded = await uploaded
        if isinstance(uploaded, dict):
            entry["remote_url"] = uploaded.get("web_link")
            entry["remote_file_id"] = uploaded.get("file_id")
        elif isinstance(uploaded, str):
            entry["remote_url"] = uploaded
        else:
            return None, f"{path.name}: gdrive uploader returned unsupported payload."
        return entry, None

    return None, (
        f"{path.name}: file too large for inline transfer ({len(payload)} bytes). "
        "Enable gdrive upload if remote transfer is needed."
    )


def _default_multimodal_preprocessor_placeholder(
    state: ConversationState,
) -> dict[str, Any]:
    """Placeholder for extraction mode.

    TODO:
    - Implement OCR for images
    - Implement ASR for audio
    - Implement keyframe+audio processing for video
    - Implement page-wise extraction for PDF
    """
    pass
    return {}


def _build_public_gdrive_uploader_from_settings() -> Callable[[bytes, str], Any]:
    """Build default uploader for model-share assets (public folder only)."""
    from pyomnix.agents.models_settings import get_settings
    from pyomnix.agents.storage import DriveManager

    settings = get_settings()
    public_folder_id = settings.gdrive_folder_id_public
    if not public_folder_id:
        raise ValueError(
            "gdrive_folder_id_public is not configured. "
            "Set it in api_config.json for multimodal model-sharing uploads."
        )

    drive_manager: DriveManager | None = None

    async def uploader(data: bytes, filename: str) -> dict[str, Any]:
        nonlocal drive_manager
        if drive_manager is None:
            drive_manager = DriveManager(settings=settings, folder_id=public_folder_id)
            await drive_manager.initialize()
        return await drive_manager.upload_asset(data, filename, folder_id=public_folder_id)

    return uploader


def create_multimodal_ingest_node(
    *,
    processing_mode: str = "passthrough",
    preprocessor: Callable[[ConversationState], dict[str, Any]] | None = None,
    inline_max_bytes: int = 5_000_000,
    use_gdrive_for_large: bool = False,
    gdrive_uploader: Callable[[bytes, str], Any] | None = None,
):
    """Create a multimodal ingest node.

    Behavior:
    1. Always normalizes multimodal fields so text-only requests remain compatible.
    2. "passthrough": no extraction, only convert files into transferable payloads.
    3. "preprocess": run custom extractor (placeholder by default, TODO).
    """
    valid_modes = {"passthrough", "preprocess"}
    if processing_mode not in valid_modes:
        raise ValueError(f"processing_mode must be one of {sorted(valid_modes)}")

    resolved_uploader = gdrive_uploader
    if use_gdrive_for_large and resolved_uploader is None:
        resolved_uploader = _build_public_gdrive_uploader_from_settings()

    async def multimodal_ingest_node(
        state: ConversationState, config: RunnableConfig
    ) -> dict[str, Any]:
        updates: dict[str, Any] = _normalize_multimodal_fields(state)
        transport_entries: list[dict[str, Any]] = []
        model_parts: list[dict[str, Any]] = []
        notes: list[str] = []

        for field in ("images", "audio_files", "video_files", "pdf_files"):
            for path_str in updates.get(field, []):
                entry, error = await _build_transport_entry(
                    path_str=path_str,
                    modality=field,
                    inline_max_bytes=inline_max_bytes,
                    use_gdrive_for_large=use_gdrive_for_large,
                    gdrive_uploader=resolved_uploader,
                )
                if error:
                    notes.append(error)
                    continue
                if entry is None:
                    continue
                transport_entries.append(entry)
                parts, extra_notes = _transport_entry_to_model_parts(entry)
                model_parts.extend(parts)
                notes.extend(extra_notes)

        if processing_mode == "preprocess":
            extractor = preprocessor or _default_multimodal_preprocessor_placeholder
            extra_updates = extractor(state)
            if isinstance(extra_updates, dict):
                updates.update(extra_updates)
            if preprocessor is None:
                notes.append("Preprocess mode selected, extractor placeholder is active (TODO).")

        updates.update(
            {
                "multimodal_processing_mode": processing_mode,
                "multimodal_transport": transport_entries,
                "multimodal_model_parts": model_parts,
                "multimodal_notes": notes,
            }
        )
        return updates

    return multimodal_ingest_node


def create_chat_node(chain: Runnable):
    """
    Create an agent node function with the model bound to tools.

    Args:
        chain: The LCEL chain to use for the agent.
    """

    async def chat_node(state: ConversationState, config: RunnableConfig) -> dict[str, Any]:
        prepared_state = _prepare_state_for_model(state)
        response = await chain.ainvoke(prepared_state, config=config)
        return {"messages": [response]}

    return chat_node


def create_named_chat_node(node_name: str, chain: Runnable):
    """
    Create a debater node function with the model bound to tools.

    Args:
        node_name: The name of the node.
        chain: The LCEL chain to use for the debater.
    """

    async def named_node(state: ConversationState, config: RunnableConfig) -> dict[str, Any]:
        prepared_state = _prepare_state_for_model(state)
        response = await chain.ainvoke(prepared_state, config=config)
        labeled_message = AIMessage(content=response.content, name=node_name)
        return {"messages": [labeled_message]}

    return named_node


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


def _normalize_tools(tools: Sequence[Any]) -> list[Any]:
    """Validate and normalize tools for ToolNode creation."""
    if not tools:
        raise ValueError("`tools` must not be empty.")

    normalized = list(tools)
    tool_names = [getattr(t, "name", None) for t in normalized]
    if any(not name for name in tool_names):
        raise ValueError("Each tool must expose a non-empty `name` attribute.")
    if len(set(tool_names)) != len(tool_names):
        raise ValueError(f"Duplicate tool names detected: {tool_names}")
    return normalized


def create_standard_tools_node(tools: Sequence[Any]) -> ToolNode:
    """Recommended factory for creating extensible ToolNode instances.

    This helper centralizes:
    1. Input normalization and validation
    2. Unified tool-error handling behavior
    """
    normalized_tools = _normalize_tools(tools)
    return create_tools_node(normalized_tools)


def create_tavily_search_tools_node() -> ToolNode:
    """Reference ToolNode example with a single Tavily search tool.

    Keep this function as the baseline style when adding new dedicated
    tool nodes in the future.
    """
    return create_standard_tools_node([tavily_search])

def create_summarize_node(
    model: BaseChatModel,
    temperature: float = 0.3,
    summary_max_length: int = 1000,
):
    """
    Factory function to create a summarize node.
    note the config of summary_node is not the same as the config of other nodes, so we need to specify alone.

    Args:
        model: BaseChatModel instance (or configurable model)
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
            config=config,
        )
        return {"summary": response, "messages": delete_messages}

    return summarize_node


# ---------------------------------------------------------------------------
# Human-in-the-loop
# ---------------------------------------------------------------------------


def _default_interrupt_builder(state: ConversationState) -> dict[str, Any]:
    """Default interrupt payload: show last message and offer continue/end."""
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    content = getattr(last_msg, "content", "") if last_msg else ""
    name = getattr(last_msg, "name", None)
    return {
        "message": "Review the last response. Continue or end?",
        "last_speaker": name or getattr(last_msg, "type", "unknown"),
        "last_response": content[:300] if content else "",
        "options": ["continue", "end"],
    }


def _default_resume_router(
    continue_to: str,
) -> Callable[[str, ConversationState], str]:
    """Return a simple router: 'continue' -> *continue_to*, else -> __end__."""

    def router(decision: str, state: ConversationState) -> str:
        if decision == "continue":
            return continue_to
        return "__end__"

    return router


def create_human_review_node(
    *,
    destinations: tuple[str, ...],
    resume_router: Callable[[str, ConversationState], str] | None = None,
    interrupt_builder: Callable[[ConversationState], dict[str, Any]] | None = None,
    continue_to: str | None = None,
):
    """Create a reusable human-in-the-loop review node.

    The returned async function can be passed directly to
    ``StateGraph.add_node()``.  It calls ``interrupt()`` to pause
    execution and waits for a human decision string (via
    ``Command(resume=...)``), then routes to the next node.

    Two usage styles:

    **Simple** -- provide ``continue_to`` for the common
    "continue -> X, else -> __end__" pattern::

        node = create_human_review_node(
            destinations=("chat", "__end__"),
            continue_to="chat",
        )
        workflow.add_node("human_review", node, destinations=node.destinations)

    **Custom** -- provide ``resume_router`` and optionally
    ``interrupt_builder`` for full control::

        node = create_human_review_node(
            destinations=("dougen", "penggen", "__end__"),
            resume_router=my_router,      # (decision, state) -> node_name
            interrupt_builder=my_builder,  # (state) -> interrupt payload dict
        )
        workflow.add_node("human_review", node, destinations=node.destinations)

    Args:
        destinations: All possible destination node names.  Store on the
            returned function as ``node.destinations`` so callers can
            forward it to ``add_node()``.
        resume_router: ``(decision_str, state) -> node_name``.
            If *None*, a default router is built from *continue_to*.
        interrupt_builder: ``(state) -> dict`` payload for ``interrupt()``.
            If *None*, a default is used that shows the last message and
            offers ``["continue", "end"]``.
        continue_to: Shorthand for a simple router where ``"continue"``
            maps to this node and anything else maps to ``"__end__"``.
            Ignored when *resume_router* is provided.

    Returns:
        An async node function with an attached ``.destinations`` attribute.

    Raises:
        ValueError: If neither *resume_router* nor *continue_to* is given.
    """
    if resume_router is None:
        if continue_to is None:
            raise ValueError(
                "Provide either `resume_router` or `continue_to`."
            )
        resume_router = _default_resume_router(continue_to)

    if interrupt_builder is None:
        interrupt_builder = _default_interrupt_builder

    async def human_review_node(
        state: ConversationState,
        config: RunnableConfig,
    ) -> Command[str]:
        """Pause for human review and route based on the decision."""
        payload = interrupt_builder(state)  # type: ignore[misc]
        decision = interrupt(payload)
        goto = resume_router(decision, state)  # type: ignore[misc]
        return Command(goto=goto)

    # Attach metadata so callers can do:
    #   workflow.add_node("x", node, destinations=node.destinations)
    human_review_node.destinations = destinations  # type: ignore[attr-defined]
    return human_review_node


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
