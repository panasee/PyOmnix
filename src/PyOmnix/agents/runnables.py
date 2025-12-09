"""
Runnables for PyOmnix agents.

This module provides factory functions for creating stateless, atomic LCEL chains.
Each factory function accepts a language model and returns a compiled Runnable,
following the Factory Pattern for easy model injection and testing.

Design Philosophy:
- Runnables are pure, stateless transformation units
- No LangGraph or State dependencies - keep this module clean
- Each chain is atomic and composable (does ONE thing well)
- Use dependency injection for models
- No conditional routing - that belongs at the node level

Available Factories:
- create_chat_chain: Basic chat with system prompt and message history
- create_chat_chain_with_context: Chat with additional context injection
- create_summarization_chain: Summarize conversation history
- create_summary_update_chain: Update existing summary with new content
- create_query_rewrite_chain: Rephrase follow-up questions to standalone queries
- create_text_transform_chain: Generic text transformation with custom instructions
- create_structured_output_chain: Extract structured data using Pydantic schema
- create_json_output_chain: Extract JSON data using JSON schema
- create_classification_chain: Classify text into predefined categories
"""

import json
from typing import Any, cast

from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from pyomnix.agents.prompts import PROMPTS
from pyomnix.agents.schemas import ConversationState
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)

def empty_string_placeholder(variable_name: str) -> str:
    """
    Return a placeholder string for empty variables.
    """
    return f"No {variable_name} available"

def format_docs(docs: list[str | dict | Any]) -> str:
    """
    Format the documents list into a markdown string.
    """
    formatted: list[str] = []
    for i, doc in enumerate(docs):
        if isinstance(doc, str):
            formatted.append(f"--- Document {i + 1}---\n{doc}")
        elif isinstance(doc, dict):
            formatted.append(f"--- Document {i + 1} ---\n{json.dumps(doc)}")
        else:
            try:
                source = doc.metadata.get("source", "unknown")
                content = doc.page_content.replace("\n", " ")
                formatted.append(f"--- Document {i + 1} (Source: {source}) ---\n{content}")
            except AttributeError as err:
                raise ValueError(f"Invalid document type: {type(doc)}") from err
    return "\n\n".join(formatted)


def map_conversation_state_to_input(state: ConversationState | dict[str, Any]) -> dict[str, Any]:
    """
    Map the conversation state to the input for the chat chain.
    """
    structured_memory = state.get("structured_memory", empty_string_placeholder("structured memory"))
    if isinstance(structured_memory, dict):
        structured_memory = json.dumps(structured_memory)

    retrieved_docs = state.get("retrieved_docs", empty_string_placeholder("retrieved docs"))
    if isinstance(retrieved_docs, list):
        retrieved_docs = format_docs(retrieved_docs)

    return {
        "messages": state.get("messages", []),
        "summary": state.get("summary", empty_string_placeholder("summary")),
        "user_profile": state.get("user_profile", empty_string_placeholder("user profile")),
        "structured_memory": structured_memory,
        "retrieved_docs": retrieved_docs,
        "current_intent": state.get("current_intent", empty_string_placeholder("current intent")),
    }


def create_chat_chain(
    llm: BaseChatModel | _ConfigurableModel,
    system_prompt: str = PROMPTS.get("default"),
    tools: list | None = None,
) -> Runnable:
    """
    Create a basic chat chain using the given language model and system prompt.
    Optional system prompts: summary, user_profile, structured_memory, retrieved_docs, current_intent.

    Args:
        llm: The chat language model.
        system_prompt: The system prompt.
        tools: The tools to bind to the language model.

    Returns:
        Runnable: A chain that takes chat messages and returns a response string.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt + "\nCurrent Summary: {summary}\n"
                "User Profile: {user_profile}\n"
                "Structured Memory: {structured_memory}\n"
                "Retrieved Docs: {retrieved_docs}\n"
                "Current Intent: {current_intent}\n",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    if tools is not None:
        chain = RunnableLambda(map_conversation_state_to_input) | prompt | llm.bind_tools(tools)
    else:
        chain = RunnableLambda(map_conversation_state_to_input) | prompt | llm
    return chain


def create_structured_output_chain(
    llm: BaseChatModel,
    schema: type[Any] | dict[str, Any],
    system_prompt: str = PROMPTS.get("default"),
) -> Runnable:
    """
    Create a chain that extracts structured data from text using a Pydantic schema or JSON schema.
    Optional system prompts: summary, user_profile, structured_memory, retrieved_docs, current_intent.

    Args:
        llm: The language model to use for extraction.
        schema: The output structure, can be a Pydantic BaseModel, a TypedDict, or a JSON Schema (dict).
        system_prompt: The system prompt. Defaults to DEFAULT_SYSTEM_PROMPT.

    Returns:
        Runnable: A compiled LCEL chain that outputs the desired structure.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt + "Summary: {summary}\n"
                "User Profile: {user_profile}\n"
                "Structured Memory: {structured_memory}\n"
                "Retrieved Docs: {retrieved_docs}\n"
                "Current Intent: {current_intent}\n",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    if isinstance(schema, type) and (
        issubclass(schema, BaseModel) or hasattr(schema, "__annotations__")
    ):
        structured_llm = llm.with_structured_output(schema)
    elif isinstance(schema, dict):
        structured_llm = llm.with_structured_output(schema)
    else:
        raise ValueError(f"Invalid schema type: {type(schema)}")
    chain = prompt | structured_llm
    return cast(Runnable, chain)
