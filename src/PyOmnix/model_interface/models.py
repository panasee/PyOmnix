"""
This module provides a configuration manager for different model APIs.
It allows adding, removing, and retrieving API keys and URLs for various providers.
The configuration is stored in a JSON file and can be manually edited.
"""

import base64
import importlib
import io
import json
import os
from collections.abc import Generator
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Optional

import tiktoken
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable
from langgraph.graph import add_messages
from PIL import Image
from pydantic import BaseModel, Field, field_validator, model_validator

from pyomnix.consts import OMNIX_PATH, SUMMARIZE_PROMPT_HUMAN, SUMMARIZE_PROMPT_SYS
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


class ModelConfig:
    """
    Class to manage API keys and URLs for different model providers.
    Handles loading, saving, and retrieving configuration for various model APIs.
    Implements the Singleton pattern to ensure only one instance exists.
    """

    _instance: Optional["ModelConfig"] = None

    def __new__(cls) -> "ModelConfig":
        """
        Implement the Singleton pattern by ensuring only one instance is created.

        Returns:
            The single instance of ModelAPIConfig
        """
        if cls._instance is None:
            logger.debug("Creating new ModelAPIConfig instance")
            cls._instance = super(ModelConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, tracing: bool = False):
        """
        Initialize the ModelAPIConfig with the default config file path.
        This will only run once for the singleton instance.
        """
        if getattr(self, "_initialized", False):
            return

        if OMNIX_PATH is None:
            raise ValueError("OMNIX_PATH must be set to use ModelAPI")
        self.config_json = Path(f"{OMNIX_PATH}/api_config.json")
        self.config: dict[str, dict[str, Any]] = {}
        self._load_config()
        if tracing:
            self.setup_langsmith()
        self._initialized = True
        self.models = {}

    def _load_config(self) -> None:
        """Load the configuration from the JSON file if it exists."""
        if self.config_json.exists():
            with open(self.config_json, encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            logger.info(
                "No config file found at %s. Initialize empty configuration.",
                self.config_json,
            )
            self.config = {}
            with open(self.config_json, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)

    def setup_langsmith(self):
        """
        Setup LangSmith for tracing and monitoring.
        """
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = self.get_api_config("langsmith")[1]
        os.environ["LANGSMITH_API_KEY"] = self.get_api_config("langsmith")[0]
        os.environ["LANGSMITH_PROJECT"] = "pyomnix"

    @staticmethod
    def get_provider_model(model_full: str) -> tuple[str, str]:
        """
        Get the provider and model from the model full name.
        """
        provider_model = model_full.split("-")
        if len(provider_model) == 2:
            provider = provider_model[0]
            model = provider_model[1]
        else:
            model = provider_model[0]
            provider = model
        return provider, model

    def setup_models(self, models_fullname: str | list[str] = "deepseek"):
        """
        Setup providers and model apis (the api is a interface of langchain, not the real model called, actually deepseek/openai api can be used for most models). Indicate the provider before the model name if used. (e.g. "siliconflow-deepseek")

        Args:
            models_name(str | list[str]): The fullname of the model to use
        """
        model_module_dict = {
            "openai": ["langchain_openai", "ChatOpenAI"],
            "gemini": ["langchain_google_genai", "ChatGoogleGenerativeAI"],
            "claude": ["langchain_anthropic", "ChatAnthropic"],
            "deepseek": ["langchain_deepseek", "ChatDeepSeek"],
            "qwen": ["langchain_community.chat_models.tongyi", "ChatTongyi"],
        }
        if isinstance(models_fullname, str):
            models_fullname = [models_fullname]

        for model_full in models_fullname:
            provider, model = self.get_provider_model(model_full)

            logger.validate(
                model in model_module_dict,
                f"Model {model} not found in model_module_dict.",
            )
            if model_full in self.models:
                logger.info("Model %s already exists in models.", model)
                continue
            module = importlib.import_module(model_module_dict[model][0])
            if model == "deepseek":
                self.models[model_full] = partial(
                    getattr(module, model_module_dict[model][1]),
                    api_key=self.get_api_config(provider)[0],
                    api_base=self.get_api_config(provider)[1],
                )
            else:
                self.models[model_full] = partial(
                    getattr(module, model_module_dict[model][1]),
                    api_key=self.get_api_config(provider)[0],
                    base_url=self.get_api_config(provider)[1],
                )
            # base_url and api_base are the same, for different apis
            logger.info("Model %s initialized successfully.", model)
        return {model: self.models[model] for model in models_fullname}
        ##TODO: add interface for local models

    def save_config(self) -> None:
        """Save the current configuration to the JSON file."""
        with open(self.config_json, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)

    def set_api_config(
        self, provider: str, *, api_key: str | None, api_url: str | None
    ) -> None:
        """
        Set/Add the API key and URL for a specific provider.

        Args:
            provider: The model provider (e.g., 'openai', 'google', 'anthropic')
            api_key: The API key to set
            api_url: The API URL to set
        """
        if provider not in self.config:
            self.config[provider] = {}
        if api_key is not None:
            self.config[provider]["api_key"] = api_key
        if api_url is not None:
            self.config[provider]["api_url"] = api_url
        self.save_config()

    def get_api_config(self, provider: str) -> tuple[str, str] | None:
        """
        Get the API key for a specific provider.

        Args:
            provider: The model provider (e.g., 'openai', 'google', 'anthropic')

        Returns:
            The API key if found, None otherwise
        """
        if (
            provider in self.config
            and "api_key" in self.config[provider]
            and "api_url" in self.config[provider]
        ):
            return self.config[provider]["api_key"], self.config[provider]["api_url"]
        else:
            logger.error("Uncomplete API config found for provider %s", provider)
            return None

    def list_providers(self) -> list[str]:
        """
        List all configured providers.

        Returns:
            List of provider names
        """
        return list(self.config.keys())

    def check_provider_models(self, provider: str) -> list[str]:
        """
        Check the models supported by a specific provider.
        """
        if provider in self.config and "models" in self.config[provider]:
            return self.config[provider]["models"]
        return []

    def remove_provider(self, provider: str) -> bool:
        """
        Remove a provider from the configuration.

        Args:
            provider: The model provider to remove

        Returns:
            True if provider was removed, False if it didn't exist
        """
        if provider in self.config:
            del self.config[provider]
            self.save_config()
            logger.info("Provider %s removed from configuration.", provider)
            return True
        logger.warning("Provider %s not found in configuration.", provider)
        return False


class ChatMessageDict(BaseModel):
    """
    A class for representing a single message in a chat.
    """

    role: str = Field(
        description="The role of the message",
        pattern="(?i)^(system|user|human|assistant|tool|function)$",
        examples=["system", "user", "human", "assistant", "tool", "function"],
    )
    content: str | list
    tool_call_id: str | None = None
    name: str | None = None

    def to_langchain_message(self) -> BaseMessage:
        """Convert to appropriate LangChain message type."""
        match self.role.lower():
            case "system":
                return SystemMessage(content=self.content)
            case "user" | "human":
                return HumanMessage(content=self.content)
            case "assistant":
                return AIMessage(content=self.content)
            case "tool":
                return ToolMessage(
                    content=self.content, tool_call_id=self.tool_call_id or ""
                )
            case "function":
                return FunctionMessage(content=self.content, name=self.name or "")
            case _:
                return ChatMessage(content=self.content, role=self.role)


class ChatMessagesRaw(BaseModel):
    """
    A class for representing a list of messages. No ordering or structural rules.
    """

    messages: Annotated[
        list[BaseMessage],
        Field(description="The list of messages", default_factory=list),
        add_messages,
    ]

    def __add__(self, other: "ChatMessagesRaw") -> "ChatMessagesRaw":
        """
        Add two ChatMessages objects together.
        """
        logger.validate(
            isinstance(other, ChatMessagesRaw),
            "Other must be a ChatMessagesRaw object.",
        )
        return ChatMessagesRaw(messages=add_messages(self.messages, other.messages))


class ChatMessages(ChatMessagesRaw):
    """
    A class for representing a list of messages in a chat.
    This could be replacing MessagesState in langgraph.

    Be careful when modifying members directly, as this may break the structural validation.
    RESET final_check to False after any BREAKING modification and do final check every time before invoking.
    """

    final_check: Annotated[
        bool,
        Field(
            description="Whether to check the structure of the messages",
            default=False,
            exclude=True,
        ),
    ]

    file_sync: Annotated[
        bool,
        Field(
            description="Whether to sync the messages to local file",
            default=False,
        ),
    ]

    file_name: Annotated[
        str,
        Field(
            description="The name of the file to sync the messages to",
            default=f"chat_messages_{datetime.now().strftime('%Y%m%d_%H%M')}",
        ),
    ]
    trimed_file_name: Annotated[
        str,
        Field(
            description="The name of the file to sync the trimmed messages to",
            default=f"trimed_chat_messages_{datetime.now().strftime('%Y%m%d_%H%M')}",
        ),
    ]
    file_path: Path | None = None
    json_file_path: Path | None = None
    trimed_file_path: Path | None = None
    trimed_json_file_path: Path | None = None
    trimed_messages: Annotated[
        list[BaseMessage],
        Field(description="The list of trimmed messages", default_factory=list),
        add_messages,
    ]

    def __init__(self, **data):
        """Initialize the ChatMessages instance."""
        super().__init__(**data)
        # used to store the trimmed messages
        if self.file_sync:
            (OMNIX_PATH / "chat_files").mkdir(parents=True, exist_ok=True)
            self.file_path = (OMNIX_PATH / "chat_files" / self.file_name).with_suffix(
                ".txt"
            )
            self.json_file_path = (
                OMNIX_PATH / "chat_files" / self.file_name
            ).with_suffix(".json")
            self.trimed_file_path = (
                OMNIX_PATH / "chat_files" / self.trimed_file_name
            ).with_suffix(".txt")
            self.trimed_json_file_path = (
                OMNIX_PATH / "chat_files" / self.trimed_file_name
            ).with_suffix(".json")
            self._sync_to_file()

    @field_validator("messages", mode="before")
    @classmethod
    def convert_dict_to_chat_messages(cls, v):
        """
        Convert dictionaries in the messages list to ChatMessage objects.
        """
        logger.validate(
            isinstance(v, list), "ChatMessages Init: Messages must be a list."
        )
        v_new = []
        for i in v:
            if isinstance(i, BaseMessage):
                v_new.append(i)
            elif isinstance(i, dict):
                v_new.append(ChatMessageDict(**i).to_langchain_message())
            elif isinstance(i, ChatMessageDict):
                v_new.append(i.to_langchain_message())
            else:
                logger.raise_error("Invalid message type.", TypeError)

        return v_new

    @field_validator("messages", mode="after")
    @classmethod
    def check_structure(cls, v):
        """
        Check if the messages are in the correct structure.
        """
        logger.validate(
            isinstance(v[0], (SystemMessage, HumanMessage)),
            "The first message must be either a system message or a user message.",
        )

        if isinstance(v[0], SystemMessage) and len(v) > 1:
            logger.validate(
                isinstance(v[1], HumanMessage),
                "When starting with a system message, the second message must be a user message.",
            )

        for i in range(1, len(v)):
            if isinstance(v[i], ToolMessage):
                logger.validate(
                    isinstance(v[i - 1], AIMessage),
                    "A tool message should only follow an assistant message that requested the tool invocation.",
                )
            logger.validate(
                isinstance(
                    v[i], (HumanMessage, AIMessage, ToolMessage, FunctionMessage)
                ),
                "The message must be a user, assistant, tool or function message.",
            )
            logger.validate(
                type(v[i]) is not type(v[i - 1]),
                "Adjacent messages cannot be of the same type.",
            )

        return v

    @model_validator(mode="after")
    def check_structure_final(self) -> "ChatMessages":
        """
        Check if the messages are in the correct structure.
        """
        if self.final_check:
            logger.debug("Final check the structure of the messages.")
            if len(self.messages) > 0:
                logger.validate(
                    isinstance(self.messages[-1], (HumanMessage, ToolMessage)),
                    "The last message should be either a user message or a tool message.",
                )
        else:
            logger.debug("Skip final check.")
        return self

    def _sync_to_file(self) -> None:
        """Sync current messages to two files,
        one for human readable, one for machine readable (json)."""
        if not self.file_sync or self.file_path is None:
            return

        with open(self.file_path, "w", encoding="utf-8") as f:
            with open(self.json_file_path, "w", encoding="utf-8") as f_json:
                for msg in self.messages:
                    role = msg.__class__.__name__.replace("Message", "").lower()
                    if isinstance(msg.content, str):
                        content = msg.content
                    else:
                        content = str(
                            msg.content
                        )  # Basic serialization for complex content
                    f.write(
                        f"{role}\n\t Reasoning: {msg.additional_kwargs.get('reasoning_content', '')}\n\t Content: {content}\n"
                    )
                # Write JSON with proper formatting between messages
                json.dump(
                    messages_to_dict(self.messages),
                    f_json,
                    ensure_ascii=False,
                    indent=4,
                )

    def _sync_to_trimed_file(self) -> None:
        """Sync current trimmed messages to the file."""
        if not self.file_sync or self.trimed_file_path is None:
            return

        with open(self.trimed_file_path, "w", encoding="utf-8") as f:
            with open(self.trimed_json_file_path, "w", encoding="utf-8") as f_json:
                for msg in self.trimed_messages:
                    role = msg.__class__.__name__.replace("Message", "").lower()
                    if isinstance(msg.content, str):
                        content = msg.content
                    else:
                        content = str(
                            msg.content
                        )  # Basic serialization for complex content
                    f.write(
                        f"{role}\n\t Reasoning: {msg.additional_kwargs.get('reasoning_content', '')}\n\t Content: {content}\n"
                    )
                    # Write JSON with proper formatting between messages
                    if msg != self.trimed_messages[0]:
                        f_json.write(",\n")  # Add comma between JSON objects
                    else:
                        f_json.write("[\n")  # Start JSON array for first message
                    f_json.write(msg.model_dump_json())
                    if msg == self.trimed_messages[-1]:
                        f_json.write("\n]")  # Close JSON array after last message

    def load_from_file(self, file_path: Path) -> None:
        """Load messages from the json file."""
        if not os.path.exists(file_path):
            return

        self.messages.clear()
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            self.messages = messages_from_dict(data)
        self.final_check = False

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to handle message updates."""
        super().__setattr__(name, value)
        if name == "messages" and self.file_sync:
            self._sync_to_file()

    def request_response(
        self,
        model: Runnable,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        schema: dict | type = None,
    ) -> AIMessage:
        """
        Request a response from the model. (only supports invoke currently). The response will be appended to the messages and also be returned.
        Args:
            schema: The schema of the response, if None, the response will be a normal string.
        """
        if not self.final_check:
            checked_messages = ChatMessages(messages=self.messages, final_check=True)
        else:
            checked_messages = self

        if temperature is not None:
            model.temperature = temperature
        if max_tokens is not None:
            model.max_tokens = max_tokens
        if schema is not None:
            model = model.with_structured_output(schema)
        response = model.invoke(checked_messages.messages)
        # reset final_check to False, as appending a response will break the structural validation
        self.messages.append(response)
        self.final_check = False
        if self.file_sync:
            self._sync_to_file()
        return response

    def __add__(
        self, other: ChatMessagesRaw | BaseMessage | list[BaseMessage]
    ) -> "ChatMessages":
        """
        Add two ChatMessages objects together, with structural validation.
        Be careful with the implementation of add_messages, this will merge messages with same id.

        Args:
            other: Either a ChatMessagesRaw object, a single BaseMessage, or a list of BaseMessage objects

        Returns:
            A new ChatMessages instance with the combined messages

        Raises:
            ValueError: If other is not of the correct type or if the resulting messages violate structural rules
        """
        logger.validate(
            isinstance(other, (ChatMessagesRaw, BaseMessage, list)),
            "Other must be a ChatMessagesRaw object or a BaseMessage object or a list of BaseMessage objects.",
        )

        # Create a copy of the current instance's data
        data = self.model_dump()
        logger.debug("data: %s", data)

        # Handle different types of input
        if isinstance(other, ChatMessagesRaw):
            data["messages"] = add_messages(self.messages, other.messages)
        elif isinstance(other, BaseMessage):
            data["messages"] = self.messages + [other]
        else:
            logger.validate(
                all(isinstance(i, BaseMessage) for i in other),
                "All items in the list must be BaseMessage objects.",
            )
            data["messages"] = self.messages + other

        # Create new instance with combined data
        result = ChatMessages(**data)
        result.final_check = False

        # Sync to file if needed
        if result.file_sync:
            result._sync_to_file()
            result._sync_to_trimed_file()

        return result

    def drop_last(self) -> None:
        """Drop the last message."""
        self.messages.pop()
        self.final_check = False
        if self.file_sync:
            self._sync_to_file()

    def _generate_to_summarize_messages(self) -> "ChatMessages":
        """
        Summarize the messages into a single message within the max_tokens.
        """
        logger.validate(
            len(self.messages) > 2, "There must be at least 3 messages to summarize."
        )
        system_message = SystemMessage(
            content=SUMMARIZE_PROMPT_SYS,
        )
        combined_old_sys_first_human_mess = HumanMessage(
            content=f"The original system message is: '{self.messages[0].content}'.\n The starting user message is: '{self.messages[1].content}'"
        )
        if isinstance(self.messages[-1], HumanMessage):
            newest_human_message = HumanMessage(
                content=f"The last user message is: '{self.messages[-1].content}'\n Please summarize the conversation above"
            )
            return ChatMessages(
                messages=[system_message]
                + [combined_old_sys_first_human_mess]
                + self.messages[2:-1]
                + [newest_human_message]
            )
        else:
            newest_human_message = HumanMessage(content=SUMMARIZE_PROMPT_HUMAN)
            return ChatMessages(
                messages=[system_message]
                + [combined_old_sys_first_human_mess]
                + self.messages[2:]
                + [newest_human_message]
            )

    def summarize(
        self, model: Runnable, max_tokens: int = 1000, temperature: float = 0.6
    ) -> AIMessage:
        """
        Summarize the messages into a single message, will not change the stored messages.
        """
        model.temperature = temperature
        model.max_tokens = max_tokens
        return model.invoke(self._generate_to_summarize_messages().messages)

    def summarize_and_trim(
        self,
        model: Runnable,
        max_tokens: int = 1000,
        temperature: float = 0.6,
        preserve_conversation_turns: int = 2,
    ) -> None:
        """
        Summarize the messages into a single message and trim the messages to the max_tokens.

        Args:
            model: The chat model to use for summarization
            max_tokens: Maximum number of tokens for the summary
            temperature: Temperature setting for the model
            preserve_conversation_turns: Number of recent conversation turns to preserve

        Returns:
            None
        """
        sys_message = SystemMessage(content=self.messages[0].content)
        summary_human_message = HumanMessage(content=SUMMARIZE_PROMPT_HUMAN)
        summary_ai = self.summarize(model, max_tokens, temperature)

        if preserve_conversation_turns > 0:
            # Find the last N human messages and their corresponding AI responses
            human_index = [
                i
                for i, msg in enumerate(self.messages)
                if isinstance(msg, HumanMessage)
            ][-preserve_conversation_turns]

            preserved_messages = self.messages[human_index:]
            self.trimed_messages += self.messages[1:human_index]
        else:
            preserved_messages = []
            self.trimed_messages += self.messages[1:]

        self.messages = [
            sys_message,
            summary_human_message,
            summary_ai,
        ] + preserved_messages
        self.final_check = False
        if self.file_sync:
            self._sync_to_file()
            self._sync_to_trimed_file()

    def count_message_tokens(self, encoding: str = "cl100k_base") -> int:
        """
        Calculate the number of tokens in a message.

        Args:
            message: A LangChain message object

        Returns:
            int: The number of tokens in the message
        """
        encoding = tiktoken.get_encoding(encoding)
        content_tokens = 0
        role_tokens = 0

        for message in self.messages:
            if isinstance(message.content, str):
                content_tokens += len(encoding.encode(message.content))
            elif isinstance(message.content, list):
                # For multimodal content
                for item in message.content:
                    if isinstance(item, str) or (
                        isinstance(item, dict) and item.get("type") == "text"
                    ):
                        text = item if isinstance(item, str) else item.get("text", "")
                        content_tokens += len(encoding.encode(text))
                    # Images typically count as tokens based on size, but this is a simplification
                    elif isinstance(item, dict) and item.get("type") == "image":
                        content_tokens += 1024  # Placeholder estimate for images
            else:
                content_tokens += 0

            # Add tokens for message role (typically 1-4 tokens)
            role_tokens += len(
                encoding.encode(message.__class__.__name__.replace("Message", ""))
            )

        logger.debug("Content tokens: %s, Role tokens: %s", content_tokens, role_tokens)
        return content_tokens + role_tokens

    def auto_trim(
        self,
        model: Runnable,
        token_limit: int = 51000,
        summarize_tokens: int = 1000,
        temperature: float = 0.6,
        preserve_conversation_turns: int = 2,
    ) -> None:
        """
        Automatically trim the messages to the max_tokens.
        """
        if self.count_message_tokens() > token_limit:
            logger.debug("Summarizing and trimming the messages.")
            self.summarize_and_trim(
                model, summarize_tokens, temperature, preserve_conversation_turns
            )
        else:
            logger.debug("No need to trim the messages.")


class ChatRequest(BaseModel):
    """
    A class for representing a whole chat request for the RawModels.
    """

    provider_api: Annotated[
        str, Field(description="The full name of the provider-modelapi")
    ]
    model_name: Annotated[
        str, Field(description="The exact model name within the provider")
    ]
    messages: list[ChatMessages | ChatPromptValue]  # to be able to do batching
    stream: bool = False
    invoke_config: dict[str, str] | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: int | None = None
    max_retries: int = 2
    out_schema: dict | type | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def convert_dict_to_chat_messages(cls, v):
        """
        Convert dictionaries in the messages list to ChatMessage objects.
        """
        logger.validate(
            isinstance(v, list | ChatPromptValue | ChatMessages), "ChatRequest Init: Messages must be a list or a ChatPromptValue."
        )
        if isinstance(v, ChatPromptValue | ChatMessages):
            return [v]
        elif isinstance(v[0], ChatMessages):
            return v
        elif isinstance(v[0], list):
            return [ChatMessages(messages=i) for i in v]
        else:
            return [ChatMessages(messages=v)]


class RawModels:
    """
    class for a raw model interface.
    This class could embed and use multiple models at a time.
    """

    def __init__(self, provider_api: str | list[str]):
        self.models = ModelConfig().setup_models(provider_api)

        if isinstance(provider_api, str):
            provider_api = [provider_api]
        providers = [ModelConfig.get_provider_model(i)[0] for i in provider_api]
        self.input_tokens = {provider: {} for provider in providers}
        self.output_tokens = {provider: {} for provider in providers}

    def _update_token_count(self, provider: str, response: AIMessage) -> None:
        """
        Update the token count for the model.
        """
        if not isinstance(response, AIMessage):
            logger.debug("Response is not AIMessage, cannot count the tokens.")
            return
        model_name = response.response_metadata["model_name"]
        if model_name not in self.input_tokens[provider]:
            self.input_tokens[provider][model_name] = 0
        if model_name not in self.output_tokens[provider]:
            self.output_tokens[provider][model_name] = 0

        self.output_tokens[provider][model_name] += response.usage_metadata[
            "output_tokens"
        ]
        self.input_tokens[provider][model_name] += response.usage_metadata[
            "input_tokens"
        ]

    def _get_response(
        self,
        request_details: ChatRequest,
        **kwargs,
    ):
        """
        Get the response from the model with ChatRequest dataclass.
        extra kwargs will only be used for invoke method
        """
        chat_model = self.models[request_details.provider_api](
            model=request_details.model_name,
            temperature=request_details.temperature,
            max_tokens=request_details.max_tokens,
            timeout=request_details.timeout,
            max_retries=request_details.max_retries,
        )

        if request_details.out_schema is not None:
            chat_model = chat_model.with_structured_output(request_details.out_schema)

        if len(request_details.messages) > 1:
            logger.info("Batching messages for %s", chat_model)
            logger.validate(
                not request_details.stream, "Batch does not support streaming."
            )
            response = chat_model.batch([i.messages for i in request_details.messages])
        else:
            response = (
                chat_model.invoke(
                    request_details.messages[0].messages,
                    config=request_details.invoke_config,
                    **kwargs,
                )
                if not request_details.stream
                else chat_model.stream(request_details.messages[0])
            )

        return response

    def chat_completion(
        self,
        provider_api: str,
        model: str,
        messages: list[dict[str, str] | BaseMessage]
        | list[list[dict[str, str] | BaseMessage]],
        *,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: int | None = None,
        max_retries: int = 2,
        invoke_config: dict[str, Any] | None = None,
        schema: dict | type | None = None,
        **kwargs,
    ) -> list[AIMessage | Generator[dict, None, None]]:
        """
        Send a chat completion request to the model.

        Args:
            provider_api: The full name of the provider-modelapi (like "siliconflow-deepseek")
            model: The specific model to use (like "deepseek-chat")
            messages: The messages to send to the model or a batch of messages
            stream: Whether to stream the response
            invoke_config: Additional parameters to pass to the model
            temperature: The temperature to use for the model
            max_tokens: The maximum number of tokens to generate
            timeout: The timeout for the request
            max_retries: The maximum number of retries for the request
            **kwargs: Additional parameters to pass to the model invoke method
        """
        # Validate input using Pydantic
        request = ChatRequest(
            provider_api=provider_api,
            model_name=model,
            messages=messages,
            stream=stream,
            invoke_config=invoke_config,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            out_schema=schema,
        )

        response = self._get_response(request, **kwargs)

        if request.stream:
            logger.warning("Streaming is not supported for token counting.")
            return response

        provider = ModelConfig.get_provider_model(request.provider_api)[0]
        if not isinstance(response, list):
            response = [response]
        for r in response:
            self._update_token_count(provider, r)

        return response

    def chat_with_images(
        self,
        provider_api: str,
        model: str,
        texts: str | list[str | BaseMessage],
        images: list[str | Path | Image.Image] | list[list[str | Path | Image.Image]],
        system_messages: str | list[str | BaseMessage] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Send a multimodal chat request with text and images.

        Args:
            provider_api: The full name of the provider-modelapi (like "siliconflow-deepseek")
            model: The specific model to use
            texts: The text prompt to send
            images: List of image paths, URLs, or PIL Image objects
            system_messages: Optional system message to include
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the model

        Returns:
            The model's response as a string

        Raises:
            ValueError: If the provider does not support multimodal inputs
        """
        if not (
            isinstance(texts, list)
            == isinstance(images[0], list)
            == isinstance(system_messages, list)
        ):
            logger.error(
                "text, images and system_message must be consistent on if_batch and the batch length."
            )
            return
        # below the text, images and system_message are assured to be same format
        if not isinstance(texts, list):
            texts = [texts]
            images = [images]
            system_messages = [system_messages]
        elif not (len(texts) == len(images) == len(system_messages)):
            logger.error("text, images and system_message must have the same length.")
            return
        batch_length = len(texts)
        # below the text, images and system_message are assured to be lists of same length
        logger.info("Please confirm the exact model is multimodal by yourself.")
        # Prepare messages
        batch_messages = [[]] * batch_length
        for i in range(batch_length):
            images_single = images[i]
            text = texts[i]
            batch_messages[i].append(SystemMessage(content=system_messages[i]))
            # Process images based on model type
            if ModelConfig.get_provider_model(provider_api)[1] in [
                "openai",
                "google",
                "deepseek",
            ]:
                # OpenAI format for multimodal
                content = [{"type": "text", "text": text}]
                for img in images_single:
                    image_data = self._process_image(img)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        }
                    )

                batch_messages[i].append(HumanMessage(content=content))

            elif ModelConfig.get_provider_model(provider_api)[1] == "anthropic":
                # Anthropic format for multimodal
                content = [{"type": "text", "text": text}]

                for img in images_single:
                    image_data = self._process_image(img)
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        }
                    )

                batch_messages[i].append(HumanMessage(content=content))

            else:
                logger.error(
                    "no implementation for %s",
                    ModelConfig.get_provider_model(provider_api)[1],
                )
                return

        return self.chat_completion(
            provider_api, model, batch_messages, stream=stream, **kwargs
        )

    def _process_image(self, image: str | Path | Image.Image) -> str:
        """
        Process an image into base64 format.

        Args:
            image: Image path, URL, or PIL Image object

        Returns:
            Base64-encoded image data
        """
        if isinstance(image, (str, Path)):
            img_path = str(image)
            if img_path.startswith(("http://", "https://")):
                # For URLs, we need to download the image first
                import requests

                response = requests.get(img_path)
                img = Image.open(io.BytesIO(response.content))
            else:
                img = Image.open(img_path)
        else:
            img = image

        # Convert PIL Image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
