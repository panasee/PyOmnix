"""
This module provides a configuration manager for different model APIs.
It allows adding, removing, and retrieving API keys and URLs for various providers.
The configuration is stored in a JSON file and can be manually edited.
"""

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import _ConfigurableModel
from pydantic import BaseModel, Field, PostgresDsn, field_validator
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from pyomnix.consts import OMNIX_PATH
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)

PROVIDER_ALIASES = {  # use small letters
    "openai": {"openai", "gpt"},
    "google_vertexai": {"vertexai", "vertex"},
    "google_genai": {"gemini", "aistudio", "genai", "google"},
    "anthropic": {"anthropic", "claude"},
    "deepseek": {"deepseek", "ds"},
    "groq": {"groq"},
}


class ProviderConfig(BaseModel):
    """
    Configuration for a model provider loaded from JSON.

    JSON format:
    {
        "provider_name": {
            "api_key": "...",       // required
            "base_url": "...",      // required
            "provider_kwargs": {}   // optional
        }
    }
    """

    api_key: str = Field(description="The API key for the model provider")
    base_url: str = Field(description="The base URL for the model provider")
    provider_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Optional keyword arguments for the model provider"
    )


class Settings(BaseSettings):
    """
    Application settings loaded from a JSON configuration file.

    The JSON file should be located at {OMNIX_PATH}/api_config.json with format:
    {
        "langsmith": {"api_key": "...", "base_url": "..."},
        "openai": {"api_key": "...", "base_url": "...", "provider_kwargs": {...}},
        ...
    }

    For provider configs, "api_key" and "base_url" are required,
    while "provider_kwargs" is optional.
    """

    langsmith: ProviderConfig
    request_appendix: dict[str, str] = Field(default_factory=dict)
    openai: ProviderConfig
    google: ProviderConfig
    vertex: ProviderConfig
    deepseek: ProviderConfig
    groq: ProviderConfig
    zhipu: ProviderConfig
    aliyun: ProviderConfig
    volcengine: ProviderConfig
    siliconflow: ProviderConfig
    supabase_dsn: PostgresDsn | None = None
    supabase_ssl_cert: Path | None = None
    gdrive_key: dict[str, Any] | None = None
    gdrive_folder_id: str | None = None

    postgres_pool_min_size: int = Field(
        default=1, description="Minimum number of connections in the async pool"
    )
    postgres_pool_max_size: int = Field(
        default=10, description="Maximum number of connections in the async pool"
    )

    @field_validator("supabase_ssl_cert", mode="before")
    @classmethod
    def validate_supabase_ssl_cert(cls, v: Path | str | None) -> Path | None:
        """
        Validate the Supabase SSL certificate path.
        """
        if v is None:
            return None
        return (OMNIX_PATH / Path(v)).expanduser()

    model_config = SettingsConfigDict(
        json_file=f"{OMNIX_PATH}/api_config.json",
        json_file_encoding="utf-8",
        extra="forbid",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize settings sources to use dynamic JSON config path.
        Priority (highest to lowest): init_settings > json_config > env > dotenv > secrets
        """
        return (
            init_settings,
            JsonConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Loads configuration from {OMNIX_PATH}/api_config.json at runtime.
    """
    return Settings()  # type: ignore[call-arg]


class ModelConfig:
    """
    Class to manage API keys and URLs for different model providers.
    Handles loading, saving, and retrieving configuration for various model APIs.
    Implements the Singleton pattern to ensure only one instance exists.
    """

    _instance: "ModelConfig | None" = None

    _instance_lock = threading.Lock()

    def __new__(cls) -> "ModelConfig":
        """
        Implement the Singleton pattern by ensuring only one instance is created,
        with multi-thread safety.

        Returns:
            The single instance of ModelAPIConfig
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    logger.debug("Creating new ModelConfig instance thread-safely")
                    cls._instance = super(ModelConfig, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance  # type: ignore[return-value]

    def __init__(self, tracing: bool = False):
        """
        Initialize the ModelAPIConfig with the default config file path.
        This will only run once for the singleton instance.
        """
        if getattr(self, "_initialized", False):
            return

        if OMNIX_PATH is None:
            raise ValueError("OMNIX_PATH must be set to use ModelAPI")
        self.settings = get_settings()
        if tracing:
            self.setup_langsmith()
        self._initialized = True
        self.model_factories: dict[str, _ConfigurableModel] = {}

    def setup_langsmith(self):
        """
        Setup LangSmith for tracing and monitoring.
        """
        api_cfg = self.get_api_config("langsmith")
        if api_cfg is None:
            logger.error("LangSmith API config missing; skipping tracing setup.")
            return
        api_key, base_url, _ = api_cfg
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = base_url
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = "pyomnix"

    @staticmethod
    def extract_provider_model(full_name: str) -> tuple[str, str]:
        """
        Get the provider and api name from the full name.
        The format is "provider-api", where "api" is actually the model api used
        instead of the specific model.
        (deepseek/google_genai/google_vertexai/openai/anthropic)
        """
        provider_model = full_name.split("-")
        if len(provider_model) == 2:
            provider, api_name = provider_model
        elif len(provider_model) == 1:
            provider = provider_model[0]
            api_name = provider
        else:
            raise ValueError(f"Invalid model full name: {full_name}")
        return provider, api_name

    def setup_model_factory(self, factory_fullname: str | list[str] = "deepseek", **user_kwargs):
        """
        Setup providers and model apis (deepseek/openai api can be used for most models). Indicate the provider before the model name if used. (e.g. "siliconflow-deepseek")
        The provider portion is looked up in the local config for credentials, while the API portion selects the LangChain integration to use.

        Returns:
            A dictionary of initialized model factories, they can be used to invoke with messages, or preferably configed before use.
        """

        def canonicalize_provider(provider: str) -> tuple[str, bool]:
            """
            Canonicalize the model name and judge if it is a intrinsically supported provider.
            """
            for target, aliases in PROVIDER_ALIASES.items():
                if provider.strip().lower() in aliases:
                    return target, True
            return provider, False

        if isinstance(factory_fullname, str):
            factory_fullname = [factory_fullname]

        initialized: dict[str, _ConfigurableModel] = {}

        for factory_name in factory_fullname:
            provider, api_name = self.extract_provider_model(factory_name)

            if factory_name in self.model_factories:
                logger.info("Model %s already initialized; reusing factory.", factory_name)
                initialized[factory_name] = self.model_factories[factory_name]
                continue

            provider, is_intrinsically_supported = canonicalize_provider(provider)
            api_key, base_url, provider_kwargs = self.get_api_config(provider)
            total_kwargs = {
                **provider_kwargs,
                **user_kwargs,
            }  # user_kwargs will override provider_kwargs

            if not is_intrinsically_supported:
                model_factory = init_chat_model(
                    model_provider=provider,
                    configurable_fields="any",
                    api_key=api_key,
                    base_url=base_url,
                    **total_kwargs,
                )
            else:
                model_factory = init_chat_model(
                    model_provider=api_name,
                    configurable_fields="any",
                    api_key=api_key,
                    **total_kwargs,
                )
            self.model_factories[factory_name] = model_factory
            initialized[factory_name] = model_factory
            logger.info(
                "Model API %s initialized successfully for provider %s.",
                api_name,
                provider,
            )

        return initialized
        ##TODO: add interface for local models

    def get_api_config(self, provider: str) -> tuple[str, str, dict[str, Any]]:
        """
        Get the API configuration for a specific provider.

        Args:
            provider: The model provider (e.g., 'openai', 'google', 'deepseek')

        Returns:
            A tuple of (api_key, base_url, provider_kwargs)

        Raises:
            ValueError: If the provider is not found in configuration
        """
        provider_cfg: ProviderConfig | None = getattr(self.settings, provider, None)
        if provider_cfg is None:
            logger.raise_error(f"Provider {provider} not found in configuration", ValueError)  # type: ignore[return-value]

        return provider_cfg.api_key, provider_cfg.base_url, provider_cfg.provider_kwargs

    def list_providers(self) -> list[str]:
        """
        List all configured providers.

        Returns:
            List of provider names that have ProviderConfig
        """
        providers = []
        for field_name in self.settings.model_fields.keys():
            field_value = getattr(self.settings, field_name, None)
            if isinstance(field_value, ProviderConfig):
                providers.append(field_name)
        return providers


# class ChatMessages(ChatMessagesRaw):
#    """
#    A class for representing a list of messages in a chat.
#    This could be replacing MessagesState in langgraph.
#
#    Be careful when modifying members directly, as this may break the structural validation.
#    RESET final_check to False after any BREAKING modification and do final check every time before invoking.
#    """
#
#    final_check: Annotated[
#        bool,
#        Field(
#            description="Whether to check the structure of the messages",
#            default=False,
#            exclude=True,
#        ),
#    ]
#
#    file_sync: Annotated[
#        bool,
#        Field(
#            description="Whether to sync the messages to local file",
#            default=False,
#        ),
#    ]
#
#    file_name: Annotated[
#        str,
#        Field(
#            description="The name of the file to sync the messages to",
#            default=f"chat_messages_{datetime.now().strftime('%Y%m%d_%H%M')}",
#        ),
#    ]
#    trimed_file_name: Annotated[
#        str,
#        Field(
#            description="The name of the file to sync the trimmed messages to",
#            default=f"trimed_chat_messages_{datetime.now().strftime('%Y%m%d_%H%M')}",
#        ),
#    ]
#    file_path: Path | None = None
#    json_file_path: Path | None = None
#    trimed_file_path: Path | None = None
#    trimed_json_file_path: Path | None = None
#    trimed_messages: Annotated[
#        list[BaseMessage],
#        Field(description="The list of trimmed messages", default_factory=list),
#        add_messages,
#    ]
#
#    def __init__(self, **data):
#        """Initialize the ChatMessages instance."""
#        super().__init__(**data)
#        # used to store the trimmed messages
#        if self.file_sync:
#            (OMNIX_PATH / "chat_files").mkdir(parents=True, exist_ok=True)
#            self.file_path = (OMNIX_PATH / "chat_files" / self.file_name).with_suffix(
#                ".txt"
#            )
#            self.json_file_path = (
#                OMNIX_PATH / "chat_files" / self.file_name
#            ).with_suffix(".json")
#            self.trimed_file_path = (
#                OMNIX_PATH / "chat_files" / self.trimed_file_name
#            ).with_suffix(".txt")
#            self.trimed_json_file_path = (
#                OMNIX_PATH / "chat_files" / self.trimed_file_name
#            ).with_suffix(".json")
#            self._sync_to_file()
#
#    @field_validator("messages", mode="before")
#    @classmethod
#    def convert_dict_to_chat_messages(cls, v):
#        """
#        Convert dictionaries in the messages list to ChatMessage objects.
#        """
#        logger.validate(
#            isinstance(v, list), "ChatMessages Init: Messages must be a list."
#        )
#        v_new = []
#        for i in v:
#            if isinstance(i, BaseMessage):
#                v_new.append(i)
#            elif isinstance(i, dict):
#                v_new.append(ChatMessageDict(**i).to_langchain_message())
#            elif isinstance(i, ChatMessageDict):
#                v_new.append(i.to_langchain_message())
#            else:
#                logger.raise_error("Invalid message type.", TypeError)
#
#        return v_new
#
#    @field_validator("messages", mode="after")
#    @classmethod
#    def check_structure(cls, v):
#        """
#        Check if the messages are in the correct structure.
#        """
#        logger.validate(
#            isinstance(v[0], (SystemMessage, HumanMessage)),
#            "The first message must be either a system message or a user message.",
#        )
#
#        if isinstance(v[0], SystemMessage) and len(v) > 1:
#            logger.validate(
#                isinstance(v[1], HumanMessage),
#                "When starting with a system message, the second message must be a user message.",
#            )
#
#        for i in range(1, len(v)):
#            if isinstance(v[i], ToolMessage):
#                logger.validate(
#                    isinstance(v[i - 1], AIMessage),
#                    "A tool message should only follow an assistant message that requested the tool invocation.",
#                )
#            logger.validate(
#                isinstance(
#                    v[i], (HumanMessage, AIMessage, ToolMessage, FunctionMessage)
#                ),
#                "The message must be a user, assistant, tool or function message.",
#            )
#            logger.validate(
#                type(v[i]) is not type(v[i - 1]),
#                "Adjacent messages cannot be of the same type.",
#            )
#
#        return v
#
#    @model_validator(mode="after")
#    def check_structure_final(self) -> "ChatMessages":
#        """
#        Check if the messages are in the correct structure.
#        """
#        if self.final_check:
#            logger.debug("Final check the structure of the messages.")
#            if len(self.messages) > 0:
#                logger.validate(
#                    isinstance(self.messages[-1], (HumanMessage, ToolMessage)),
#                    "The last message should be either a user message or a tool message.",
#                )
#        else:
#            logger.debug("Skip final check.")
#        return self
#
#    def _sync_to_file(self) -> None:
#        """Sync current messages to two files,
#        one for human readable, one for machine readable (json)."""
#        if not self.file_sync or self.file_path is None:
#            return
#
#        with open(self.file_path, "w", encoding="utf-8") as f:
#            with open(self.json_file_path, "w", encoding="utf-8") as f_json:
#                for msg in self.messages:
#                    role = msg.__class__.__name__.replace("Message", "").lower()
#                    if isinstance(msg.content, str):
#                        content = msg.content
#                    else:
#                        content = str(
#                            msg.content
#                        )  # Basic serialization for complex content
#                    f.write(
#                        f"{role}\n\t Reasoning: {msg.additional_kwargs.get('reasoning_content', '')}\n\t Content: {content}\n"
#                    )
#                # Write JSON with proper formatting between messages
#                json.dump(
#                    messages_to_dict(self.messages),
#                    f_json,
#                    ensure_ascii=False,
#                    indent=4,
#                )
#
#    def _sync_to_trimed_file(self) -> None:
#        """Sync current trimmed messages to the file."""
#        if not self.file_sync or self.trimed_file_path is None:
#            return
#
#        with open(self.trimed_file_path, "w", encoding="utf-8") as f:
#            with open(self.trimed_json_file_path, "w", encoding="utf-8") as f_json:
#                for msg in self.trimed_messages:
#                    role = msg.__class__.__name__.replace("Message", "").lower()
#                    if isinstance(msg.content, str):
#                        content = msg.content
#                    else:
#                        content = str(
#                            msg.content
#                        )  # Basic serialization for complex content
#                    f.write(
#                        f"{role}\n\t Reasoning: {msg.additional_kwargs.get('reasoning_content', '')}\n\t Content: {content}\n"
#                    )
#                    # Write JSON with proper formatting between messages
#                    if msg != self.trimed_messages[0]:
#                        f_json.write(",\n")  # Add comma between JSON objects
#                    else:
#                        f_json.write("[\n")  # Start JSON array for first message
#                    f_json.write(msg.model_dump_json())
#                    if msg == self.trimed_messages[-1]:
#                        f_json.write("\n]")  # Close JSON array after last message
#
#    def load_from_file(self, file_path: Path) -> None:
#        """Load messages from the json file."""
#        if not os.path.exists(file_path):
#            return
#
#        self.messages.clear()
#        with open(file_path, encoding="utf-8") as f:
#            data = json.load(f)
#            self.messages = messages_from_dict(data)
#        self.final_check = False
#
#    def __setattr__(self, name: str, value: Any) -> None:
#        """Override setattr to handle message updates."""
#        super().__setattr__(name, value)
#        if name == "messages" and self.file_sync:
#            self._sync_to_file()
#
#    def request_response(
#        self,
#        model: Runnable,
#        *,
#        max_tokens: int | None = None,
#        temperature: float | None = None,
#        schema: dict | type = None,
#    ) -> AIMessage:
#        """
#        Request a response from the model. (only supports invoke currently). The response will be appended to the messages and also be returned.
#        Args:
#            schema: The schema of the response, if None, the response will be a normal string.
#        """
#        if not self.final_check:
#            checked_messages = ChatMessages(messages=self.messages, final_check=True)
#        else:
#            checked_messages = self
#
#        if temperature is not None:
#            model.temperature = temperature
#        if max_tokens is not None:
#            model.max_tokens = max_tokens
#        if schema is not None:
#            model = model.with_structured_output(schema)
#        response = model.invoke(checked_messages.messages)
#        # reset final_check to False, as appending a response will break the structural validation
#        self.messages.append(response)
#        self.final_check = False
#        if self.file_sync:
#            self._sync_to_file()
#        return response
#
#    def __add__(
#        self, other: ChatMessagesRaw | BaseMessage | list[BaseMessage]
#    ) -> "ChatMessages":
#        """
#        Add two ChatMessages objects together, with structural validation.
#        Be careful with the implementation of add_messages, this will merge messages with same id.
#
#        Args:
#            other: Either a ChatMessagesRaw object, a single BaseMessage, or a list of BaseMessage objects
#
#        Returns:
#            A new ChatMessages instance with the combined messages
#
#        Raises:
#            ValueError: If other is not of the correct type or if the resulting messages violate structural rules
#        """
#        logger.validate(
#            isinstance(other, (ChatMessagesRaw, BaseMessage, list)),
#            "Other must be a ChatMessagesRaw object or a BaseMessage object or a list of BaseMessage objects.",
#        )
#
#        # Create a copy of the current instance's data
#        data = self.model_dump()
#        logger.debug("data: %s", data)
#
#        # Handle different types of input
#        if isinstance(other, ChatMessagesRaw):
#            data["messages"] = add_messages(self.messages, other.messages)
#        elif isinstance(other, BaseMessage):
#            data["messages"] = self.messages + [other]
#        else:
#            logger.validate(
#                all(isinstance(i, BaseMessage) for i in other),
#                "All items in the list must be BaseMessage objects.",
#            )
#            data["messages"] = self.messages + other
#
#        # Create new instance with combined data
#        result = ChatMessages(**data)
#        result.final_check = False
#
#        # Sync to file if needed
#        if result.file_sync:
#            result._sync_to_file()
#            result._sync_to_trimed_file()
#
#        return result
#
#    def drop_last(self) -> None:
#        """Drop the last message."""
#        self.messages.pop()
#        self.final_check = False
#        if self.file_sync:
#            self._sync_to_file()
#
#    def _generate_to_summarize_messages(self) -> "ChatMessages":
#        """
#        Summarize the messages into a single message within the max_tokens.
#        """
#        logger.validate(
#            len(self.messages) > 2, "There must be at least 3 messages to summarize."
#        )
#        system_message = SystemMessage(
#            content=SUMMARIZE_PROMPT_SYS,
#        )
#        combined_old_sys_first_human_mess = HumanMessage(
#            content=f"The original system message is: '{self.messages[0].content}'.\n The starting user message is: '{self.messages[1].content}'"
#        )
#        if isinstance(self.messages[-1], HumanMessage):
#            newest_human_message = HumanMessage(
#                content=f"The last user message is: '{self.messages[-1].content}'\n Please summarize the conversation above"
#            )
#            return ChatMessages(
#                messages=[system_message]
#                + [combined_old_sys_first_human_mess]
#                + self.messages[2:-1]
#                + [newest_human_message]
#            )
#        else:
#            newest_human_message = HumanMessage(content=SUMMARIZE_PROMPT_HUMAN)
#            return ChatMessages(
#                messages=[system_message]
#                + [combined_old_sys_first_human_mess]
#                + self.messages[2:]
#                + [newest_human_message]
#            )
#
#    def summarize(
#        self, model: Runnable, max_tokens: int = 1000, temperature: float = 0.6
#    ) -> AIMessage:
#        """
#        Summarize the messages into a single message, will not change the stored messages.
#        """
#        model.temperature = temperature
#        model.max_tokens = max_tokens
#        return model.invoke(self._generate_to_summarize_messages().messages)
#
#    def summarize_and_trim(
#        self,
#        model: Runnable,
#        max_tokens: int = 1000,
#        temperature: float = 0.6,
#        preserve_conversation_turns: int = 2,
#    ) -> None:
#        """
#        Summarize the messages into a single message and trim the messages to the max_tokens.
#
#        Args:
#            model: The chat model to use for summarization
#            max_tokens: Maximum number of tokens for the summary
#            temperature: Temperature setting for the model
#            preserve_conversation_turns: Number of recent conversation turns to preserve
#
#        Returns:
#            None
#        """
#        sys_message = SystemMessage(content=self.messages[0].content)
#        summary_human_message = HumanMessage(content=SUMMARIZE_PROMPT_HUMAN)
#        summary_ai = self.summarize(model, max_tokens, temperature)
#
#        if preserve_conversation_turns > 0:
#            # Find the last N human messages and their corresponding AI responses
#            human_index = [
#                i
#                for i, msg in enumerate(self.messages)
#                if isinstance(msg, HumanMessage)
#            ][-preserve_conversation_turns]
#
#            preserved_messages = self.messages[human_index:]
#            self.trimed_messages += self.messages[1:human_index]
#        else:
#            preserved_messages = []
#            self.trimed_messages += self.messages[1:]
#
#        self.messages = [
#            sys_message,
#            summary_human_message,
#            summary_ai,
#        ] + preserved_messages
#        self.final_check = False
#        if self.file_sync:
#            self._sync_to_file()
#            self._sync_to_trimed_file()
#
#    def count_message_tokens(self, encoding: str = "cl100k_base") -> int:
#        """
#        Calculate the number of tokens in a message.
#
#        Args:
#            message: A LangChain message object
#
#        Returns:
#            int: The number of tokens in the message
#        """
#        encoding = tiktoken.get_encoding(encoding)
#        content_tokens = 0
#        role_tokens = 0
#
#        for message in self.messages:
#            if isinstance(message.content, str):
#                content_tokens += len(encoding.encode(message.content))
#            elif isinstance(message.content, list):
#                # For multimodal content
#                for item in message.content:
#                    if isinstance(item, str) or (
#                        isinstance(item, dict) and item.get("type") == "text"
#                    ):
#                        text = item if isinstance(item, str) else item.get("text", "")
#                        content_tokens += len(encoding.encode(text))
#                    # Images typically count as tokens based on size, but this is a simplification
#                    elif isinstance(item, dict) and item.get("type") == "image":
#                        content_tokens += 1024  # Placeholder estimate for images
#            else:
#                content_tokens += 0
#
#            # Add tokens for message role (typically 1-4 tokens)
#            role_tokens += len(
#                encoding.encode(message.__class__.__name__.replace("Message", ""))
#            )
#
#        logger.debug("Content tokens: %s, Role tokens: %s", content_tokens, role_tokens)
#        return content_tokens + role_tokens
#
#    def auto_trim(
#        self,
#        model: Runnable,
#        token_limit: int = 51000,
#        summarize_tokens: int = 1000,
#        temperature: float = 0.6,
#        preserve_conversation_turns: int = 2,
#    ) -> None:
#        """
#        Automatically trim the messages to the max_tokens.
#        """
#        if self.count_message_tokens() > token_limit:
#            logger.debug("Summarizing and trimming the messages.")
#            self.summarize_and_trim(
#                model, summarize_tokens, temperature, preserve_conversation_turns
#            )
#        else:
#            logger.debug("No need to trim the messages.")
#
#
# class ChatRequest(BaseModel):
#    """
#    A class for representing a whole chat request for the RawModels.
#    """
#
#    provider_api: Annotated[
#        str, Field(description="The full name of the provider-modelapi")
#    ]
#    model_name: Annotated[
#        str, Field(description="The exact model name within the provider")
#    ]
#    messages: list[ChatMessages | ChatPromptValue]  # to be able to do batching
#    stream: bool = False
#    invoke_config: dict[str, str] | None = None
#    temperature: float = 0.7
#    max_tokens: int | None = None
#    timeout: int | None = None
#    max_retries: int = 2
#    out_schema: dict | type | None = None
#
#    @field_validator("messages", mode="before")
#    @classmethod
#    def convert_dict_to_chat_messages(cls, v):
#        """
#        Convert dictionaries in the messages list to ChatMessage objects.
#        """
#        logger.validate(
#            isinstance(v, list | ChatPromptValue | ChatMessages), "ChatRequest Init: Messages must be a list or a ChatPromptValue."
#        )
#        if isinstance(v, ChatPromptValue | ChatMessages):
#            return [v]
#        elif isinstance(v[0], ChatMessages):
#            return v
#        elif isinstance(v[0], list):
#            return [ChatMessages(messages=i) for i in v]
#        else:
#            return [ChatMessages(messages=v)]


# class RawModels:
#    """
#    class for a raw model interface.
#    This class could embed and use multiple models at a time.
#    """
#
#    def __init__(self, provider_api: str | list[str]):
#        self.models = ModelConfig().setup_model_factory(provider_api)
#
#        if isinstance(provider_api, str):
#            provider_api = [provider_api]
#        providers = [ModelConfig.extract_provider_model(i)[0] for i in provider_api]
#        self.input_tokens = {provider: {} for provider in providers}
#        self.output_tokens = {provider: {} for provider in providers}
#
#    def _update_token_count(self, provider: str, response: AIMessage) -> None:
#        """
#        Update the token count for the model.
#        """
#        if not isinstance(response, AIMessage):
#            logger.debug("Response is not AIMessage, cannot count the tokens.")
#            return
#        model_name = response.response_metadata["model_name"]
#        if model_name not in self.input_tokens[provider]:
#            self.input_tokens[provider][model_name] = 0
#        if model_name not in self.output_tokens[provider]:
#            self.output_tokens[provider][model_name] = 0
#
#        self.output_tokens[provider][model_name] += response.usage_metadata[
#            "output_tokens"
#        ]
#        self.input_tokens[provider][model_name] += response.usage_metadata[
#            "input_tokens"
#        ]
#
#    def _get_response(
#        self,
#        request_details: ChatRequest,
#        **kwargs,
#    ):
#        """
#        Get the response from the model with ChatRequest dataclass.
#        extra kwargs will only be used for invoke method
#        """
#        chat_model = self.models[request_details.provider_api](
#            model=request_details.model_name,
#            temperature=request_details.temperature,
#            max_tokens=request_details.max_tokens,
#            timeout=request_details.timeout,
#            max_retries=request_details.max_retries,
#        )
#
#        if request_details.out_schema is not None:
#            chat_model = chat_model.with_structured_output(request_details.out_schema)
#
#        if len(request_details.messages) > 1:
#            logger.info("Batching messages for %s", chat_model)
#            logger.validate(
#                not request_details.stream, "Batch does not support streaming."
#            )
#            response = chat_model.batch([i.messages for i in request_details.messages])
#        else:
#            response = (
#                chat_model.invoke(
#                    request_details.messages[0].messages,
#                    config=request_details.invoke_config,
#                    **kwargs,
#                )
#                if not request_details.stream
#                else chat_model.stream(request_details.messages[0].messages)
#            )
#
#        return response
#
#    def chat_completion(
#        self,
#        provider_api: str,
#        model: str,
#        messages: list[dict[str, str] | BaseMessage]
#        | list[list[dict[str, str] | BaseMessage]],
#        *,
#        stream: bool = False,
#        temperature: float = 0.7,
#        max_tokens: int | None = None,
#        timeout: int | None = None,
#        max_retries: int = 2,
#        invoke_config: dict[str, Any] | None = None,
#        schema: dict | type | None = None,
#        **kwargs,
#    ) -> list[AIMessage | Generator[dict, None, None]]:
#        """
#        Send a chat completion request to the model.
#
#        Args:
#            provider_api: The full name of the provider-modelapi (like "siliconflow-deepseek")
#            model: The specific model to use (like "deepseek-chat")
#            messages: The messages to send to the model or a batch of messages
#            stream: Whether to stream the response
#            invoke_config: Additional parameters to pass to the model
#            temperature: The temperature to use for the model
#            max_tokens: The maximum number of tokens to generate
#            timeout: The timeout for the request
#            max_retries: The maximum number of retries for the request
#            **kwargs: Additional parameters to pass to the model invoke method
#        """
#        # Validate input using Pydantic
#        request = ChatRequest(
#            provider_api=provider_api,
#            model_name=model,
#            messages=messages,
#            stream=stream,
#            invoke_config=invoke_config,
#            temperature=temperature,
#            max_tokens=max_tokens,
#            timeout=timeout,
#            max_retries=max_retries,
#            out_schema=schema,
#        )
#
#        response = self._get_response(request, **kwargs)
#
#        if request.stream:
#            logger.warning("Streaming is not supported for token counting.")
#            return response
#
#        provider = ModelConfig.extract_provider_model(request.provider_api)[0]
#        if not isinstance(response, list):
#            response = [response]
#        for r in response:
#            self._update_token_count(provider, r)
#
#        return response
#
#    def chat_with_images(
#        self,
#        provider_api: str,
#        model: str,
#        texts: str | list[str | BaseMessage],
#        images: list[str | Path | Image.Image] | list[list[str | Path | Image.Image]],
#        system_messages: str | list[str | BaseMessage] | None = None,
#        stream: bool = False,
#        **kwargs,
#    ) -> list[AIMessage | Generator[dict, None, None]]:
#        """
#        Send a multimodal chat request with text and images.
#
#        Args:
#            provider_api: The full name of the provider-modelapi (like "siliconflow-deepseek")
#            model: The specific model to use
#            texts: The text prompt to send
#            images: List of image paths, URLs, or PIL Image objects
#            system_messages: Optional system message to include
#            stream: Whether to stream the response
#            **kwargs: Additional parameters to pass to the model
#
#        Returns:
#            The model's response as a string
#
#        Raises:
#            ValueError: If the provider does not support multimodal inputs
#        """
#        # Normalize inputs into batched lists
#        texts_list = texts if isinstance(texts, list) else [texts]
#        if isinstance(images, list) and images and isinstance(images[0], list):
#            images_list = images  # already batched
#        else:
#            images_list = [images]  # wrap single batch
#        if isinstance(system_messages, list):
#            system_list = system_messages
#        else:
#            system_list = [system_messages] * len(texts_list)
#        if not (len(texts_list) == len(images_list) == len(system_list)):
#            logger.error(
#                "texts, images, and system_messages must have the same batch length."
#            )
#            return []
#        batch_length = len(texts_list)
#        # below the text, images and system_message are assured to be lists of same length
#        logger.info("Please confirm the exact model is multimodal by yourself.")
#        # Prepare messages
#        batch_messages = [[] for _ in range(batch_length)]
#        for i in range(batch_length):
#            images_single = images_list[i]
#            text = texts_list[i]
#            if system_list[i] is not None:
#                batch_messages[i].append(SystemMessage(content=system_list[i]))
#            # Process images based on model type
#            _, base_model = ModelConfig.extract_provider_model(provider_api)
#            if base_model in [
#                "openai",
#                "gemini",
#                "deepseek",
#                "qwen",
#            ]:
#                # OpenAI format for multimodal
#                content = [{"type": "text", "text": text}]
#                for img in images_single:
#                    image_data = self._process_image(img)
#                    content.append(
#                        {
#                            "type": "image_url",
#                            "image_url": {
#                                "url": f"data:image/jpeg;base64,{image_data}"
#                            },
#                        }
#                    )
#
#                batch_messages[i].append(HumanMessage(content=content))
#            elif base_model == "claude":
#                # Anthropic format for multimodal
#                content = [{"type": "text", "text": text}]
#
#                for img in images_single:
#                    image_data = self._process_image(img)
#                    content.append(
#                        {
#                            "type": "image",
#                            "source": {
#                                "type": "base64",
#                                "media_type": "image/jpeg",
#                                "data": image_data,
#                            },
#                        }
#                    )
#
#                batch_messages[i].append(HumanMessage(content=content))
#
#            else:
#                logger.error("no implementation for %s", base_model)
#                return []
#
#        return self.chat_completion(
#            provider_api, model, batch_messages, stream=stream, **kwargs
#        )
#
#    def _process_image(self, image: str | Path | Image.Image) -> str:
#        """
#        Process an image into base64 format.
#
#        Args:
#            image: Image path, URL, or PIL Image object
#
#        Returns:
#            Base64-encoded image data
#        """
#        if isinstance(image, (str, Path)):
#            img_path = str(image)
#            if img_path.startswith(("http://", "https://")):
#                # For URLs, we need to download the image first
#                import requests
#
#                response = requests.get(img_path, timeout=15)
#                img = Image.open(io.BytesIO(response.content))
#            else:
#                img = Image.open(img_path)
#        else:
#            img = image
#
#        # Convert PIL Image to base64
#        buffered = io.BytesIO()
#        img.save(buffered, format="JPEG")
#        return base64.b64encode(buffered.getvalue()).decode()
