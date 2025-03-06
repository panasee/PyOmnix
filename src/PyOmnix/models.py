"""
This module provides a configuration manager for different model APIs.
It allows adding, removing, and retrieving API keys and URLs for various providers.
The configuration is stored in a JSON file and can be manually edited.
"""
import json
import os
import importlib
from functools import partial
from typing import Dict, Optional, Any, List
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from . import utils
from .omnix_logger import get_logger, setup_logger

logger = get_logger(__name__)

class ModelConfig:
    """
    Class to manage API keys and URLs for different model providers.
    Handles loading, saving, and retrieving configuration for various model APIs.
    Implements the Singleton pattern to ensure only one instance exists.
    """
    
    _instance: Optional['ModelAPIConfig'] = None
    
    def __new__(cls) -> 'ModelAPIConfig':
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
    
    def __init__(self):
        """
        Initialize the ModelAPIConfig with the default config file path.
        This will only run once for the singleton instance.
        """
        if getattr(self, '_initialized', False):
            return
            
        if utils.OMNIX_PATH is None:
            raise ValueError("OMNIX_PATH must be set to use ModelAPI")
        self.config_json = Path(f"{utils.OMNIX_PATH}/api_config.json")
        self.config: Dict[str, Dict[str, Any]] = {}
        self._load_config()
        self.setup_langsmith()
        self._initialized = True
        self.models = {}
    
    def _load_config(self) -> None:
        """Load the configuration from the JSON file if it exists."""
        if self.config_json.exists():
            with open(self.config_json, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            logger.info("No config file found at %s. Initialize empty configuration.", self.config_json)
            self.config = {}
            with open(self.config_json, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
    
    def setup_langsmith(self):
        """
        Setup LangSmith for tracing and monitoring.
        """
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = self.get_api_config("langsmith")[1]
        os.environ["LANGSMITH_API_KEY"] = self.get_api_config("langsmith")[0]
        os.environ["LANGSMITH_PROJECT"] = "pyomnix"

    def setup_models(self, models_name: str | list[str]):
        """
        Setup a specific model for tracing and monitoring.
        """
        model_module_dict = {
            "chatgpt": ["langchain_openai", "ChatOpenAI"],
            "gemini": ["langchain_google_genai", "ChatGoogleGenerativeAI"],
            "claude": ["langchain_anthropic", "ChatAnthropic"],
            "deepseek": ["langchain_deepseek", "ChatDeepSeek"],
            "qwen": ["langchain_community.chat_models.tongyi", "ChatTongyi"]
        }
        if isinstance(models_name, str):
            models_name = [models_name]
        for model in models_name:
            logger.validate(model in model_module_dict, "Model %s not found in model_module_dict.", model)
            if model in self.models:
                logger.info("Model %s already exists in models.", model)
                continue
            module = importlib.import_module(model_module_dict[model][0])
            self.models[model] = partial(getattr(module, model_module_dict[model][1]), api_key=self.get_api_config(model)[0], base_url=self.get_api_config(model)[1])
            logger.info("Model %s initialized successfully.", model)
        return {model: self.models[model] for model in models_name}
        ##TODO: add interface for local models
    
    def save_config(self) -> None:
        """Save the current configuration to the JSON file."""
        with open(self.config_json, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)
    
    def set_api_config(self, provider: str, *, api_key: Optional[str], api_url: Optional[str]) -> None:
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
            self.config[provider]['api_key'] = api_key
        if api_url is not None:
            self.config[provider]['api_url'] = api_url
        self.save_config()
    
    def get_api_config(self, provider: str) -> Optional[tuple[str, str]]:
        """
        Get the API key for a specific provider.
        
        Args:
            provider: The model provider (e.g., 'openai', 'google', 'anthropic')
            
        Returns:
            The API key if found, None otherwise
        """
        if provider in self.config and 'api_key' in self.config[provider] and 'api_url' in self.config[provider]:
            return self.config[provider]['api_key'], self.config[provider]['api_url']
        else:
            logger.error("Uncomplete API config found for provider %s", provider)
            return None
    
    def list_providers(self) -> List[str]:
        """
        List all configured providers.
        
        Returns:
            List of provider names
        """
        return list(self.config.keys())
    
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

class Models():
    """
    class for calling model APIs.
    """
    def __init__(self, models_name: str | list[str]):
        self.logger = setup_logger(name=self.__class__.__name__, log_file=Path(f"{utils.OMNIX_PATH}/logs/chats.log"))
        self.models = ModelConfig().setup_models(models_name)

    def chat_completion(self, provider: str, model: str, messages: list[BaseMessage], *, stream: bool = False, **kwargs) -> dict:
        """
        Send a chat completion request to the model.
        """
        chat = self.models[provider](model=model, **kwargs)
        if stream:
            response = chat.stream(messages)
        else:
            response = chat.invoke(messages)
        return response

    def predict_next(self, past_array: list, model: str, **kwargs) -> dict:
        """
        Predict the next message in the conversation.
        """
