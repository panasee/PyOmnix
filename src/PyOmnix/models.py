"""
This module provides a configuration manager for different model APIs.
It allows adding, removing, and retrieving API keys and URLs for various providers.
The configuration is stored in a JSON file and can be manually edited.
"""
import json
import os
import importlib
import io
import base64
from functools import partial
from typing import Dict, Optional, Any, List, Generator
from pathlib import Path
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage, FunctionMessage, ChatMessage
from .omnix_logger import get_logger, setup_logger
from .consts import OMNIX_PATH, LOG_FILE_PATH

logger = get_logger(__name__)

class ModelConfig:
    """
    Class to manage API keys and URLs for different model providers.
    Handles loading, saving, and retrieving configuration for various model APIs.
    Implements the Singleton pattern to ensure only one instance exists.
    """
    
    _instance: Optional['ModelConfig'] = None
    
    def __new__(cls) -> 'ModelConfig':
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
            
        if OMNIX_PATH is None:
            raise ValueError("OMNIX_PATH must be set to use ModelAPI")
        self.config_json = Path(f"{OMNIX_PATH}/api_config.json")
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

    def setup_models(self, models_name: str | list[str] = "deepseek"):
        """
        Setup providers and model apis (the api is a interface of langchain, not the real model called, actually deepseek/openai api can be used for most models). Indicate the provider before the model name if used. (e.g. "siliconflow-deepseek")
        """
        model_module_dict = {
            "openai": ["langchain_openai", "ChatOpenAI"],
            "gemini": ["langchain_google_genai", "ChatGoogleGenerativeAI"],
            "claude": ["langchain_anthropic", "ChatAnthropic"],
            "deepseek": ["langchain_deepseek", "ChatDeepSeek"],
            "qwen": ["langchain_community.chat_models.tongyi", "ChatTongyi"]
        }
        if isinstance(models_name, str):
            models_name = [models_name]

        for model_full in models_name:
            provider, model = self.get_provider_model(model_full)

            logger.validate(model in model_module_dict, f"Model {model} not found in model_module_dict.")
            if model_full in self.models:
                logger.info("Model %s already exists in models.", model)
                continue
            module = importlib.import_module(model_module_dict[model][0])
            if model == "deepseek":
                self.models[model_full] = partial(
                    getattr(module, model_module_dict[model][1]),
                    api_key=self.get_api_config(provider)[0],
                    api_base=self.get_api_config(provider)[1])
            else:
                self.models[model_full] = partial(
                    getattr(module, model_module_dict[model][1]),
                    api_key=self.get_api_config(provider)[0],
                    base_url=self.get_api_config(provider)[1])
            # base_url and api_base are the same, for different apis
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

    def check_provider_models(self, provider: str) -> List[str]:
        """
        Check the models supported by a specific provider.
        """
        if provider in self.config and 'models' in self.config[provider]:
            return self.config[provider]['models']
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

class Models():
    """
    class for calling model APIs.
    """
    def __init__(self, models_name: str | list[str]):
        self.logger = setup_logger(name=self.__class__.__name__, log_file=Path(f"{LOG_FILE_PATH}/models.log"))
        self.models = ModelConfig().setup_models(models_name)

    def _get_response(self, chat_model, basemessages: list[list[BaseMessage]], stream: bool = False, invoke_config: Optional[dict[str, Any]] = None, **kwargs):
        """
        Get the response from the model with BaseMessage list.
        invoke_config will only be used for invoke method
        """
        if len(basemessages) > 1:
            logger.info("Batching messages for %s", chat_model)
            logger.validate(not stream, "Batch does not support streaming.")
            response = chat_model.batch(basemessages)
        else:
            response = chat_model.invoke(basemessages[0], config=invoke_config) if not stream else chat_model.stream(basemessages[0])
        return response

    def chat_completion(self,
                        full_name: str,
                        model: str,
                        messages: list[dict[str, str] | BaseMessage] | list[list[dict[str, str] | BaseMessage]],
                        *,
                        stream: bool = False,
#                        temperature: float = 0.6,
#                        max_tokens: Optional[int] = None,
#                        timeout: Optional[int] = None,
#                        max_retries: int = 2,
                        invoke_config: Optional[dict[str, Any]] = None,
                        **kwargs
                        ) -> dict | Generator[dict, None, None]:
        """
        Send a chat completion request to the model.
        
        Args:
            full_name(str): The full name of the model to use
            model(str): The specific model to use
            messages(list[dict[str, str] | BaseMessage] | list[list[dict[str, str] | BaseMessage]]): The messages to send to the model or a batch of messages
            stream(bool): Whether to stream the response
            invoke_config(dict[str, Any]): Additional parameters to pass to the model (no stream and no batch)
            **kwargs: Additional parameters to pass to the model
        """
        chat = self.models[full_name](model=model, **kwargs)
        if not isinstance(messages[0], list):
            messages = [messages]
            return self.chat_completion(full_name, model, messages, stream=stream, invoke_config=invoke_config, **kwargs)
        elif all(isinstance(msg, BaseMessage) for msg_lst in messages for msg in msg_lst):
            return self._get_response(chat, messages, stream, invoke_config)

        lc_batch_messages = []
        for msg_item in messages:
            lc_messages = []
            for msg in msg_item:
                if isinstance(msg, BaseMessage):
                    lc_messages.append(msg)
                    continue
                logger.validate(isinstance(msg, dict), "Message must be a dictionary or BaseMessage.")
                role = msg["role"].lower()
                content = msg["content"]
                match role:
                    case "system":
                        lc_messages.append(SystemMessage(content=content))
                    case "user":
                        lc_messages.append(HumanMessage(content=content))
                    case "assistant":
                        lc_messages.append(AIMessage(content=content))
                    case "tool":
                        lc_messages.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
                    case "function":
                        lc_messages.append(FunctionMessage(content=content, name=msg.get("name", "")))
                    case _:
                        lc_messages.append(ChatMessage(content=content, role=role))
            lc_batch_messages.append(lc_messages)

        return self._get_response(chat, lc_batch_messages, stream, invoke_config)

    def chat_with_images(
        self,
        full_name: str,
        model: str,
        texts: str | list[str | BaseMessage],
        images: list[str | Path | Image.Image] | list[list[str | Path | Image.Image]],
        system_messages: Optional[str | list[str | BaseMessage]] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Send a multimodal chat request with text and images.
        
        Args:
            full_name: The full name of the model to use
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
        if not (isinstance(texts,list) == isinstance(images[0], list) == isinstance(system_messages, list)):
            logger.error("text, images and system_message must be consistent on if_batch and the batch length.")
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
            if ModelConfig.get_provider_model(full_name)[1] in ["openai", "google", "deepseek"]:
                # OpenAI format for multimodal
                content = [{"type": "text", "text": text}]
                for img in images_single:
                    image_data = self._process_image(img)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    })

                batch_messages[i].append(HumanMessage(content=content))

            elif ModelConfig.get_provider_model(full_name)[1] == "anthropic":
                # Anthropic format for multimodal
                content = [{"type": "text", "text": text}]

                for img in images_single:
                    image_data = self._process_image(img)
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    })

                batch_messages[i].append(HumanMessage(content=content))

            else:
                logger.error("no implementation for %s", ModelConfig.get_provider_model(full_name)[1])
                return
        
        return self.chat_completion(full_name, model, batch_messages, stream=stream, **kwargs)
    
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
            if img_path.startswith(('http://', 'https://')):
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
