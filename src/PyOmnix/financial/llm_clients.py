import os

from langchain_core.messages import BaseMessage

from pyomnix.consts import ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from pyomnix.model_interface.models import RawModels
from pyomnix.omnix_logger import get_logger

# 设置日志记录
logger = get_logger("llm_clients")


class LLMClient(RawModels):
    """compatible layer"""

    def __init__(self, model=None, provider_api=None):
        """set corresponding provider_api and model"""
        self.provider_api = provider_api if provider_api is not None else "deepseek"
        self.model = model if model is not None else "deepseek-chat"
        print("Provider and Model names are not correctly defined")

    def get_completion(
        self, messages: list[BaseMessage | dict[str, str]], max_retries=3, **kwargs
    ):
        """获取聊天完成结果，包含重试逻辑"""
        try:
            logger.info(f"{WAIT_ICON} 使用 {self.provider_api} 模型: {self.model}")
            logger.debug(f"消息内容: {messages}")

            model_selected = kwargs.get("model", self.model)
            if model_selected is None:
                model_selected = self.model
            response = self.chat_completion(
                self.provider_api,
                model_selected,
                messages,
                max_retries=max_retries,
                **kwargs,
            )[0]

            logger.debug(f"API 响应: {response.content[:500]}...")
            logger.info(f"{SUCCESS_ICON} 成功获取响应")

            return response.content

        except Exception as e:
            logger.error(f"{ERROR_ICON} get_completion 发生错误: {e!s}")
            return None
