import os
from dataclasses import dataclass

from dotenv import load_dotenv

from pyomnix.consts import ERROR_ICON, OMNIX_PATH, SUCCESS_ICON, WAIT_ICON
from pyomnix.omnix_logger import get_logger

from .llm_clients import LLMClient

# 设置日志记录
logger = get_logger("openrouter_config")


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


project_root = OMNIX_PATH / "financial"
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} loaded env: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} .env file not found: {env_path}")

provider_api = os.getenv("PROVIDER_API", "deepseek")
model = os.getenv("MODEL", "deepseek-chat")
logger.info(f"provider_api: {provider_api}, model: {model}")


client = LLMClient(model, provider_api)
logger.info(f"{SUCCESS_ICON} 客户端初始化成功")


# TODO: contents type unknown
def generate_content_with_retry(model, contents, config=None):
    """带重试机制的内容生成函数"""
    logger.info(f"{WAIT_ICON} 正在调用 API...")
    logger.debug(f"请求内容: {contents}")
    logger.debug(f"请求配置: {config}")

    response = client.get_completion(contents, model=model, config=config)

    logger.info(f"{SUCCESS_ICON} API 调用成功")
    logger.debug(f"响应内容: {response.text[:500]}...")
    return response


def get_chat_completion(
    messages,
    model_in=None,
    max_retries=3,
    initial_retry_delay=1,
    client_type="auto",
    api_key=None,
    base_url=None,
):
    """
    获取聊天完成结果，包含重试逻辑

    Args:
        messages: 消息列表，OpenAI 格式
        model: 模型名称（可选）
        max_retries: 最大重试次数
        initial_retry_delay: 初始重试延迟（秒）
        client_type: 客户端类型 ("auto", "gemini", "openai_compatible")
        api_key: API 密钥（可选，仅用于 OpenAI Compatible API）
        base_url: API 基础 URL（可选，仅用于 OpenAI Compatible API）

    Returns:
        str: 模型回答内容或 None（如果出错）
    """
    try:
        if client_type != "auto":
            if client_type == "gemini":
                model_in = model_in if model_in is not None else "gemini-1.5-flash"
                provider_api_in = "gemini"
            else:
                model_in = model_in if model_in is not None else "deepseek-chat"
                provider_api_in = "deepseek"

            client = LLMClient(model_in, provider_api_in)
        else:
            client = LLMClient(model, provider_api)

        # 获取回答
        return client.get_completion(
            messages=messages,
            max_retries=max_retries,
        )
    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {e!s}")
        return None
