"""
This module provides workflows for using language models.
Some is borrowed from GitHub repositories:
https://github.com/langchain-ai/langgraph-swarm-py
"""

import re
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, validate_call

from pyomnix.model_interface.models import ChatMessages, ModelConfig
from pyomnix.omnix_logger import get_logger

WHITESPACE_RE = re.compile(r"\s+")
METADATA_KEY_HANDOFF_DESTINATION = "__handoff_destination"


logger = get_logger(__name__)


class Conversation:
    """State maintained between steps of the chat workflow."""

    @validate_call(max_args=10)
    def __init__(
        self,
        *,
        ans_model: list[str, str],
        que_model: list[str, str],
        supervisor_model: list[str, str] | None = None,
        trim_model: list[str, str] | None = None,
        begin_message: str,
        que_sys_prompt: str,
        que_prompt: str,
        ans_sys_prompt: str,
        supervisor_sys_prompt: str = "",
        supervisor_prompt: str = "",
        topic_prompt: str = "",
        max_tokens: int = 51000,
    ):
        """Initialize the chat state.

        Args:
            ans_model: The model to use for the answer agent, [provider_api, model_name]
            que_model: The model to use for the question raising agent, [provider_api, model_name]
            supervisor_model: The model to use for the supervisor agent, [provider_api, model_name]
            trim_model: The model to use for the trimming/summarizing agent, [provider_api, model_name]
            current_tokens: The current token count
            max_tokens: The maximum tokens triggering summarization
            begin_message: The first human message to start the conversation
            que_sys_prompt: The system prompt for the question raising agent
            que_prompt: The human prompt for the question raising agent
            ans_sys_prompt: The system prompt for the answer agent
            supervisor_sys_prompt: The prompt for the supervisor agent
            supervisor_prompt: The human prompt for the supervisor agent
            topic_prompt: The prompt for the topic of conversation(used for the first message and also for the supervisor agent)
        """
        self._load_models(ans_model, que_model, supervisor_model, trim_model)
        self.max_tokens = max_tokens
        self.begin_message = begin_message
        self.que_sys_prompt = que_sys_prompt
        self.que_prompt = que_prompt
        self.ans_sys_prompt = ans_sys_prompt
        self.supervisor_sys_prompt = supervisor_sys_prompt
        self.supervisor_prompt = supervisor_prompt
        self.topic_prompt = f"Topic: {topic_prompt}.\n"
        self.token_count = 0
        self._initialize_messages()

    def _load_models(
        self,
        ans_model: list[str, str],
        que_model: list[str, str],
        supervisor_model: list[str, str] | None,
        trim_model: list[str, str] | None,
    ) -> None:
        """generate a list of models to use"""
        self.models = {
            "ans": ModelConfig().setup_models(ans_model[0])[ans_model[0]](model=ans_model[1]),
            "que": ModelConfig().setup_models(que_model[0])[que_model[0]](model=que_model[1]),
            "supervisor": ModelConfig().setup_models(supervisor_model[0])[supervisor_model[0]](
                model=supervisor_model[1]
            )
            if supervisor_model is not None
            else None,
            "trim": ModelConfig().setup_models(trim_model[0])[trim_model[0]](model=trim_model[1])
            if trim_model is not None
            else None,
        }

    def _initialize_messages(self) -> None:
        """initialize the chat messages, only executed once at the beginning of the conversation"""
        self.chat_messages = ChatMessages(
            messages=[
                SystemMessage(content=self.topic_prompt + self.ans_sys_prompt),
                HumanMessage(content=self.begin_message),
            ],
            file_sync=True,
        )

    def set_temperature(
        self, role: Literal["ans", "que", "supervisor"], temperature: float
    ) -> None:
        """
        set the temperature for the given role model
        Args:
            role: The role of the model to set the temperature for
            temperature: The temperature to set for the model
        """
        if role == "ans":
            self.models["ans"].temperature = temperature
        elif role == "que":
            self.models["que"].temperature = temperature
        elif role == "supervisor":
            if self.models["supervisor"] is not None:
                self.models["supervisor"].temperature = temperature
            else:
                logger.warning("No supervisor model found")

    def set_max_tokens(self, role: Literal["ans", "que", "supervisor"], max_tokens: int) -> None:
        """
        set the max tokens for the given role model
        Args:
            role: The role of the model to set the max tokens for
            max_tokens: The max tokens to set for the model
        """
        if role == "ans":
            self.models["ans"].max_tokens = max_tokens
        elif role == "que":
            self.models["que"].max_tokens = max_tokens
        elif role == "supervisor":
            if self.models["supervisor"] is not None:
                self.models["supervisor"].max_tokens = max_tokens
            else:
                logger.warning("No supervisor model found")

    def answer(self, message: str = "", *, with_topic: bool = False) -> None:
        """
        get the answer from the answer model.
        (use this for general manual request)
        If the last message is a AIMessage, it will use it to replace the last HumanMessage and request answer.
        Args:
            with_topic: Whether to include the topic in the answer system prompt
        """
        if with_topic:
            self.chat_messages[0] = SystemMessage(content=self.topic_prompt + self.ans_sys_prompt)
        else:
            self.chat_messages[0] = SystemMessage(content=self.ans_sys_prompt)

        if isinstance(self.chat_messages.messages[-1], HumanMessage):
            self.chat_messages.messages[-1] = HumanMessage(
                content=self.chat_messages.messages[-1].content + "." + message
            )
        elif isinstance(self.chat_messages.messages[-1], AIMessage):
            self.chat_messages.messages[-2] = HumanMessage(
                content=self.chat_messages.messages[-1].content + "." + message
            )
            self.chat_messages.drop_last()
        else:
            logger.error(
                "The last message is not a HumanMessage or AIMessage, check the chat_messages"
            )
            return
        self.chat_messages.request_response(self.models["ans"])

    def ask(self, with_topic: bool = False) -> None:
        """
        let the question model ask a question. It's equivalent to automatically changing system prompt to questioner and adding a HumanMessage(urge for a question) and then call answer().
        """
        if with_topic:
            self.chat_messages[0] = SystemMessage(content=self.topic_prompt + self.que_sys_prompt)
        else:
            self.chat_messages[0] = SystemMessage(content=self.que_sys_prompt)

        if isinstance(self.chat_messages.messages[-1], AIMessage):
            self.chat_messages += HumanMessage(content=self.que_prompt)
        elif isinstance(self.chat_messages.messages[-1], HumanMessage):
            logger.warning(
                "Unexpected Behavior: The last message is a HumanMessage, adding the question prompt to it"
            )
            self.chat_messages.messages[-1] = HumanMessage(
                content=self.chat_messages.messages[-1].content + self.que_prompt
            )
        else:
            logger.error(
                "The last message is not a AIMessage or HumanMessage, check the chat_messages"
            )
            return
        self.chat_messages.request_response(self.models["que"])

    def trim_conversation(self, assessment: bool = False) -> None:
        """
        trim the conversation to the last 2 turns.
        """
        if self.models["trim"] is None:
            logger.debug("No trimming model found, using the answer model to trim")
            self.chat_messages.auto_trim(
                self.models["ans"],
                token_limit=self.max_tokens,
                summarize_tokens=3000,
                temperature=0.3,
                preserve_conversation_turns=3,
            )
        else:
            self.chat_messages.auto_trim(
                self.models["trim"],
                token_limit=self.max_tokens,
                summarize_tokens=3000,
                temperature=0.3,
                preserve_conversation_turns=3,
            )

    def assess_topic(self) -> str:
        """
        assess the topic of the conversation. If the final message is not AIMessage, it will drop all messages behind the last AIMessage.
        Returns:
            str: The changing prompt if necessary, otherwise an empty string
        """

        class AssessTopicSchema(BaseModel):
            """Always use this tool to structure your response to the user."""

            topic: Annotated[str, "The current topic of the conversation"]
            deviation: Annotated[str, "The deviation of the topic from the original topic"]
            necessity: Annotated[
                float,
                Field(
                    description="The necessity of changing the topic back, from 0 to 1", ge=0, le=1
                ),
            ]
            prompt: Annotated[str, "The prompt to change the topic back"]

        self.chat_messages.messages[0] = SystemMessage(
            content=self.topic_prompt + self.supervisor_sys_prompt
        )
        while not isinstance(self.chat_messages.messages[-1], AIMessage):
            self.chat_messages.drop_last()
        assessment = self.chat_messages.request_response(
            self.models["supervisor"], schema=AssessTopicSchema
        )
        if assessment.content.necessity > 0.5:
            return f"The expected topic is: {self.topic_prompt}.\n But the deviation is {assessment.content.deviation}.\n {assessment.content.prompt}"
        else:
            return ""
