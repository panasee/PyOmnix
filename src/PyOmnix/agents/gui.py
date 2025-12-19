"""
Reflex GUI for Dual Agents Workflow.

A chatroom-style interface where multiple agents communicate with each other,
with human-in-the-loop support for reviewing and controlling the conversation flow.

Usage:
    reflex run
"""

from datetime import datetime
from typing import Any, Literal

import reflex as rx
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from pyomnix.agents.graphs import GraphSession, build_self_correction_graph
from pyomnix.agents.models_settings import ModelConfig
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Message Model
# =============================================================================


class ChatMessage(rx.Base):
    """Represents a single chat message in the UI."""

    id: str
    role: Literal["user", "chat", "critic", "system"]
    content: str
    timestamp: str
    is_streaming: bool = False


# =============================================================================
# Agent Colors and Styling
# =============================================================================


AGENT_COLORS = {
    "user": {
        "bg": "var(--blue-3)",
        "border": "var(--blue-6)",
        "text": "var(--blue-11)",
        "avatar_bg": "var(--blue-9)",
    },
    "chat": {
        "bg": "var(--green-3)",
        "border": "var(--green-6)",
        "text": "var(--green-11)",
        "avatar_bg": "var(--green-9)",
    },
    "critic": {
        "bg": "var(--orange-3)",
        "border": "var(--orange-6)",
        "text": "var(--orange-11)",
        "avatar_bg": "var(--orange-9)",
    },
    "system": {
        "bg": "var(--gray-3)",
        "border": "var(--gray-6)",
        "text": "var(--gray-11)",
        "avatar_bg": "var(--gray-9)",
    },
}

AGENT_LABELS = {
    "user": "You",
    "chat": "Debater",
    "critic": "Critic",
    "system": "System",
}

AGENT_ICONS = {
    "user": "user",
    "chat": "message-circle",
    "critic": "shield-alert",
    "system": "info",
}


# =============================================================================
# Application State
# =============================================================================


class AgentState(rx.State):
    """Main application state for the dual agents GUI."""

    # Chat messages - Reflex state vars are reactive, not ClassVar
    messages: list[ChatMessage] = []  # noqa: RUF012

    # Input state
    user_input: str = ""
    is_processing: bool = False

    # Human review state
    awaiting_review: bool = False
    review_context: str = ""

    # Settings
    model_name: str = "deepseek"
    llm_model: str = "deepseek-chat"
    temperature: float = 0.7
    max_iterations: int = 5

    # Session state
    thread_id: str = "gui-session-001"
    current_iteration: int = 0

    # Graph session (stored separately to avoid serialization issues)
    _graph_session: GraphSession | None = None

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%H:%M:%S")

    def _add_message(
        self,
        role: Literal["user", "chat", "critic", "system"],
        content: str,
        is_streaming: bool = False,
    ) -> str:
        """Add a new message and return its ID."""
        msg_id = f"{role}-{len(self.messages)}-{datetime.now().timestamp()}"
        self.messages = self.messages + [
            ChatMessage(
                id=msg_id,
                role=role,
                content=content,
                timestamp=self._get_timestamp(),
                is_streaming=is_streaming,
            )
        ]
        return msg_id

    def _update_message(self, msg_id: str, content: str, is_streaming: bool = False):
        """Update an existing message by ID."""
        updated_messages = []
        for msg in self.messages:
            if msg.id == msg_id:
                updated_messages.append(
                    ChatMessage(
                        id=msg.id,
                        role=msg.role,
                        content=content,
                        timestamp=msg.timestamp,
                        is_streaming=is_streaming,
                    )
                )
            else:
                updated_messages.append(msg)
        self.messages = updated_messages

    def set_user_input(self, value: str):
        """Update user input field."""
        self.user_input = value

    def set_temperature(self, value: list[float]):
        """Update temperature setting."""
        if value:
            self.temperature = value[0]

    def set_max_iterations(self, value: str):
        """Update max iterations setting."""
        try:
            self.max_iterations = int(value)
        except ValueError:
            pass

    def clear_chat(self):
        """Clear all messages and reset state."""
        self.messages = []
        self.current_iteration = 0
        self.awaiting_review = False
        self.review_context = ""
        self._add_message("system", "Chat cleared. Ready for new conversation.")

    async def _initialize_graph(self) -> GraphSession:
        """Initialize the graph session."""
        model_factory = ModelConfig()
        models = model_factory.setup_model_factory(self.model_name)
        model = models[self.model_name].with_config(
            llm_model=self.llm_model,
            llm_temperature=self.temperature,
        )

        workflow = build_self_correction_graph(model)
        graph = workflow.compile(checkpointer=MemorySaver())

        return GraphSession(graph, thread_id=self.thread_id)

    async def send_message(self):
        """Send user message and start the agent workflow."""
        if not self.user_input.strip() or self.is_processing:
            return

        user_message = self.user_input.strip()
        self.user_input = ""
        self.is_processing = True
        self.current_iteration = 0

        # Add user message
        self._add_message("user", user_message)

        try:
            # Initialize graph session
            graph_session = await self._initialize_graph()

            # Start the workflow
            input_state = {"messages": [HumanMessage(content=user_message)]}

            # Run the first iteration
            await self._run_iteration(graph_session, input_state)

        except Exception as e:
            logger.error("Error in agent workflow: %s", e)
            self._add_message("system", f"Error: {str(e)}")
            self.is_processing = False

    async def _run_iteration(
        self,
        graph_session: GraphSession,
        input_data: Any,
    ):
        """Run a single iteration of the agent workflow."""
        self.current_iteration += 1

        try:
            # Stream the response
            chat_msg_id = None
            critic_msg_id = None

            async for event in graph_session.astream(input_data):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        # Human review interrupt
                        interrupt_data = node_output[0].value if node_output else {}
                        self.awaiting_review = True
                        self.review_context = interrupt_data.get(
                            "critic_response", "Critic has provided feedback."
                        )
                        self._graph_session = graph_session
                        self.is_processing = False
                        return

                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if not hasattr(msg, "content") or not msg.content:
                            continue

                        content = msg.content
                        msg_name = getattr(msg, "name", None)

                        if msg_name == "chat" or node_name == "chat":
                            if chat_msg_id is None:
                                chat_msg_id = self._add_message(
                                    "chat", content, is_streaming=True
                                )
                            else:
                                self._update_message(
                                    chat_msg_id, content, is_streaming=True
                                )
                        elif msg_name == "critic" or node_name == "critic":
                            if critic_msg_id is None:
                                critic_msg_id = self._add_message(
                                    "critic", content, is_streaming=True
                                )
                            else:
                                self._update_message(
                                    critic_msg_id, content, is_streaming=True
                                )

            # Mark messages as complete
            if chat_msg_id:
                for msg in self.messages:
                    if msg.id == chat_msg_id:
                        self._update_message(chat_msg_id, msg.content, is_streaming=False)
            if critic_msg_id:
                for msg in self.messages:
                    if msg.id == critic_msg_id:
                        self._update_message(critic_msg_id, msg.content, is_streaming=False)

        except Exception as e:
            logger.error("Error in iteration %d: %s", self.current_iteration, e)
            self._add_message("system", f"Error in iteration {self.current_iteration}: {str(e)}")
            self.is_processing = False

    async def continue_workflow(self):
        """Continue the agent workflow after human review."""
        if not self.awaiting_review or self._graph_session is None:
            return

        self.awaiting_review = False
        self.is_processing = True
        self._add_message("system", "Continuing the debate...")

        if self.current_iteration >= self.max_iterations:
            self._add_message(
                "system",
                f"Maximum iterations ({self.max_iterations}) reached. Ending workflow.",
            )
            self.is_processing = False
            return

        try:
            await self._run_iteration(
                self._graph_session,
                Command(resume="continue"),
            )
        except Exception as e:
            logger.error("Error continuing workflow: %s", e)
            self._add_message("system", f"Error: {str(e)}")
            self.is_processing = False

    async def end_workflow(self):
        """End the agent workflow after human review."""
        if not self.awaiting_review:
            return

        self.awaiting_review = False
        self._add_message("system", "Workflow ended by user.")
        self.is_processing = False
        self._graph_session = None


# =============================================================================
# UI Components
# =============================================================================


def agent_avatar(role: str) -> rx.Component:
    """Render an agent avatar with icon."""
    colors = AGENT_COLORS.get(role, AGENT_COLORS["system"])
    icon = AGENT_ICONS.get(role, "circle")

    return rx.box(
        rx.icon(icon, size=16, color="white"),
        width="32px",
        height="32px",
        border_radius="50%",
        background=colors["avatar_bg"],
        display="flex",
        align_items="center",
        justify_content="center",
        flex_shrink="0",
    )


def message_bubble(message: ChatMessage) -> rx.Component:
    """Render a single message bubble."""
    colors = AGENT_COLORS.get(message.role, AGENT_COLORS["system"])
    label = AGENT_LABELS.get(message.role, message.role.capitalize())
    is_user = message.role == "user"

    return rx.box(
        rx.hstack(
            rx.cond(
                is_user,
                rx.fragment(),
                agent_avatar(message.role),
            ),
            rx.vstack(
                rx.hstack(
                    rx.text(
                        label,
                        font_weight="600",
                        font_size="0.85em",
                        color=colors["text"],
                    ),
                    rx.text(
                        message.timestamp,
                        font_size="0.75em",
                        color="var(--gray-9)",
                    ),
                    rx.cond(
                        message.is_streaming,
                        rx.spinner(size="1"),
                        rx.fragment(),
                    ),
                    spacing="2",
                    align_items="center",
                ),
                rx.box(
                    rx.markdown(
                        message.content,
                        component_map={
                            "p": lambda text: rx.text(text, margin="0"),
                        },
                    ),
                    background=colors["bg"],
                    border=f"1px solid {colors['border']}",
                    border_radius="12px",
                    padding="12px 16px",
                    max_width="80%",
                ),
                align_items="flex-start" if not is_user else "flex-end",
                spacing="1",
            ),
            rx.cond(
                is_user,
                agent_avatar(message.role),
                rx.fragment(),
            ),
            width="100%",
            justify_content="flex-start" if not is_user else "flex-end",
            spacing="3",
        ),
        width="100%",
        padding_y="8px",
    )


def chat_messages() -> rx.Component:
    """Render the chat messages container."""
    return rx.box(
        rx.foreach(AgentState.messages, message_bubble),
        width="100%",
        flex="1",
        overflow_y="auto",
        padding="16px",
        id="chat-container",
    )


def human_review_panel() -> rx.Component:
    """Render the human review decision panel."""
    return rx.cond(
        AgentState.awaiting_review,
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.icon("alert-circle", color="var(--orange-9)"),
                    rx.text(
                        "Human Review Required",
                        font_weight="600",
                        color="var(--orange-11)",
                    ),
                    spacing="2",
                    align_items="center",
                ),
                rx.text(
                    "The critic has provided feedback. "
                    "Would you like to continue the debate or end the workflow?",
                    color="var(--gray-11)",
                    font_size="0.9em",
                ),
                rx.hstack(
                    rx.button(
                        rx.icon("play", size=16),
                        "Continue Debate",
                        color_scheme="green",
                        on_click=AgentState.continue_workflow,
                    ),
                    rx.button(
                        rx.icon("square", size=16),
                        "End Workflow",
                        color_scheme="red",
                        variant="outline",
                        on_click=AgentState.end_workflow,
                    ),
                    spacing="3",
                ),
                rx.text(
                    rx.text.span("Iteration: ", font_weight="500"),
                    rx.text.span(AgentState.current_iteration),
                    rx.text.span(" / "),
                    rx.text.span(AgentState.max_iterations),
                    font_size="0.85em",
                    color="var(--gray-10)",
                ),
                spacing="3",
                align_items="flex-start",
            ),
            background="var(--orange-2)",
            border="1px solid var(--orange-6)",
            border_radius="12px",
            padding="16px",
            margin="16px",
        ),
        rx.fragment(),
    )


def input_area() -> rx.Component:
    """Render the message input area."""
    return rx.box(
        rx.hstack(
            rx.input(
                value=AgentState.user_input,
                placeholder="Type your message to start the debate...",
                on_change=AgentState.set_user_input,
                on_key_down=rx.cond(
                    rx.key_down("Enter"),
                    AgentState.send_message,
                    rx.noop(),
                ),
                disabled=AgentState.is_processing | AgentState.awaiting_review,
                flex="1",
                size="3",
            ),
            rx.button(
                rx.cond(
                    AgentState.is_processing,
                    rx.spinner(size="2"),
                    rx.icon("send", size=18),
                ),
                on_click=AgentState.send_message,
                disabled=AgentState.is_processing | AgentState.awaiting_review,
                size="3",
                color_scheme="blue",
            ),
            spacing="3",
            width="100%",
        ),
        padding="16px",
        border_top="1px solid var(--gray-5)",
        background="var(--gray-1)",
    )


def settings_sidebar() -> rx.Component:
    """Render the settings sidebar."""
    return rx.box(
        rx.vstack(
            rx.heading("Settings", size="4", margin_bottom="16px"),
            # Temperature slider
            rx.vstack(
                rx.hstack(
                    rx.text("Temperature", font_weight="500"),
                    rx.text(
                        AgentState.temperature,
                        color="var(--gray-9)",
                        font_size="0.9em",
                    ),
                    justify_content="space-between",
                    width="100%",
                ),
                rx.slider(
                    default_value=[0.7],
                    min=0,
                    max=2,
                    step=0.1,
                    on_value_commit=AgentState.set_temperature,
                    width="100%",
                ),
                width="100%",
                spacing="2",
            ),
            # Max iterations
            rx.vstack(
                rx.text("Max Iterations", font_weight="500"),
                rx.input(
                    # Reflex reactive state method
                    value=AgentState.max_iterations.to_string(),  # type: ignore[attr-defined]
                    on_change=AgentState.set_max_iterations,
                    type="number",
                    width="100%",
                ),
                width="100%",
                spacing="2",
                align_items="flex-start",
            ),
            rx.divider(margin_y="16px"),
            # Status
            rx.vstack(
                rx.heading("Status", size="3"),
                rx.hstack(
                    rx.box(
                        width="8px",
                        height="8px",
                        border_radius="50%",
                        background=rx.cond(
                            AgentState.is_processing,
                            "var(--green-9)",
                            "var(--gray-6)",
                        ),
                    ),
                    rx.text(
                        rx.cond(
                            AgentState.is_processing,
                            "Processing...",
                            rx.cond(
                                AgentState.awaiting_review,
                                "Awaiting Review",
                                "Idle",
                            ),
                        ),
                        font_size="0.9em",
                    ),
                    spacing="2",
                    align_items="center",
                ),
                rx.text(
                    rx.text.span("Messages: "),
                    # Reflex reactive state method
                    rx.text.span(AgentState.messages.length()),  # type: ignore[attr-defined]
                    font_size="0.85em",
                    color="var(--gray-10)",
                ),
                width="100%",
                spacing="2",
                align_items="flex-start",
            ),
            rx.divider(margin_y="16px"),
            # Actions
            rx.button(
                rx.icon("trash-2", size=16),
                "Clear Chat",
                variant="outline",
                color_scheme="red",
                on_click=AgentState.clear_chat,
                width="100%",
            ),
            width="100%",
            spacing="4",
            align_items="stretch",
        ),
        width="280px",
        padding="20px",
        border_left="1px solid var(--gray-5)",
        background="var(--gray-1)",
        height="100%",
        overflow_y="auto",
    )


def header() -> rx.Component:
    """Render the application header."""
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.icon("messages-square", size=28, color="var(--blue-9)"),
                rx.heading("Dual Agents Debate", size="5"),
                spacing="3",
                align_items="center",
            ),
            rx.hstack(
                rx.badge(
                    rx.icon("message-circle", size=12),
                    "Debater",
                    color_scheme="green",
                    variant="soft",
                ),
                rx.badge(
                    rx.icon("shield-alert", size=12),
                    "Critic",
                    color_scheme="orange",
                    variant="soft",
                ),
                spacing="2",
            ),
            justify_content="space-between",
            width="100%",
            align_items="center",
        ),
        padding="16px 24px",
        border_bottom="1px solid var(--gray-5)",
        background="var(--gray-1)",
    )


def main_chat_area() -> rx.Component:
    """Render the main chat area."""
    return rx.box(
        rx.vstack(
            chat_messages(),
            human_review_panel(),
            input_area(),
            width="100%",
            height="100%",
            spacing="0",
        ),
        flex="1",
        display="flex",
        flex_direction="column",
        overflow="hidden",
    )


def index() -> rx.Component:
    """Main page layout."""
    return rx.box(
        rx.vstack(
            header(),
            rx.hstack(
                main_chat_area(),
                settings_sidebar(),
                width="100%",
                height="calc(100vh - 73px)",
                spacing="0",
            ),
            width="100%",
            height="100vh",
            spacing="0",
        ),
        width="100%",
        height="100vh",
        overflow="hidden",
    )


# =============================================================================
# App Configuration
# =============================================================================


app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="blue",
        radius="medium",
    ),
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    style={
        "font_family": "Inter, sans-serif",
    },
)

app.add_page(index, route="/", title="Dual Agents Debate")
