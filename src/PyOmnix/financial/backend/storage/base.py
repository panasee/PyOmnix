from abc import ABC, abstractmethod

from ..schemas import AgentExecutionLog, LLMInteractionLog


class BaseLogStorage(ABC):
    """Abstract base class for LLM interaction log storage."""

    @abstractmethod
    def add_log(self, log: LLMInteractionLog) -> None:
        """Adds a new log entry to the storage."""
        pass

    @abstractmethod
    def get_logs(
        self,
        agent_name: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[LLMInteractionLog]:
        """Retrieves logs, optionally filtering by agent name or run ID."""
        pass

    @abstractmethod
    def add_agent_log(self, log: AgentExecutionLog) -> None:
        """添加Agent执行日志"""
        pass

    @abstractmethod
    def get_agent_logs(
        self,
        agent_name: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[AgentExecutionLog]:
        """获取Agent执行日志，可按agent名称或run ID过滤"""
        pass

    @abstractmethod
    def get_unique_run_ids(self) -> list[str]:
        """获取所有唯一的运行ID列表"""
        pass
