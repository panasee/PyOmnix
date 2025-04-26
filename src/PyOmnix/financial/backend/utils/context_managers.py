"""
上下文管理器模块

提供各种API相关的上下文管理器
"""

from contextlib import contextmanager

from pyomnix.omnix_logger import get_logger

from ..state import api_state

logger = get_logger("context_managers")


@contextmanager
def workflow_run(run_id: str):
    """
    工作流运行上下文管理器

    用法:
    with workflow_run(run_id):
        # 执行工作流
    """
    api_state.register_run(run_id)
    try:
        yield
        api_state.complete_run(run_id, "completed")
    except Exception:
        api_state.complete_run(run_id, "error")
        raise
