from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_log_storage
from ..schemas import LLMInteractionLog
from ..storage.base import BaseLogStorage

router = APIRouter(
    prefix="/logs",
    tags=["LLM Interaction Logs"],  # Tag for Swagger UI grouping
)


@router.get("/", response_model=list[LLMInteractionLog])
def read_logs(
    agent_name: str | None = Query(None, description="Filter logs by agent name"),
    run_id: str | None = Query(None, description="Filter logs by run ID"),
    # Default limit 50, non-negative
    limit: int | None = Query(
        50, description="Maximum number of logs to return (most recent)", ge=0
    ),
    storage: BaseLogStorage = Depends(get_log_storage),
):
    """Retrieve LLM interaction logs, with optional filtering and limit."""
    try:
        logs = storage.get_logs(agent_name=agent_name, run_id=run_id, limit=limit)
        return logs
    except Exception as e:
        # Basic error handling
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {e!s}")
