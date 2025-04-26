"""
Serialization tools - convert complex Python objects to JSON-serializable format
"""

from datetime import UTC, datetime
from typing import Any


def serialize_agent_state(state: dict) -> dict:
    """
    convert AgentState object to JSON-serializable dictionary

    Args:
        state: Agent state dictionary, may contain non-JSON-serializable objects

    Returns:
        converted JSON-friendly dictionary
    """
    if not state:
        return {}

    try:
        return _convert_to_serializable(state)
    except Exception as e:
        # if serialization fails, at least return a useful error message
        return {
            "error": f"unable to serialize state: {e!s}",
            "serialization_error": True,
            "timestamp": datetime.now(UTC).isoformat(),
        }


def _convert_to_serializable(obj: Any) -> Any:
    """recursively convert object to JSON-serializable format"""
    if hasattr(obj, "to_dict"):  # handle Pandas Series/DataFrame
        return obj.to_dict()
    elif hasattr(obj, "content") and hasattr(obj, "type"):  # may be LangChain message
        return {"content": _convert_to_serializable(obj.content), "type": obj.type}
    elif hasattr(obj, "__dict__"):  # handle custom objects
        return _convert_to_serializable(obj.__dict__)
    elif isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)  # fallback to string representation
