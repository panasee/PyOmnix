from datetime import datetime
from functools import lru_cache

from langchain.tools import tool


def handle_tool_error(error: Exception) -> str:
    """
    Custom error handling logic.
    Return a friendly message to the LLM instead of a Python Traceback.
    """
    return (
        f"Error: {error!r}. Please try again with different parameters or handle the missing data."
    )


@lru_cache(maxsize=1)
def _get_google_search():
    """Lazy initialization of Google Search wrapper."""
    from langchain_google_community import GoogleSearchAPIWrapper

    return GoogleSearchAPIWrapper(k=7)


@tool
def google_search(search_query: str) -> str:
    """
    Search Google for recent results.

    Args:
        query: The query to search for.

    Returns:
        str: The search results.
    """
    search = _get_google_search()
    return search.run(search_query)


@tool
def get_current_time() -> str:
    """
    Get the current date and time.

    Returns:
        str: Current date and time in ISO format.
    """
    return datetime.now().isoformat()


@tool
def dummy_weather_tool(city: str) -> str:
    """
    Dummy weather tool for testing.

    Args:
        city: The city to query the weather for.

    Returns:
        str: Hardcoded weather string for deterministic testing.
    """
    return f"The weather in {city} is sunny and -271.15°C."


TEST_TOOL = [dummy_weather_tool]
