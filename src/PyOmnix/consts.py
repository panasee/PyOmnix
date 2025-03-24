"""
Constants for the PyOmnix project.
"""

# use standard logging here to avoid circular import
import os
from pathlib import Path

from .omnix_logger import get_logger

logger = get_logger(__name__)
# Path
OMNIX_PATH: Path | None = None
LOG_FILE_PATH: str | Path | None = None  # folder for log files

# Model
SUMMARIZE_PROMPT_SYS = """
You are a multilingual conversation summarizer. Your task is:
First analyze the primary language used in the conversation (detect Chinese/English)
Then create a summary using the same language as the conversation
Preserve the original:
Core meaning and key details
Conversational tone and style
Participant roles and dialog structure
Strictly avoid mixing languages in the summary
If the conversation contains bilingual content:
Use the majority language (≥60%)
For balanced languages (50/50), default to original initiator's language

Additionally, monitor for topic deviation:
1. Identify the initial topic of conversation
2. Detect if the conversation has significantly deviated from this topic
3. If deviation is detected, include a clear warning in your summary
4. Suggest how participants can return to the main topic
5. Phrase this warning in the same language as your summary
"""
SUMMARIZE_PROMPT_HUMAN = """
Please summarize the conversation above
"""


def set_paths(*, omnix_path: Path | str | None = None) -> None:
    """
    two ways are provided to set the paths:
    1. set the paths directly in the function (before other modules are imported)
    2. set the paths in the environment variables OMNIX_PATH
    """
    global OMNIX_PATH, LOG_FILE_PATH
    if omnix_path is not None:
        OMNIX_PATH = Path(omnix_path)
        LOG_FILE_PATH = OMNIX_PATH / "logs"
    elif os.getenv("OMNIX_PATH") is None:
        if os.getenv("PYLAB_DB_LOCAL") is None:
            logger.info("OMNIX_PATH not set")
            return
        else:
            logger.info("read from PYLAB_DB_LOCAL:%s", os.getenv("PYLAB_DB_LOCAL"))
            OMNIX_PATH = Path(os.getenv("PYLAB_DB_LOCAL"))
            LOG_FILE_PATH = OMNIX_PATH / "logs"
    else:
        logger.info("read from OMNIX_PATH:%s", os.getenv("OMNIX_PATH"))
        OMNIX_PATH = Path(os.getenv("OMNIX_PATH"))
        LOG_FILE_PATH = OMNIX_PATH / "logs"
