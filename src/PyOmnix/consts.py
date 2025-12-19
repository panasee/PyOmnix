"""
Constants for the PyOmnix project.
"""
from __future__ import annotations

# use standard logging here to avoid circular import
import os
from pathlib import Path

from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)
# Path
OMNIX_PATH: Path | None = None
LOG_FILE_PATH: str | Path | None = None  # folder for log files

SUCCESS_ICON = "✔️"
ERROR_ICON = "❌"
WAIT_ICON = "⏳"

def enable_rich_traceback() -> None:
    """
    enable rich traceback
    """
    from rich.traceback import install
    install(show_locals=True)

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
    else:
        omnix_path_str: str | None = os.getenv("OMNIX_PATH")
        if omnix_path_str is None:
            pylab_path_str: str | None = os.getenv("PYLAB_DB_LOCAL")
            if pylab_path_str is not None:
                logger.info("( *・ω・) read from PYLAB_DB_LOCAL:%s", pylab_path_str)
                OMNIX_PATH = Path(pylab_path_str)
                LOG_FILE_PATH = OMNIX_PATH / "logs"
            else:
                logger.info("•᷄ࡇ•᷅ PYLAB_DB_LOCAL not set")
                return
        else:
            logger.info("( *・ω・) read from OMNIX_PATH:%s", omnix_path_str)
            OMNIX_PATH = Path(omnix_path_str)
            LOG_FILE_PATH = OMNIX_PATH / "logs"

if __name__ == "__main__":
    set_paths(omnix_path="test")
    print(OMNIX_PATH)
    print(LOG_FILE_PATH)
    print(os.getenv("OMNIX_PATH"))
    print(os.getenv("PYLAB_DB_LOCAL"))