"""
Constants for the PyOmnix project.
"""
from pathlib import Path
import os
# use standard logging here to avoid circular import
import logging
from .omnix_logger import get_logger

logger = get_logger(__name__)
# Path
OMNIX_PATH: Path | None = None
LOG_FILE_PATH: str | Path = None   # folder for log files
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
        if os.getenv("OMNIX_PATH") is None:
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
