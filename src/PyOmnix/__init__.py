# judge if the special path exists in environment variables
# the special path is used to store private data, won't affect most usages

from .utils import set_envs
from .consts import set_paths, OMNIX_PATH, LOG_FILE_PATH
from . import utils

from .omnix_logger import setup_logger, get_logger
# get_logger will return the logger instance if already exists
# setup_logger will reset existing logger

set_envs()
set_paths()
if OMNIX_PATH is not None:
    OMNIX_PATH.mkdir(parents=True, exist_ok=True)
    LOG_FILE_PATH.mkdir(parents=True, exist_ok=True)
