# judge if the special path exists in environment variables
# the special path is used to store private data, won't affect most usages

from . import utils
from .consts import LOG_FILE_PATH, OMNIX_PATH, set_paths
from .omnix_logger import get_logger, setup_logger
from .utils import set_envs
from .data_process import DataManipulator

# get_logger will return the logger instance if already exists
# setup_logger will reset existing logger

set_envs()
set_paths()
if OMNIX_PATH is not None:
    OMNIX_PATH.mkdir(parents=True, exist_ok=True)
    LOG_FILE_PATH.mkdir(parents=True, exist_ok=True)
