# judge if the special path exists in environment variables
# the special path is used to store private data, won't affect most usages

from .utils import set_envs, set_paths
from . import utils

from .ominix_logger import setup_logger, get_logger
# get_logger will return the logger instance if already exists
# setup_logger will reset existing logger

set_envs()
utils.OMNIX_PATH = set_paths()
