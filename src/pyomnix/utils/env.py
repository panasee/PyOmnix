import os
import sys

from ..omnix_logger import get_logger

logger = get_logger(__name__)


def set_envs() -> None:
    """
    automatically set the environment variables from related environment variables and
    set the environment variables from related environment variables
    for compatibility with another package pyflexlab, PYLAB_DB_LOCAL will also be recognized when there is no OMNIX_PATH
    """
    for env_var in ["OMNIX_PATH", "PYLAB_DB_LOCAL"]:
        if env_var in os.environ:
            return

        for key in os.environ:
            if key.startswith(env_var):
                os.environ[env_var] = os.environ[key]
                logger.info("set with %s", key)
                return

        logger.info("%s not found in environment variables", env_var)


def is_notebook() -> bool:
    """
    judge if the code is running in a notebook environment.
    """
    if "ipykernel" in sys.modules and "IPython" in sys.modules:
        try:
            from IPython import get_ipython

            if "IPKernelApp" in get_ipython().config:
                return True
        except:
            pass
    return False
