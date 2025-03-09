"""
Utility functions for the PyOmnix package.

This module re-exports commonly used utility functions for convenience.
"""

from .env import set_envs, is_notebook
from .math import (
    split_no_str,
    factor,
    convert_unit,
    gen_seq,
    constant_generator,
    time_generator,
    combined_generator_list,
    next_lst_gen,
    timestr_convert,
    CM_TO_INCH,
    HPLANCK,
    HBAR,
    HBAR_THZ,
    KB,
    UNIT_FACTOR_FROMSI,
    UNIT_FACTOR_TO_SI,
    DEFAULT_PLOT_DICT,
    SWITCH_DICT
)
from .plot import (
    print_progress_bar,
    hex_to_rgb,
    truncate_cmap,
    combine_cmap
)
from .data import rename_duplicates

# For backward compatibility
__all__ = [
    "set_envs",
    "CM_TO_INCH", "HPLANCK", "HBAR", "KB", "factor",
    "hex_to_rgb", "is_notebook",
]