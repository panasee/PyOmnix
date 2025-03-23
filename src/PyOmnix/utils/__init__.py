"""
Utility functions for the PyOmnix package.

This module re-exports commonly used utility functions for convenience.
"""

from .data import (
    CacheArray,
    ObjectArray,
    difference,
    identify_direction,
    loop_diff,
    match_with_tolerance,
    rename_duplicates,
    symmetrize,
)
from .env import is_notebook, set_envs
from .math import (
    CM_TO_INCH,
    HBAR,
    HBAR_THZ,
    HPLANCK,
    KB,
    SWITCH_DICT,
    UNIT_FACTOR_FROMSI,
    UNIT_FACTOR_TO_SI,
    combined_generator_list,
    constant_generator,
    convert_unit,
    factor,
    gen_seq,
    get_unit_factor_and_texname,
    next_lst_gen,
    split_no_str,
    time_generator,
    timestr_convert,
)
from .plot import (
    DEFAULT_PLOT_DICT,
    PlotParam,
    combine_cmap,
    hex_to_rgb,
    print_progress_bar,
    truncate_cmap,
)

# For backward compatibility
__all__ = [
    "CM_TO_INCH",
    "HBAR",
    "HPLANCK",
    "KB",
    "factor",
    "hex_to_rgb",
    "is_notebook",
    "set_envs",
]
