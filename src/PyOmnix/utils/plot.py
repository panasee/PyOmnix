import copy
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from ..omnix_logger import get_logger
from ..pltconfig import color_preset as colors
from .data import ObjectArray

logger = get_logger(__name__)

# define plotting default settings
DEFAULT_PLOT_DICT = {
    "color": colors.Presets["Nl"][0],
    "linewidth": 1,
    "linestyle": "-",
    "marker": "o",
    "markersize": 1.5,
    "markerfacecolor": "None",
    "markeredgecolor": "black",
    "markeredgewidth": 0.3,
    "label": "",
    "alpha": 0.77,
}


class PlotParam(ObjectArray):
    """
    This class is used to store the parameters for the plot
    """

    def __init__(self, *dims: int) -> None:
        """
        initialize the PlotParam

        Args:
        - no_of_figs: the number of figures to be plotted
        """
        super().__init__(*dims, fill_value=copy.deepcopy(DEFAULT_PLOT_DICT))
        # define a tmp params used for temporary storage, especially in class methods for convenience
        self.tmp = copy.deepcopy(DEFAULT_PLOT_DICT)


def print_progress_bar(
    iteration: float,
    total: float,
    prefix="",
    suffix="",
    decimals=1,
    length=50,
    fill="#",
    print_end="\r",
) -> None:
    """
    Call in a loop to create terminal progress bar

    Args:
        iteration (float): current iteration
        total (float): total iterations
        prefix (str): prefix string
        suffix (str): suffix string
        decimals (int): positive number of decimals in percent complete
        length (int): character length of bar
        fill (str): bar fill character
        print_end (str): end character (e.g. "\r", "\r\n")
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    barr = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} [{barr}] {percent}% {suffix}", end=print_end, flush=True)
    # Print New Line on Complete
    if iteration == total:
        print()


def hex_to_rgb(
    hex_str: str, fractional: bool = True
) -> tuple[int, ...] | tuple[float, ...]:
    """
    convert hex color to rgb color

    Args:
        hex_str (str): hex color
        fractional (bool): if the return value is fractional or not
    """
    hex_str = hex_str.lstrip("#")
    if fractional:
        return tuple(int(hex_str[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))


def truncate_cmap(cmap, min_val: float = 0.0, max_val: float = 1.0, n: int = 256):
    """
    truncate the colormap to the specific range

    Args:
        cmap : LinearSegmentedColormap | ListedColormap
            the colormap to be truncated
        min_val : float
            the minimum value of the colormap
        max_val : float
            the maximum value of the colormap
        n : int
            the number of colors in the colormap
    """
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{min_val:.2f},{max_val:.2f})",
        cmap(np.linspace(min_val, max_val, n)),
    )
    return new_cmap


def combine_cmap(cmap_lst: list, segment: int = 128):
    """
    combine the colormaps in the list

    Args:
        cmap_lst : list
            the list of colormaps to be combined
        segment : int
            the number of segments in each colormap
    """
    c_lst = []
    for cmap in cmap_lst:
        c_lst.extend(cmap(np.linspace(0, 1, segment)))
    new_cmap = LinearSegmentedColormap.from_list("combined", c_lst)
    return new_cmap
