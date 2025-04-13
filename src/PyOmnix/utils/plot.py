import colorsys
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from ..omnix_logger import get_logger
from ..pltconfig import color_preset as colors
from .data import ObjectArray, sph_to_cart

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
    if np.sign(iteration) == np.sign(total):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
    else:
        percent = ("{0:." + str(decimals) + "f}").format(0)
        filled_length = 0
    barr = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} [{barr}] {percent}% {suffix}", end=print_end, flush=True)
    # Print New Line on Complete
    # if iteration == total:
    #    print()


def hex_to_rgb(hex_str: str, fractional: bool = True) -> tuple[int, ...] | tuple[float, ...]:
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


def hsv_analyze(
    image: Path | str | Image.Image, sample_factor: float = 1.0, if_plot: bool = False
) -> tuple[float, float, float]:
    """
    analyze the hsv value of the image

    Args:
        image : Path | str | Image.Image
            the image to be analyzed
        sample_factor : float
            the sample factor of the image
        if_plot : bool
            if the plot is shown

    Returns:
        x_plot : np.ndarray
            the x coordinate of the plot
        y_plot : np.ndarray
            the y coordinate of the plot
        z_plot : np.ndarray
            the z coordinate of the plot
        colors_lst : list
            the list of colors for each point
    """
    if isinstance(image, Path | str):
        image = Image.open(image)
    hsv_image = image.convert("HSV")
    h, s, v = hsv_image.split()
    h_array = np.array(h)
    s_array = np.array(s)
    v_array = np.array(v)
    height, width = h_array.shape[0], h_array.shape[1]
    sample_size = int(height * width * sample_factor)

    random_indices = np.random.choice(height * width, sample_size, replace=False)
    h_norm = h_array.reshape(-1)[random_indices] / 255
    s_norm = s_array.reshape(-1)[random_indices] / 255
    v_norm = v_array.reshape(-1)[random_indices] / 255

    # coordinate transform
    phi_h = h_norm * 2 * np.pi
    r_s = s_norm
    theta_v = v_norm * np.pi

    x_plot, y_plot, z_plot = sph_to_cart(r_s, theta_v, phi_h)

    if if_plot:
        colors_lst = list(map(lambda h, s, v: colorsys.hsv_to_rgb(h, s, v), h_norm, s_norm, v_norm))
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x_plot, y_plot, z_plot, c=colors_lst, s=1, alpha=0.6)
        ax.set_xlabel("X (Saturation * cos(Hue))")
        ax.set_ylabel("Y (Saturation * sin(Hue))")
        ax.set_zlabel("Z (Lightness)")
        ax.set_title(f"HSL Color Sphere for {image} (Sampled {sample_factor * 100:.1f}% pixels)")
        plt.tight_layout()
        plt.show()

    return x_plot, y_plot, z_plot, colors_lst


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
