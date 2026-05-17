from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyomnix.data_process.data_manipulator import DataManipulator
from pyomnix.utils import PlotParam


class TestPlottingCanvas(unittest.TestCase):
    def test_init_mosaic_canvas_returns_figure_axes_mapping_and_plot_params(self) -> None:
        mosaic = [["main", "side"], ["main", "bottom"]]

        fig, axes, params = DataManipulator.init_mosaic_canvas(
            mosaic,
            12,
            8,
            lines_per_fig=3,
        )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(set(axes), {"main", "side", "bottom"})
        self.assertTrue(all(isinstance(ax, Axes) for ax in axes.values()))
        self.assertIsInstance(params, PlotParam)
        self.assertEqual(params.shape, (2, 2, 3))
        plt.close(fig)

    def test_init_canvas_keeps_existing_subplots_return_shape(self) -> None:
        fig, ax, params = DataManipulator.init_canvas(1, 1, 8, 6)

        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)
        self.assertIsInstance(params, PlotParam)
        self.assertEqual(params.shape, (1, 1, 2))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
