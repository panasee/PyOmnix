from __future__ import annotations

import unittest
from unittest.mock import patch

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

    def test_label_subplots_supports_mosaic_axes_mapping(self) -> None:
        fig, axes, _ = DataManipulator.init_mosaic_canvas(
            [["main", "side"], ["main", "bottom"]],
            12,
            8,
        )

        labels = DataManipulator.label_subplots(axes, labels={"main": "A", "side": "B"})

        self.assertEqual(len(labels), 2)
        self.assertEqual([text.get_text() for text in labels], ["A", "B"])
        self.assertIs(labels[0].axes, axes["main"])
        self.assertIs(labels[1].axes, axes["side"])
        plt.close(fig)

    def test_label_subplots_supports_subplots_axes_array(self) -> None:
        fig, axes = plt.subplots(2, 2)

        labels = DataManipulator.label_subplots(axes, labels=["A", "B", "C", "D"])

        self.assertEqual([text.get_text() for text in labels], ["A", "B", "C", "D"])
        self.assertIs(labels[3].axes, axes[1, 1])
        plt.close(fig)

    def test_polish_axes_restores_pyomnix_static_plot_style(self) -> None:
        fig, ax = plt.subplots()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        DataManipulator.polish_axes(ax)

        self.assertTrue(ax.spines["top"].get_visible())
        self.assertTrue(ax.spines["right"].get_visible())
        self.assertAlmostEqual(ax.spines["left"].get_linewidth(), 0.7)
        self.assertAlmostEqual(ax.spines["bottom"].get_linewidth(), 0.7)
        plt.close(fig)

    def test_live_plot_init_passes_plotly_specs_and_skips_empty_cells(self) -> None:
        dm = DataManipulator(1)
        specs = [[{"rowspan": 2}, {}], [None, {}]]
        titles = [["main", "side"], ["", "bottom"]]

        with patch.object(DataManipulator, "create_dash"):
            dm.live_plot_init(
                2,
                2,
                lines_per_fig=1,
                titles=titles,
                specs=specs,
                inline_jupyter=False,
            )

        self.assertEqual(len(dm.go_f.data), 3)
        self.assertEqual(len(dm.live_dfs[0][0]), 1)
        self.assertEqual(len(dm.live_dfs[0][1]), 1)
        self.assertEqual(len(dm.live_dfs[1][0]), 0)
        self.assertEqual(len(dm.live_dfs[1][1]), 1)
        self.assertEqual(
            [annotation.text for annotation in dm.go_f.layout.annotations],
            ["main", "side", "bottom"],
        )


if __name__ == "__main__":
    unittest.main()
