#!/usr/bin/env python
"""This module is responsible for processing and plotting the data"""

import importlib
import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plotly.subplots import make_subplots
from waitress import create_server

from ..omnix_logger import get_logger
from ..utils import (
    CM_TO_INCH,
    ObjectArray,
    PlotParam,
    get_unit_factor_and_texname,
    hex_to_rgb,
    is_notebook,
)

logger = get_logger(__name__)
logging.getLogger("waitress").setLevel(logging.ERROR)


class DataManipulator:
    """
    This class is responsible for manipulating the data.
    """

    # define static variables
    legend_font: dict
    """A constant dict used to set the font of the legend in the plot"""

    def __init__(
        self,
        *dims: int,
        plot_params: tuple[int] | int | None = None,
        usetex: bool = False,
        usepgf: bool = False,
    ) -> None:
        """
        Initialize the DataManipulator and load the settings for matplotlib saved in another file

        Args:
        - *dims: the dimensions of the data
        - plot_params: the parameters for the plot, the format is (n_row, n_col, lines_per_fig) or just a single integer
        - usetex: whether to use the TeX engine to render text
        - usepgf: whether to use the pgf backend
        """
        self.load_settings(usetex, usepgf)
        # data manipulating
        self.datas = ObjectArray(*dims)
        self.labels = ObjectArray(*dims)
        # static plotting
        self.plot_types: list[list[str]] = []
        self.unit = {
            "I": "A",
            "V": "V",
            "R": "Ohm",
            "T": "K",
            "B": "T",
            "f": "Hz",
            "1": "",
        }
        if plot_params is None:
            plot_params = dims
        self.params = PlotParam(*plot_params)
        # dynamic plotting
        self.live_dfs: list[list[list[go.Scatter]]] = []
        self.go_f: go.FigureWidget | None = None
        self._stop_event = threading.Event()
        self._thread = None
        self._dash_app = None
        self._dash_thread = None
        self._dash_server = None

    #####################
    # data manipulating #
    #####################
    def extend_dims(self, *dims: int) -> None:
        """
        Extend the dimensions of the data
        """
        self.datas.extend(*dims)
        self.labels.extend(*dims)

    def load_dfs(
        self,
        loc: int | tuple[Sequence[int], ...] | tuple[int, ...],
        data_in: Path | str | pd.DataFrame | Sequence[Path | str | pd.DataFrame],
        label_in: str | Sequence[str] | None = None,
        **kwargs,
    ) -> None:
        """
        Load the data from the given path(s), use list or tuple to specify multiple files at the same time

        Args:
        - loc: the location of the data
        - data_in: the dataframe or the path to the data
        - label_in: the label of the data
        - **kwargs: keyword arguments for pd.read_csv (sep, skiprow, header, float_precision, ...), shared by all files
        """
        if not isinstance(data_in, tuple | list):
            logger.validate(
                not isinstance(label_in, tuple | list) or label_in is None,
                "data_in must be in same shape with label_in",
            )
            data_in = (data_in,)
            label_in = (label_in,)
            loc = (loc,)
        else:
            logger.validate(
                isinstance(label_in, tuple | list),
                "data_in must be in same shape with label_in",
            )

        if label_in is None:
            label_in = ["" for _ in range(len(data_in))]
        logger.validate(
            len(data_in) == len(loc) == len(label_in),
            "data_path and loc must have the same length",
        )

        for path, loc, label in zip(data_in, loc, label_in, strict=False):
            if isinstance(path, pd.DataFrame):
                self.datas[loc] = path
            else:
                self.datas[loc] = pd.read_csv(path, **kwargs)
            self.labels[loc] = label

    def add_label(self, label: str, loc: tuple[int, ...] | int) -> None:
        """
        Add a label to the data at the given location.
        """
        if isinstance(loc, int):
            loc = (loc,)
        self.labels[loc] = label

    def get_datas(
        self,
        *,
        loc: tuple[int, ...] | int | None = None,
        label: str | Sequence[str] | None = None,
        concat: bool = False,
        suppress_warning: bool = False,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """
        Get the data dataframes from the given location.
        Either loc or label should be provided, but not both.
        (label would be ignored if loc is provided)
        for batch data, use ((1,),(2,)) instead of (1,2)
        Note: label allows for not-found, while loc does not

        Args:
            loc: The location indices to retrieve data from
            label: The label to search for in the data

        Returns:
            None if no data is found, or a list of DataFrames (even if only one df found)
        """
        dfs = None
        if loc is not None:
            if isinstance(loc, int):
                loc = (loc,)
            if isinstance(loc[0], int):
                loc = (loc,)
            if isinstance(loc[0], tuple | list | np.ndarray):
                dfs = [self.datas[loc_i] for loc_i in loc]
            else:
                logger.error("Invalid loc type")
                return None

        elif label is not None:
            # Find locations with matching label
            locations = self.labels.find_objs(label)
            # Return all matching data as a list of tuples
            if locations:
                dfs = [self.datas[loc] for loc in locations]
            else:
                if not suppress_warning:
                    logger.warning("No data found with label: %s", label)
                return None
        else:
            logger.error("Either loc or label must be provided")
            return

        if concat:
            return pd.concat(dfs)
        else:
            return dfs

    def clear_datas(self) -> None:
        """
        Clear the data in the ObjectArray
        """
        self.datas.clear()
        self.labels.clear()

    ###################
    # static plotting #
    ###################

    def unit_factor(self, property_name: str) -> float:
        """
        Used in plotting, to get the factor of the unit

        Args:
        - property_name: the property name string (like: I)
        """
        return get_unit_factor_and_texname(self.unit[property_name])[0]

    def unit_texname(self, property_name: str) -> str:
        """
        Used in plotting, to get the TeX name of the unit

        Args:
        - property_name: the property name string (like: I)
        """
        return get_unit_factor_and_texname(self.unit[property_name])[1]

    def set_unit(self, unit_new: dict[str, str] | None = None, **kwargs: str) -> None:
        """
        Set the unit for the plot, default to SI

        Args:
        - unit_new: the unit dictionary, the format is {"I":"uA", "V":"V", "R":"Ohm"}
        - **kwargs: additional units specified as keyword arguments (e.g., I="uA", V="V")
        """
        if unit_new is None:
            unit_new = {}
        self.unit.update(unit_new)
        if kwargs:
            self.unit.update(kwargs)

    def plot_df_cols(
        self,
        *,
        data_df: pd.DataFrame | None = None,
        loc: tuple[int, ...] | int | None = None,
        label: str | None = None,
    ) -> tuple[Figure, Axes] | None:
        """
        plot all columns w.r.t. the first column(not index) in the dataframe. data_df, loc and label are mutually exclusive, with descending priority

        Args:
        - data_df: the dataframe containing the data
        - loc: the location of the data
        - label: the label of the data
        """
        if data_df is None:
            data_df = self.get_datas(loc=loc, label=label, concat=True)
        fig, ax, _ = self.init_canvas(1, 1, 14, 20)
        for col in data_df.columns[1:]:
            ax.plot(data_df.iloc[:, 0], data_df[col], label=col)
        ax.set_xlabel(
            data_df.columns[0]
        )  # Set the label of the x-axis to the name of the first column
        ax.legend(edgecolor="black", prop=self.legend_font)
        return fig, ax

    def plot_mapping(
        self,
        mapping_x: any,
        mapping_y: any,
        mapping_val: any,
        *,
        data_df: pd.DataFrame | None = None,
        loc: tuple[int, ...] | int | None = None,
        label: str | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        cmap: str | Colormap = "viridis",
    ) -> tuple[Figure, Axes]:
        """
        plot the mapping of the data (x,y, and value(z))

        Args:
        - data_df: the dataframe containing the data
        - mapping_x: the column name for the x-axis
        - mapping_y: the column name for the y-axis
        - mapping_val: the column name for the mapping value
        - fig: the figure to plot the figure
        - ax: the axes to plot the figure
        - cmap: the colormap to use
        """
        if data_df is None:
            data_df = self.get_datas(loc=loc, label=label, concat=True)

        grid_df = data_df.pivot(index=mapping_x, columns=mapping_y, values=mapping_val)
        x_arr, y_arr = np.meshgrid(grid_df.columns, grid_df.index)

        if fig is None or ax is None:
            fig, ax, _ = DataManipulator.init_canvas(1, 1, 10, 8)

        contour = ax.contourf(x_arr, y_arr, grid_df, cmap=cmap)
        fig.colorbar(contour)
        return fig, ax

    @staticmethod
    def plot_spherical_background(ax: Axes, n_meridians=20, n_parallels=10, alpha=0.1):
        """
        plot the spherical background
        """
        # create the spherical grid
        u = np.linspace(0, 2 * np.pi, n_meridians)
        v = np.linspace(0, np.pi, n_parallels)
    
        # meridians (half circle from north to south)
        for phi in u:
            x = np.sin(v) * np.cos(phi)
            y = np.sin(v) * np.sin(phi)
            z = np.cos(v)
            ax.plot(x, y, z, color='gray', alpha=alpha, linestyle='dashed')
    
        # parallels (parallel circles)
        for theta in v[1:-1]:  # skip the north and south poles
            r = np.sin(theta)
            x = r * np.cos(u)
            y = r * np.sin(u)
            z = np.cos(theta) * np.ones_like(u)
            ax.plot(x, y, z, color='gray', alpha=alpha, linestyle='dashed')
    
        # add the axes
        ax.plot([-1, 1], [0, 0], [0, 0], color='red', alpha=0.5)   # X-axis
        ax.plot([0, 0], [-1, 1], [0, 0], color='green', alpha=0.5) # Y-axis
        ax.plot([0, 0], [0, 0], [-1, 1], color='blue', alpha=0.5)  # Z-axis

        # set the equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

    def plot_3d(
        self,
        x_col: str | Sequence[float | int],
        y_col: str | Sequence[float | int],
        z_col: str | Sequence[float | int],
        *,
        data_df: pd.DataFrame | None = None,
        loc: tuple[int, ...] | int | None = None,
        label: str | None = None,
        plot_type: Literal["surface", "scatter", "line"],
        cmap: str | Colormap = "viridis",
        alpha: float = 1.0,
        view_init: tuple[float, float] | None = None,
        color_lst: list[str] | None = None,
        colorbar: bool = True,
        title: str | None = None,
        grid: bool = True,
        coord_type: Literal["sphere", "cartesian"] = "cartesian",
        show_sphere: bool = True,
        sphere_alpha: float = 0.05,
        grid_alpha: float = 0.05,
        unit_limit: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Create a 3D plot of the data.

        Args:
            x_col: The column name for the x-axis data
            y_col: The column name for the y-axis data
            z_col: The column name for the z-axis data
            data_df: The dataframe containing the data. If None, will use get_datas
            loc: The location of the data in the ObjectArray
            label: The label for the data
            fig: The figure to plot on. If None, a new figure will be created
            ax: The axes to plot on. If None, new axes will be created
            plot_type: The type of 3D plot to create ("surface", "scatter", or "line")
            cmap: The colormap to use for surface plots
            alpha: The transparency of the plot
            view_init: Initial viewing angle (elevation, azimuth) in degrees

        Returns:
            tuple: The figure and axes objects
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        def get_data(col, data):
            if isinstance(col, str | bytes):
                return data[col].values if data is not None else None
            return np.asarray(col)

        x_data = get_data(x_col, data_df)
        y_data = get_data(y_col, data_df)
        z_data = get_data(z_col, data_df)

        if x_data is None or y_data is None or z_data is None:
            if data_df is None:
                data_df = self.get_datas(loc=loc, label=label, concat=True)
            if x_data is None:
                x_data = get_data(x_col, data_df)
            if y_data is None:
                y_data = get_data(y_col, data_df)
            if z_data is None:
                z_data = get_data(z_col, data_df)

        if show_sphere and coord_type == "sphere":
            u_grid = np.linspace(0, 2 * np.pi, 50)
            v_grid = np.linspace(0, np.pi, 50)
            x_grid = np.outer(np.cos(u_grid), np.sin(v_grid))
            y_grid = np.outer(np.sin(u_grid), np.sin(v_grid))
            z_grid = np.outer(np.ones(np.size(u_grid)), np.cos(v_grid))
            ax.plot_surface(x_grid, y_grid, z_grid, color='blue', alpha=sphere_alpha, rstride=1, cstride=1)

        if plot_type == "surface":
            # More efficient surface plot using griddata for irregular data
            x_unique = np.unique(x_data)
            y_unique = np.unique(y_data)
        
            if len(x_unique) * len(y_unique) != len(x_data):
                # Use interpolation for non-grid data
                from scipy.interpolate import griddata
                points = np.column_stack((x_data, y_data))
                grid_x, grid_y = np.meshgrid(
                    np.linspace(x_data.min(), x_data.max(), 100),
                    np.linspace(y_data.min(), y_data.max(), 100)
                )
                grid_z = griddata(points, z_data, (grid_x, grid_y), method='cubic')
                X, Y, Z = grid_x, grid_y, grid_z
            else:
                # Regular grid data
                X, Y = np.meshgrid(x_unique, y_unique)
                Z = z_data.reshape(len(y_unique), len(x_unique))
        
            surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha, 
                                 rstride=1, cstride=1, linewidth=0, 
                                 antialiased=False)
            if colorbar:
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=z_col if isinstance(z_col, str) else 'Z')

        elif plot_type == "scatter":
            if color_lst is None:
                scatter = ax.scatter(
                    x_data, y_data, z_data, c=z_data, cmap=cmap, alpha=alpha, s=5, edgecolors='none'
                )
            else:
                scatter = ax.scatter(
                    x_data, y_data, z_data, c=color_lst, alpha=alpha, s=5, edgecolors='none'
                )
            if colorbar:
                fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label=z_col if isinstance(z_col, str) else 'Z')

        elif plot_type == "line":
            ax.plot(x_data, y_data, z_data, alpha=alpha,
                    linewidth=2, marker='o' if len(x_data) <= 100 else None, markersize=4)

        ax.set_xlabel(x_col if isinstance(x_col, str) else 'X')
        ax.set_ylabel(y_col if isinstance(y_col, str) else 'Y')
        ax.set_zlabel(z_col if isinstance(z_col, str) else 'Z')

        if title:
            ax.set_title(title)
        if view_init is not None:
            ax.view_init(elev=view_init[0], azim=view_init[1])

        if unit_limit:
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

        if grid and coord_type == "sphere":
            DataManipulator.plot_spherical_background(ax, alpha=grid_alpha)
        elif grid and coord_type == "cartesian":
            ax.grid(grid)
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def load_settings(usetex: bool = False, usepgf: bool = False) -> None:
        """load the settings for matplotlib saved in another file"""
        file_name = "pyomnix.pltconfig.plot_config"
        if usetex:
            file_name += "_tex"
            if usepgf:
                file_name += "_pgf"
        else:
            file_name += "_notex"

        config_module = importlib.import_module(file_name)
        DataManipulator.legend_font = config_module.legend_font

    @staticmethod
    def ax_legend(ax: Axes, lst_artists: list[Line2D | Patch]) -> None:
        """
        add the legend to the axes
        """
        ax.legend(
            handles=lst_artists,
            labels=[artist.get_label() for artist in lst_artists],
            edgecolor="black",
            prop=DataManipulator.legend_font,
        )

    @staticmethod
    def paint_colors_twin_axes(
        *, ax_left: Axes, color_left: str, ax_right: Axes, color_right: str
    ) -> None:
        """
        paint the colors for the twin y axes

        Args:
        - ax: the axes to paint the colors
        - left: the color for the left y-axis
        - right: the color for the right y-axis
        """
        ax_left.tick_params("y", colors=color_left)
        ax_left.spines["left"].set_color(color_left)
        ax_right.tick_params("y", colors=color_right)
        ax_right.spines["right"].set_color(color_right)

    @staticmethod
    def init_canvas(
        n_row: int,
        n_col: int,
        figsize_x: float,
        figsize_y: float,
        sub_adj: tuple[float] = (0.19, 0.13, 0.97, 0.97, 0.2, 0.2),
        *,
        lines_per_fig: int = 2,
        **kwargs,
    ) -> tuple[Figure, Axes, PlotParam]:
        """
        initialize the canvas for the plot, return the fig and ax variables and params(n_row, n_col, 2)

        Args:
        - n_row: the fig no. of rows
        - n_col: the fig no. of columns
        - figsize_x: the width of the whole figure in cm
        - figsize_y: the height of the whole figure in cm
        - sub_adj: the adjustment of the subplots (left, bottom, right, top, wspace, hspace)
        - lines_per_fig: the number of lines per figure (used for appointing params)
        - **kwargs: keyword arguments for the plt.subplots function
        """
        fig, ax = plt.subplots(
            n_row,
            n_col,
            figsize=(figsize_x * CM_TO_INCH, figsize_y * CM_TO_INCH),
            **kwargs,
        )
        fig.subplots_adjust(
            left=sub_adj[0],
            bottom=sub_adj[1],
            right=sub_adj[2],
            top=sub_adj[3],
            wspace=sub_adj[4],
            hspace=sub_adj[5],
        )
        return fig, ax, PlotParam(n_row, n_col, lines_per_fig)

    #################
    # dynamic plots #
    #################

    def live_plot_init(
        self,
        n_rows: int,
        n_cols: int,
        lines_per_fig: int = 2,
        pixel_height: float = 600,
        pixel_width: float = 1200,
        *,
        titles: Sequence[Sequence[str]] | None = None,
        axes_labels: Sequence[Sequence[Sequence[str]]] | None = None,
        line_labels: Sequence[Sequence[Sequence[str]]] | None = None,
        plot_types: Sequence[Sequence[Literal["scatter", "contour", "heatmap"]]]
        | None = None,
        browser_open: bool = False,
        inline_jupyter: bool = True,
    ) -> None:
        """
        initialize the real-time plotter using plotly

        Args:
        - n_rows: the number of rows of the subplots
        - n_cols: the number of columns of the subplots
        - lines_per_fig: the number of lines per figure(ignored for contour plot)
        - pixel_height: the height of the figure in pixels
        - pixel_width: the width of the figure in pixels
        - titles: the titles of the subplots, shape should be (n_rows, n_cols), note the type notation
        - axes_labels: the labels of the axes, note the type notation, shape should be (n_rows, n_cols, 2[x and y axes labels])
        - line_labels: the labels of the lines, note the type notation, shape should be (n_rows, n_cols, lines_per_fig)
        - plot_types: the plot types for the lines, the type of plot for each subplot,
                options include 'scatter' and 'contour', shape should be (n_rows, n_cols)
        - browser_open: whether to open the browser automatically(only works when not in jupyter notebook)
        """
        if plot_types is None:
            plot_types = [["scatter" for _ in range(n_cols)] for _ in range(n_rows)]
        self.plot_types = plot_types
        # for contour plot, only one "line" is allowed
        traces_per_subplot = [
            [
                lines_per_fig if plot_types[i][j] == "scatter" else 1
                for j in range(n_cols)
            ]
            for i in range(n_rows)
        ]
        if titles is None:
            titles = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        flat_titles = [item for sublist in titles for item in sublist]
        if axes_labels is None:
            axes_labels = [
                [["" for _ in range(2)] for _ in range(n_cols)] for _ in range(n_rows)
            ]
        if line_labels is None:
            line_labels = [
                [["" for _ in range(lines_per_fig)] for _ in range(n_cols)] for _ in range(n_rows)
            ]

        # initial all the data arrays, not needed for just empty lists
        # x_arr = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
        # y_arr = [[[[] for _ in range(lines_per_fig)] for _ in range(n_cols)] for _ in range(n_rows)]

        def update_to_live_dfs():
            """
            put the data stored in go_f into the live_dfs to achieve real-time plotting
            """
            idx = 0
            for i in range(n_rows):
                for j in range(n_cols):
                    num_traces = traces_per_subplot[i][j]
                    for k in range(num_traces):
                        self.live_dfs[i][j].append(self.go_f.data[idx])
                        idx += 1

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=flat_titles)
        data_idx = 0
        self.live_dfs = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_cols):
                plot_type = plot_types[i][j]
                num_traces = traces_per_subplot[i][j]
                if plot_type == "scatter":
                    for k in range(num_traces):
                        fig.add_trace(
                            go.Scatter(
                                x=[],
                                y=[],
                                mode="lines+markers",
                                name=line_labels[i][j][k],
                            ),
                            row=i + 1,
                            col=j + 1,
                        )
                        data_idx += 1
                elif plot_type == "contour":
                    fig.add_trace(
                        go.Contour(z=[], x=[], y=[], name=line_labels[i][j][0]),
                        row=i + 1,
                        col=j + 1,
                    )
                    data_idx += 1
                elif plot_type == "heatmap":
                    fig.add_trace(
                        go.Heatmap(
                            z=[], x=[], y=[], name=line_labels[i][j][0], zsmooth="best"
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
                    data_idx += 1
                else:
                    raise ValueError(
                        f"Unsupported plot type '{plot_type}' at subplot ({i},{j})"
                    )
                fig.update_xaxes(title_text=axes_labels[i][j][0], row=i + 1, col=j + 1)
                fig.update_yaxes(title_text=axes_labels[i][j][1], row=i + 1, col=j + 1)

        fig.update_layout(height=pixel_height, width=pixel_width)

        if is_notebook() and inline_jupyter:
            browser_open = False
            from IPython.display import display

            self.go_f = go.FigureWidget(fig)
            update_to_live_dfs()
            display(self.go_f)

        else:
            import webbrowser

            from dash import Dash, dcc, html
            from dash.dependencies import Input, Output

            if inline_jupyter:
                logger.debug(
                    "inline_jupyter is not supported in non-notebook environment"
                )
            if not self._dash_app:
                app = Dash("live_plot_11235")
                self._dash_app = app

            self.go_f = fig
            update_to_live_dfs()
            self._dash_app.layout = html.Div(
                [
                    dcc.Graph(id="live-graph", figure=self.go_f),
                    dcc.Interval(id="interval-component", interval=500, n_intervals=0),
                ]
            )

            @self._dash_app.callback(
                Output("live-graph", "figure"),
                Input("interval-component", "n_intervals"),
                prevent_initial_call=True,
            )
            def update_graph(_):
                return self.go_f

            # Run Dash server in a separate thread
            def run_dash():
                logger.info("\nStarting real-time plot server...")
                logger.info("View the plot at: http://localhost:11235")
                # Run the server
                # Use the already created server instance instead of calling run directly
                self._dash_server = create_server(
                    self._dash_app.server, host="localhost", port=11235, threads=2
                )
                self._dash_server.run()

            if browser_open:
                webbrowser.open("http://localhost:11235")

            if not self._dash_thread:
                self._dash_thread = threading.Thread(target=run_dash, daemon=True)
                self._dash_thread.start()
                # Give the server a moment to start
                time.sleep(1)

    def save_fig_periodically(
        self, plot_path: Path | str, time_interval: int = 60
    ) -> None:
        """
        save the figure periodically
        this function will be running consistently in the background
        use threading to run this function in the background
        """
        if isinstance(plot_path, str):
            plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        while not self._stop_event.is_set():
            time.sleep(time_interval)
            max_retries = 3
            retry_delay = 2
            for attempt in range(max_retries):
                try:
                    self.go_f.write_image(plot_path)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Failed to save image (attempt {attempt + 1}/{max_retries}): {e!s}"
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Failed to save image after {max_retries} attempts: {e!s}"
                        )

    def start_saving(self, plot_path: Path | str, time_interval: int = 60) -> None:
        """
        start the thread to save the figure periodically
        """
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.save_fig_periodically, args=(plot_path, time_interval)
        )
        self._thread.start()

    def stop_saving(self) -> None:
        """
        stop the thread to save the figure periodically
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def live_plot_update(
        self,
        row: int | tuple[int],
        col: int | tuple[int],
        lineno: int | tuple[int],
        x_data: Sequence[float | str]
        | Sequence[Sequence[float | str]]
        | np.ndarray[float | str],
        y_data: Sequence[float | str]
        | Sequence[Sequence[float | str]]
        | np.ndarray[float | str],
        z_data: Sequence[float | str]
        | Sequence[Sequence[float | str]]
        | np.ndarray[float | str] = (0,),
        *,
        incremental=False,
        max_points: int | None = None,
        with_str: bool = False,
    ) -> None:
        """
        update the live data in jupyter, the row, col, lineno all can be tuples to update multiple subplots at the
        same time. Note that this function is not appending datapoints, but replot the whole line, so provide the
        whole data array for each update. The row, col, lineno, x_data, y_data should be of same length (no. of lines
        plotted). Be careful about the correspondence of data and index, e.g. when given indices like (0,1), the data
        should be like [[0],[1]], instead of [0,1] (incremental case).
        Example: live_plot_update((0,1), (0,1), (0,1), [x_arr1, x_arr2], [y_arr1, y_arr2]) will
        plot the (0,0,0) line with x_arr1 and y_arr1, and (1,1,1) line with x_arr2 and y_arr2
        SET data to empty list [] to clear the figure

        Args:
        - row: the row of the subplot (from 0)
        - col: the column of the subplot (from 0)
        - lineno: the line no. of the subplot (from 0)
        - x_data: the array-like x data (not support single number, use [x] or (x,) instead)
        - y_data: the array-like y data (not support single number, use [y] or (y,) instead)
        - z_data: the array-like z data (for contour plot only, be the same length as no of contour plots)
        - incremental: whether to update the data incrementally
        - max_points: the maximum number of points to be plotted, if None, no limit, only affect incremental line plots
        - with_str: whether there are strings (mainly for time string) in data. There will be no order for string data,
                   the string data will just be plotted evenly spaced
        """
        if not incremental and max_points is not None:
            logger.warning("max_points will be ignored when incremental is False")

        def ensure_list(data, target_type: type = np.float32) -> np.ndarray:
            def try_type(x):
                try:
                    return target_type(x)
                except (ValueError, TypeError):
                    return x

            if isinstance(data, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)):
                return np.array([try_type(i) for i in data])
            else:
                return np.array([try_type(data)])

        def ensure_2d_array(data, if_with_str=False) -> np.ndarray:
            data_arr = ensure_list(data)
            if data_arr.size == 0:
                return data_arr
            if not isinstance(data_arr[0], np.ndarray):
                if if_with_str:
                    return np.array([data_arr])
                return np.array([data_arr], dtype=np.float32)
            else:
                if if_with_str:
                    return np.array(data_arr)
                return np.array(data_arr, dtype=np.float32)

        row = ensure_list(row, target_type=int)
        col = ensure_list(col, target_type=int)
        lineno = ensure_list(lineno, target_type=int)
        if not incremental:
            x_data = ensure_2d_array(x_data, with_str)
            y_data = ensure_2d_array(y_data, with_str)
            z_data = ensure_2d_array(z_data, with_str)
        else:
            x_data = ensure_list(x_data)
            y_data = ensure_list(y_data)
            z_data = ensure_list(z_data)

        # dim_tolift = [0, 0, 0]
        with self.go_f.batch_update():
            idx_z = 0
            for no, (irow, icol, ilineno) in enumerate(
                zip(row, col, lineno, strict=False)
            ):
                plot_type = self.plot_types[irow][icol]
                trace = self.live_dfs[irow][icol][ilineno]
                if plot_type == "scatter":
                    if incremental:
                        trace.x = (
                            np.append(trace.x, x_data[no])[-max_points:]
                            if max_points is not None
                            else np.append(trace.x, x_data[no])
                        )
                        trace.y = (
                            np.append(trace.y, y_data[no])[-max_points:]
                            if max_points is not None
                            else np.append(trace.y, y_data[no])
                        )
                    else:
                        trace.x = x_data[no]
                        trace.y = y_data[no]
                if plot_type == "contour" or plot_type == "heatmap":
                    if not incremental:
                        trace.x = x_data[no]
                        trace.y = y_data[no]
                        trace.z = z_data[idx_z]
                    else:
                        trace.x = np.append(trace.x, x_data[no])
                        trace.y = np.append(trace.y, y_data[no])
                        trace.z = np.append(trace.z, z_data[idx_z])
                    idx_z += 1
            assert idx_z == len(z_data) or (idx_z == 0 and z_data == (0,)), (
                "z_data should have the same length as the number of contour plots"
            )
        if not is_notebook() and not incremental:
            self.go_f.update_layout(uirevision=True)
            time.sleep(0.5)

    ##########################
    # color selection method #
    ##########################

    @staticmethod
    def sel_pan_color(
        row: int | None = None,
        col: int | None = None,
        data_extract: bool = False,
        external_file: str | Path | None = None,
    ) -> (
        tuple[tuple[float | int, ...], str]
        | None
        | tuple[list[list[tuple[float | int, ...]]], dict]
    ):
        """
        select the color according to the position in pan_colors method (use row and col as in 2D array)
        leave row and col as None to show the color palette
        if customized file is used, the length should be similar (2305 - 2352)

        Args:
        - row: the row of the color selected
        - col: the column of the color selected
        - data_extract: used internally to get color data without plotting
        - external_file: the external file to load the color data from
        """
        if external_file is None:
            localenv_filter = re.compile(r"^PYLAB_DB_LOCAL")
            filtered_vars = {
                key: value
                for key, value in os.environ.items()
                if localenv_filter.match(key)
            }
            used_var = list(filtered_vars.keys())[0]
            if filtered_vars:
                filepath = Path(filtered_vars[used_var]) / "pan-colors.json"
                logger.info(f"load path from ENVIRON: {used_var}")
                return DataManipulator.sel_pan_color(row, col, data_extract, filepath)
            else:
                with resources.open_text("DaySpark.pltconfig", "pan_color.json") as f:
                    color_dict = json.load(f)
        else:
            with open(external_file, encoding="utf-8") as f:
                color_dict = json.load(f)
        full_rgbs = list(map(hex_to_rgb, color_dict["values"]))
        rgbs = full_rgbs[:2304]
        extra = full_rgbs[2304:]
        extra += [(1, 1, 1)] * (48 - len(extra))
        rgb_mat = [rgbs[i * 48 : (i + 1) * 48] for i in range(48)]
        rgb_mat.append(extra)
        if not data_extract:
            if row is None and col is None:
                DataManipulator.load_settings(False, False)
                fig, ax, _ = DataManipulator.init_canvas(1, 1, 20, 20)
                ax.imshow(rgb_mat)
                ax.set_xticks(np.arange(0, 48, 5))
                ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                ax.set_yticks(np.arange(0, 48, 5))
                plt.grid()
                plt.show()
            elif row is not None and col is not None:
                return rgb_mat[row][col], color_dict["names"][row * 48 + col]
            else:
                logger.error("x and y should be both None or both not None")
        else:
            return rgb_mat, color_dict

    @staticmethod
    def gui_pan_color() -> None:
        """
        GUI for selecting the color
        """
        try:
            from PyQt6.QtCore import Qt, pyqtSignal
            from PyQt6.QtGui import QBrush, QColor
            from PyQt6.QtWidgets import (
                QApplication,
                QHBoxLayout,
                QHeaderView,
                QLabel,
                QTableWidget,
                QTableWidgetItem,
                QVBoxLayout,
                QWidget,
            )
        except ImportError:
            logger.error("PyQt6 is not installed")
            return

        def rgb_float_to_int(rgb_tuple):
            return tuple(int(c * 255) for c in rgb_tuple)

        class ColorPaletteWidget(QTableWidget):
            colorSelected = pyqtSignal(str, tuple, str)

            def __init__(self, _rgb_mat, _color_dict):
                super().__init__(len(_rgb_mat), 48)
                self.rgb_mat = _rgb_mat
                self.color_dict = _color_dict
                self.init_ui()

            def init_ui(self):
                self.verticalHeader().setVisible(False)
                self.horizontalHeader().setVisible(False)
                self.horizontalHeader().setSectionResizeMode(
                    QHeaderView.ResizeMode.Stretch
                )
                self.verticalHeader().setSectionResizeMode(
                    QHeaderView.ResizeMode.Stretch
                )

                for r in range(len(self.rgb_mat)):
                    for c in range(48):
                        rgb = self.rgb_mat[r][c]
                        item = QTableWidgetItem()
                        # Convert RGB float to int (0-255)
                        rgb_int = rgb_float_to_int(rgb)
                        qcolor = QColor(*rgb_int)
                        item.setBackground(QBrush(qcolor))
                        self.setItem(r, c, item)

                self.cellClicked.connect(self.handle_cell_click)

            def handle_cell_click(self, row, col):
                index = row * 48 + col
                if index < len(self.color_dict["names"]):
                    color_name = self.color_dict["names"][index]
                else:
                    color_name = "Unknown"
                rgb = self.rgb_mat[row][col]
                rgb_int = rgb_float_to_int(rgb)
                hex_val = "#{:02X}{:02X}{:02X}".format(*rgb_int)
                self.colorSelected.emit(color_name, rgb_int, hex_val)

        class MainWindow(QWidget):
            def __init__(self):
                super().__init__()
                rgb_mat, color_dict = DataManipulator.sel_pan_color(data_extract=True)
                self.color_widget = ColorPaletteWidget(rgb_mat, color_dict)

                # Info labels
                self.name_label = QLabel("Name: N/A")
                self.rgb_label = QLabel("RGB: N/A")
                self.hex_label = QLabel("Hex: N/A")

                # Layout for info panel
                info_layout = QHBoxLayout()
                info_layout.addWidget(self.name_label)
                info_layout.addWidget(self.rgb_label)
                info_layout.addWidget(self.hex_label)

                main_layout = QVBoxLayout()
                main_layout.addWidget(self.color_widget)
                main_layout.addLayout(info_layout)

                self.setLayout(main_layout)
                self.setWindowTitle("Color Palette Selector")
                self.resize(1200, 900)

                # Connect signal
                self.color_widget.colorSelected.connect(self.update_info)

            def update_info(self, name, rgb_int, hex_str):
                self.name_label.setText(f"Name: {name}")
                self.rgb_label.setText(f"RGB: {rgb_int}")
                self.hex_label.setText(f"Hex: {hex_str}")

        app = QApplication(sys.argv)
        w = MainWindow()
        w.show()
        sys.exit(app.exec())

    @staticmethod
    def preview_colors(
        color_lst: tuple[float | int, ...]
        | list[tuple[float | int, ...]]
        | list[list[tuple[float | int, ...]]],
    ) -> None:
        """
        preview the colors in the list
        """
        self.load_settings(False, False)
        fig, ax, _ = self.init_canvas(1, 1, 13, 7)
        try:
            if isinstance(color_lst[0], float | int):
                ax.imshow([[color_lst]])
            if isinstance(color_lst[0][0], float | int):
                ax.imshow([color_lst])
            elif isinstance(color_lst[0][0][0], float | int):
                ax.imshow(color_lst)
            else:
                logger.error("wrong format")
                return
        except Exception:
            logger.error("wrong format")
            return
        plt.show()
