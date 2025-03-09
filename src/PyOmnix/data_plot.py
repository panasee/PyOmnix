#!/usr/bin/env python
"""This module is responsible for processing and plotting the data"""

import importlib
import copy
import json
import re
import os
import threading
import time
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Optional, Literal

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from .utils import CM_TO_INCH, factor, DEFAULT_PLOT_DICT, is_notebook, hex_to_rgb
from .omnix_logger import get_logger

logger = get_logger(__name__)

class DataPlot():
    """
    This class is responsible for plotting the data.
    """
    # define static variables
    legend_font: dict
    """A constant dict used to set the font of the legend in the plot"""

    class PlotParam:
        """
        This class is used to store the parameters for the plot
        """

        def __init__(self, *dims: int) -> None:
            """
            initialize the PlotParam

            Args:
            - no_of_figs: the number of figures to be plotted
            """
            self.shape = dims
            self.params_list = self._create_params_list(dims)
            # define a tmp params used for temporary storage, especially in class methods for convenience
            self.tmp = copy.deepcopy(DEFAULT_PLOT_DICT)

        def _create_params_list(self, dims: tuple[int, ...]) -> list[dict] | list[any]:
            """
            create the list of parameters for the plot

            Args:
            - dims: the dimensions of the parameters
            """
            if len(dims) == 1:
                return [copy.deepcopy(DEFAULT_PLOT_DICT) for _ in range(dims[0])]
            else:
                return [self._create_params_list(dims[1:]) for _ in range(dims[0])]

        def _get_subarray(self, array, index: tuple[int, ...]) -> list[dict]:
            """
            get the subarray of the parameters for the plot assigned by the index
            """
            if len(index) == 1:
                return array[index[0]]
            else:
                return self._get_subarray(array[index[0]], index[1:])

        def _set_subarray(self, array, index: tuple[int, ...], target_dict: dict) -> None:
            """
            set the subarray of the parameters for the plot assignated by the index
            """
            if len(index) == 1:
                array[index[0]] = copy.deepcopy(target_dict)
            else:
                self._set_subarray(array[index[0]], index[1:], target_dict)

        def _flatten(self, lst):
            """
            Flatten a multi-dimensional list using recursion
            """
            return [item for sublist in lst for item in
                    (self._flatten(sublist) if isinstance(sublist, list) else [sublist])]

        def __getitem__(self, index: tuple[int, ...] | int) -> dict:
            """
            get the parameters for the plot assignated by the index

            Args:
            - index: the index of the figure to be get
            """
            if isinstance(index, int):
                flat_list = self._flatten(self.params_list)
                return flat_list[index]
            result = self._get_subarray(self.params_list, index)
            while isinstance(result, list) and len(result) == 1:
                result = result[0]
            return result

        def __setitem__(self, index: tuple[int, ...] | int, value):
            if isinstance(index, int):
                index = (index,)
            self._set_subarray(self.params_list, index, value)

    def __init__(self, *, no_params: tuple[int] | int = 4, usetex: bool = False, usepgf: bool = False) -> None:
        """
        Initialize the FileOrganizer and load the settings for matplotlib saved in another file
        
        Args:
        - no_params: the number of params to be initiated (default:4) 
        - usetex: whether to use the TeX engine to render text
        - usepgf: whether to use the pgf backend
        """
        self.plot_types: list[list[str]] = []
        DataPlot.load_settings(usetex, usepgf)
        self.unit = {"I": "A", "V": "V", "R": "Ohm", "T": "K", "B": "T", "f": "Hz"}
        # params here are mainly used for internal methods
        self.params = DataPlot.PlotParam(no_params)
        self.live_dfs: list[list[list[go.Scatter]]] = []
        self.go_f: Optional[go.FigureWidget] = None
        self._stop_event = threading.Event()
        self._thread = None

    def unit_factor(self, axis_name: str) -> float:
        """
        Used in plotting, to get the factor of the unit

        Args:
        - axis_name: the unit name string (like: uA)
        """
        return self.get_unit_factor_and_texname(self.unit[axis_name])[0]

    def unit_name(self, axis_name: str) -> str:
        """
        Used in plotting, to get the TeX name of the unit

        Args:
        - axis_name: the unit name string (like: uA)
        """
        return self.get_unit_factor_and_texname(self.unit[axis_name])[1]

    @staticmethod
    def get_unit_factor_and_texname(unit: str) -> tuple[float, str]:
        """
        Used in plotting, to get the factor and the TeX name of the unit
        
        Args:
        - unit: the unit name string (like: uA)
        """
        _factor = factor(unit)
        if unit[0] == "u":
            namestr = rf"$\mathrm{{\mu {unit[1:]}}}$".replace("Omega", r"\Omega").replace("Ohm", r"\Omega")
        else:
            namestr = rf"$\mathrm{{{unit}}}$".replace("Omega", r"\Omega").replace("Ohm", r"\Omega")
        return _factor, namestr

    def set_unit(self, unit_new: dict = None) -> None:
        """
        Set the unit for the plot, default to SI

        Args:
        - unit_new: the unit dictionary, the format is {"I":"uA", "V":"V", "R":"Ohm"}
        """
        self.unit.update(unit_new)

    def plot_df_cols(self, data_df: pd.DataFrame) -> Optional[tuple[Figure, Axes]]:
        """
        plot all columns w.r.t. the first column(not index) in the dataframe

        Args:
        - data_df: the dataframe containing the data
        """
        fig, ax, _ = DataPlot.init_canvas(1, 1, 14, 20)
        for col in data_df.columns[1:]:
            ax.plot(data_df.iloc[:, 0], data_df[col], label=col)
        ax.set_xlabel(data_df.columns[0])  # Set the label of the x-axis to the name of the first column
        ax.legend(edgecolor='black', prop=DataPlot.legend_font)
        return fig, ax

    @staticmethod
    def plot_mapping(data_df: pd.DataFrame, mapping_x: any, mapping_y: any, mapping_val: any, *,
                     fig: Figure = None, ax: Axes = None, cmap: str = "viridis") -> tuple[Figure, Axes]:
        """
        plot the mapping of the data

        Args:
        - data_df: the dataframe containing the data
        - mapping_x: the column name for the x-axis
        - mapping_y: the column name for the y-axis
        - mapping_val: the column name for the mapping value
        - ax: the axes to plot the figure
        """
        grid_df = data_df.pivot(index=mapping_x, columns=mapping_y, values=mapping_val)
        x_arr, y_arr = np.meshgrid(grid_df.columns, grid_df.index)

        if fig is None or ax is None:
            fig, ax, _ = DataPlot.init_canvas(1, 1, 10, 8)

        contour = ax.contourf(x_arr, y_arr, grid_df, cmap=cmap)
        fig.colorbar(contour)
        return fig, ax

    @staticmethod
    def load_settings(usetex: bool = False, usepgf: bool = False) -> None:
        """load the settings for matplotlib saved in another file"""
        file_name = "PyOmnix.pltconfig.plot_config"
        if usetex:
            file_name += "_tex"
            if usepgf:
                file_name += "_pgf"
        else:
            file_name += "_notex"

        config_module = importlib.import_module(file_name)
        DataPlot.legend_font = getattr(config_module, 'legend_font')

    @staticmethod
    def paint_colors_twin_axes(*, ax_left: Axes, color_left: str, ax_right: Axes,
                               color_right: str) -> None:
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
    def init_canvas(n_row: int, n_col: int, figsize_x: float, figsize_y: float,
                    sub_adj: tuple[float] = (0.19, 0.13, 0.97, 0.97, 0.2, 0.2),*, lines_per_fig: int = 2, **kwargs) \
            -> tuple[Figure, Axes, PlotParam]:
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
        fig, ax = plt.subplots(n_row, n_col, figsize=(figsize_x * CM_TO_INCH, figsize_y * CM_TO_INCH), **kwargs)
        fig.subplots_adjust(left=sub_adj[0], bottom=sub_adj[1], right=sub_adj[2], top=sub_adj[3], wspace=sub_adj[4],
                            hspace=sub_adj[5])
        return fig, ax, DataPlot.PlotParam(n_row, n_col, lines_per_fig)

    def live_plot_init(self, n_rows: int, n_cols: int, 
                       lines_per_fig: int = 2, 
                       pixel_height: float = 600,
                       pixel_width: float = 1200, *, 
                       titles: Optional[Sequence[Sequence[str]]] = None,
                       axes_labels: Optional[Sequence[Sequence[Sequence[str]]]] = None,
                       line_labels: Optional[Sequence[Sequence[Sequence[str]]]] = None,
                       plot_types: Optional[Sequence[Sequence[Literal["scatter", "contour", "heatmap"]]]] = None) -> None:
        """
        initialize the real-time plotter using plotly

        Args:
        - n_rows: the number of rows of the subplots
        - n_cols: the number of columns of the subplots
        - lines_per_fig: the number of lines per figure
        - pixel_height: the height of the figure in pixels
        - pixel_width: the width of the figure in pixels
        - titles: the titles of the subplots, shape should be (n_rows, n_cols), note the type notation
        - axes_labels: the labels of the axes, note the type notation, shape should be (n_rows, n_cols, 2[x and y axes labels])
        - line_labels: the labels of the lines, note the type notation, shape should be (n_rows, n_cols, lines_per_fig)
        - plot_types: the plot types for the lines, the type of plot for each subplot,
                options include 'scatter' and 'contour', shape should be (n_rows, n_cols)
        """
        if plot_types is None:
            plot_types = [['scatter' for _ in range(n_cols)] for _ in range(n_rows)]
        self.plot_types = plot_types
        # for contour plot, only one "line" is allowed
        traces_per_subplot = [[lines_per_fig if plot_types[i][j] == 'scatter' else 1 for j in range(n_cols)] for i
                              in range(n_rows)]
        if titles is None:
            titles = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        flat_titles = [item for sublist in titles for item in sublist]
        if axes_labels is None:
            axes_labels = [[["" for _ in range(2)] for _ in range(n_cols)] for _ in range(n_rows)]
        if line_labels is None:
            line_labels = [[["" for _ in range(2)] for _ in range(n_cols)] for _ in range(n_rows)]

        # initial all the data arrays, not needed for just empty lists
        #x_arr = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
        #y_arr = [[[[] for _ in range(lines_per_fig)] for _ in range(n_cols)] for _ in range(n_rows)]

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=flat_titles)
        data_idx = 0
        self.live_dfs = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_cols):
                plot_type = plot_types[i][j]
                num_traces = traces_per_subplot[i][j]
                if plot_type == 'scatter':
                    for k in range(num_traces):
                        fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name=line_labels[i][j][k]),
                                      row=i + 1, col=j + 1)
                        data_idx += 1
                elif plot_type == 'contour':
                    fig.add_trace(go.Contour(z=[], x=[], y=[], name=line_labels[i][j][0]), row=i + 1, col=j + 1)
                    data_idx += 1
                elif plot_type == 'heatmap':
                    fig.add_trace(go.Heatmap(z=[], x=[], y=[], name=line_labels[i][j][0], zsmooth="best"),
                                  row=i + 1, col=j + 1)
                    data_idx += 1
                else:
                    raise ValueError(f"Unsupported plot type '{plot_type}' at subplot ({i},{j})")
                fig.update_xaxes(title_text=axes_labels[i][j][0], row=i + 1, col=j + 1)
                fig.update_yaxes(title_text=axes_labels[i][j][1], row=i + 1, col=j + 1)

        fig.update_layout(height=pixel_height, width=pixel_width)
        if is_notebook():
            from IPython.display import display
            self.go_f = go.FigureWidget(fig)
#            self.live_dfs = [
#                [[self.go_f.data[i * n_cols * lines_per_fig + j * lines_per_fig + k] for k in range(lines_per_fig)] for
#                 j in range(n_cols)] for i in range(n_rows)]
            idx = 0
            for i in range(n_rows):
                for j in range(n_cols):
                    num_traces = traces_per_subplot[i][j]
                    for k in range(num_traces):
                        self.live_dfs[i][j].append(self.go_f.data[idx])
                        idx += 1
            display(self.go_f)
        else:
            import dash
            from dash import html, dcc
            from dash.dependencies import Input, Output
            import threading
            import webbrowser

            port = 11235
            app = dash.Dash(__name__)
            app.layout = html.Div([
                dcc.Graph(id='live-graph', figure=fig),
                dcc.Interval(id='interval-component', interval= 500, n_intervals=0)
            ])

            self.go_f = fig
            idx = 0
            for i in range(n_rows):
                for j in range(n_cols):
                    num_traces = traces_per_subplot[i][j]
                    for k in range(num_traces):
                        self.live_dfs[i][j].append(self.go_f.data[idx])
                        idx += 1

            @app.callback(
                Output('live-graph', 'figure'),
                Input('interval-component', 'n_intervals'),
                prevent_initial_call=True
            )
            def update_graph(_):
                return self.go_f

            # Run Dash server in a separate thread
            def run_dash():
                logger.info("\nStarting real-time plot server...")
                logger.info(f"View the plot at: http://localhost:{port}")
                # Open the browser automatically
                webbrowser.open(f'http://localhost:{port}')
                # Run the server
                app.run_server(debug=False, port=port, dev_tools_silence_routes_logging=True,
                use_reloader=False)

            self._dash_thread = threading.Thread(target=run_dash, daemon=True)
            self._dash_thread.start()
            # Give the server a moment to start
            time.sleep(2)

    def save_fig_periodically(self, plot_path: Path | str, time_interval: int = 60) -> None:
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
            self.go_f.write_image(plot_path)

    def start_saving(self, plot_path: Path | str, time_interval: int = 60) -> None:
        """
        start the thread to save the figure periodically
        """
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.save_fig_periodically, args=(plot_path, time_interval))
        self._thread.start()

    def stop_saving(self) -> None:
        """
        stop the thread to save the figure periodically
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def live_plot_update(self, row: int | tuple[int], col: int | tuple[int], lineno: int | tuple[int],
                         x_data: Sequence[float | str] | Sequence[Sequence[float | str]] | np.ndarray[float | str],
                         y_data: Sequence[float | str] | Sequence[Sequence[float | str]] | np.ndarray[float | str],
                         z_data: Sequence[float | str] | Sequence[Sequence[float | str]] | np.ndarray[float | str] = (
                         0,), *,
                         incremental=False, max_points: Optional[int] = None, with_str: bool = False) -> None:
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

        #dim_tolift = [0, 0, 0]
        with (self.go_f.batch_update()):
            idx_z = 0
            for no, (irow, icol, ilineno) in enumerate(zip(row, col, lineno)):
                plot_type = self.plot_types[irow][icol]
                trace = self.live_dfs[irow][icol][ilineno]
                if plot_type == 'scatter':
                    if incremental:
                        trace.x = np.append(trace.x, x_data[no])[-max_points:] if max_points is not None \
                            else np.append(trace.x, x_data[no])
                        trace.y = np.append(trace.y, y_data[no])[-max_points:] if max_points is not None \
                            else np.append(trace.y, y_data[no])
                    else:
                        trace.x = x_data[no]
                        trace.y = y_data[no]
                if plot_type == 'contour' or plot_type == "heatmap":
                    if not incremental:
                        trace.x = x_data[no]
                        trace.y = y_data[no]
                        trace.z = z_data[idx_z]
                    else:
                        trace.x = np.append(trace.x, x_data[no])
                        trace.y = np.append(trace.y, y_data[no])
                        trace.z = np.append(trace.z, z_data[idx_z])
                    idx_z += 1
            assert idx_z == len(z_data) or (idx_z == 0 and z_data == (0,)), \
                "z_data should have the same length as the number of contour plots"
        if not is_notebook() and not incremental:
            self.go_f.update_layout(uirevision=True)
            time.sleep(0.5)

    @staticmethod
    def sel_pan_color(row: Optional[int] = None, col: Optional[int] = None, data_extract: bool = False, external_file: Optional[str | Path] = None) \
            -> Optional[tuple[tuple[float | int, ...], str]] | tuple[list[list[tuple[float | int, ...]]], dict]:
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
                key: value for key, value in os.environ.items() if localenv_filter.match(key)
            }
            used_var = list(filtered_vars.keys())[0]
            if filtered_vars:
                filepath = Path(filtered_vars[used_var]) / "pan-colors.json"
                logger.info(f"load path from ENVIRON: {used_var}")
                return DataPlot.sel_pan_color(row, col, data_extract, filepath)
            else:
                with resources.open_text("DaySpark.pltconfig", "pan_color.json") as f:
                    color_dict = json.load(f)
        else:
            with open(external_file, encoding='utf-8') as f:
                color_dict = json.load(f)
        full_rgbs = list(map(hex_to_rgb, color_dict["values"]))
        rgbs = full_rgbs[:2304]
        extra = full_rgbs[2304:]
        extra += [(1, 1, 1)] * (48 - len(extra))
        rgb_mat = [rgbs[i * 48:(i + 1) * 48] for i in range(48)]
        rgb_mat.append(extra)
        if not data_extract:
            if row is None and col is None:
                DataPlot.load_settings(False, False)
                fig, ax, _ = DataPlot.init_canvas(1, 1, 20, 20)
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
            from PyQt6.QtWidgets import (
                QApplication, QTableWidget, QTableWidgetItem, QHeaderView,
                QWidget, QLabel, QVBoxLayout, QHBoxLayout
            )
            from PyQt6.QtGui import QColor, QBrush
            from PyQt6.QtCore import Qt, pyqtSignal
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
                self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
                self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

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
                rgb_mat, color_dict = DataPlot.sel_pan_color(data_extract=True)
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
    def preview_colors(color_lst: tuple[float | int, ...] | list[tuple[float | int, ...]] |
                       list[list[tuple[float | int, ...]]]) -> None:
        """
        preview the colors in the list
        """
        DataPlot.load_settings(False, False)
        fig, ax, _ = DataPlot.init_canvas(1, 1, 13, 7)
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
