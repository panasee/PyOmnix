"""PyQt GUI for interactive data loading, visualization, and splitting.

This GUI ports the main ideas from the previous tkinter prototype into a
PyQt-based application, while keeping logic layered and testable. Heavy data
manipulation is delegated to small utilities (e.g., DataSplitter), and the
plotting uses Matplotlib on the QtAgg backend.

Focus in this iteration:
- File/Folder manager in a tree view (in-memory project)
- Load CSV/TXT and HDF5 (with project metadata preservation)
- Select X/Y columns and plot with Matplotlib
- Split sweeps using DataSplitter (into new virtual files)
- Export selected to CSV/Excel/HDF5

Notes:
- No changes are made to existing APIs like data_manipulator.py
- This module can be run as a standalone utility for convenience
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QUrl  # type: ignore[attr-defined]
from PyQt6.QtWidgets import (  # type: ignore[attr-defined]
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    QWebEngineView = None  # type: ignore[assignment]

matplotlib.use("QtAgg")

import plotly.graph_objects as go
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..omnix_logger import get_logger
from .data_manipulator import DataManipulator
from .data_splitter import DataSplitter

logger = get_logger(__name__)


@dataclass
class FileNode:
    name: str
    is_folder: bool = False
    parent_folder: str | None = None
    children: list[str] = field(default_factory=list)
    tree_item: QTreeWidgetItem | None = None
    data: pd.DataFrame | None = None
    modified_data: pd.DataFrame | None = None
    is_modified: bool = False


class InMemoryProject:
    """Simple project tree in memory with helper operations."""

    def __init__(self) -> None:
        self.nodes: dict[str, FileNode] = {}
        self.root_items: list[str] = []

    def add_folder(self, name: str, parent_folder: str | None = None) -> FileNode:
        if name in self.nodes:
            return self.nodes[name]
        node = FileNode(name=name, is_folder=True, parent_folder=parent_folder)
        self.nodes[name] = node
        if parent_folder and parent_folder in self.nodes:
            self.nodes[parent_folder].children.append(name)
        else:
            self.root_items.append(name)
        return node

    def add_file(self, name: str, data: pd.DataFrame, parent_folder: str | None = None) -> FileNode:
        node = FileNode(name=name, is_folder=False, parent_folder=parent_folder, data=data)
        self.nodes[name] = node
        if parent_folder and parent_folder in self.nodes:
            self.nodes[parent_folder].children.append(name)
        else:
            self.root_items.append(name)
        return node

    def get_current_data(self, name: str) -> pd.DataFrame | None:
        node = self.nodes.get(name)
        if not node or node.is_folder:
            return None
        return node.modified_data if node.modified_data is not None else node.data

    def update_data(self, name: str, df: pd.DataFrame) -> None:
        node = self.nodes.get(name)
        if node and not node.is_folder:
            node.modified_data = df
            node.is_modified = True

    def delete_item(self, name: str) -> None:
        node = self.nodes.get(name)
        if not node:
            return
        if node.is_folder:
            for child in list(node.children):
                self.delete_item(child)
        else:
            node.data = None
            node.modified_data = None
        # unlink from parent
        if node.parent_folder and node.parent_folder in self.nodes:
            self.nodes[node.parent_folder].children = [
                c for c in self.nodes[node.parent_folder].children if c != name
            ]
        else:
            self.root_items = [r for r in self.root_items if r != name]
        del self.nodes[name]

    def get_files_recursive(self, name: str) -> list[str]:
        node = self.nodes.get(name)
        if not node:
            return []
        if node.is_folder:
            out: list[str] = []
            for c in node.children:
                out.extend(self.get_files_recursive(c))
            return out
        return [name]

    def get_item_path(self, name: str) -> str:
        parts: list[str] = []
        cur = name
        while cur:
            parts.append(cur)
            n = self.nodes.get(cur)
            if n and n.parent_folder:
                cur = n.parent_folder
            else:
                break
        return "/".join(reversed(parts))


class MatplotWidget(QWidget):
    """Matplotlib Figure embedded in a QWidget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)

    def clear(self) -> None:
        self.figure.clear()
        self.canvas.draw_idle()

    def plot(self, plotter) -> None:
        plotter(self.figure)
        self.canvas.draw_idle()


class PlotlyDashWidget(QWidget):
    """Plotly Dash embedded in a QWidget via QWebEngineView."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if QWebEngineView is None:
            raise ImportError(
                "PyQt6-WebEngine is required. Install with: pip install PyQt6-WebEngine"
            )
        self.view = QWebEngineView(self)  # type: ignore[operator]
        layout = QVBoxLayout(self)
        layout.addWidget(self.view)

    def load_dash(self, url: str = "http://localhost:11235") -> None:
        self.view.setUrl(QUrl(url))


class SplitDialog(QDialog):
    """Dialog to configure sweep splitting."""

    def __init__(self, columns: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Split Sweeps")
        self.setModal(True)

        form = QFormLayout(self)
        self.combo_field = QComboBox(self)
        self.combo_field.addItems(columns)
        form.addRow("Field column:", self.combo_field)

        self.combo_param = QComboBox(self)
        self.combo_param.addItem("None")
        self.combo_param.addItems(columns)
        form.addRow("Parameter column (optional):", self.combo_param)

        self.abbrev_edit = QLineEdit(self)
        form.addRow("Parameter abbrev (e.g., T, G):", self.abbrev_edit)

        self.min_points = QSpinBox(self)
        self.min_points.setRange(2, 10_000)
        self.min_points.setValue(10)
        form.addRow("Minimum points per sweep:", self.min_points)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def values(self) -> tuple[str, str | None, str, int]:
        param = self.combo_param.currentText()
        return (
            self.combo_field.currentText(),
            None if param == "None" else param,
            self.abbrev_edit.text().strip(),
            int(self.min_points.value()),
        )


class EasyDataWindow(QMainWindow):
    """Main window for the PyQt-based data tool."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyOmnix - Data Tool (PyQt)")
        self.resize(1400, 900)

        self.project = InMemoryProject()
        self.splitter = DataSplitter()

        # Central layout: left tree, right plot panel
        central = QWidget(self)
        self.setCentralWidget(central)

        hsplit = QSplitter(Qt.Orientation.Horizontal, central)
        left_panel = QWidget(hsplit)
        right_panel = QWidget(hsplit)
        hsplit.addWidget(left_panel)
        hsplit.addWidget(right_panel)
        hsplit.setStretchFactor(0, 0)
        hsplit.setStretchFactor(1, 1)

        main_layout = QHBoxLayout(central)
        main_layout.addWidget(hsplit)

        # Left: controls and tree
        left_layout = QVBoxLayout(left_panel)

        toolbar = QHBoxLayout()
        self.btn_new_folder = QPushButton("New Folder")
        self.btn_add_files = QPushButton("Add Files")
        self.btn_import = QPushButton("Import Project")
        toolbar.addWidget(self.btn_new_folder)
        toolbar.addWidget(self.btn_add_files)
        toolbar.addWidget(self.btn_import)
        left_layout.addLayout(toolbar)

        self.tree = QTreeWidget(left_panel)
        self.tree.setHeaderLabels(["Name", "Type", "Status", "Size"])
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        left_layout.addWidget(self.tree, 1)

        # Plot controls
        plot_group = QGroupBox("Plot Controls", left_panel)
        form = QFormLayout(plot_group)
        self.combo_x = QComboBox(plot_group)
        self.combo_y = QComboBox(plot_group)
        form.addRow("X Axis:", self.combo_x)
        form.addRow("Y Axis:", self.combo_y)
        left_layout.addWidget(plot_group)

        # Operations
        ops_group = QGroupBox("Operations", left_panel)
        ops_layout = QHBoxLayout(ops_group)
        self.btn_plot = QPushButton("Plot")
        self.btn_split = QPushButton("Split Sweeps")
        self.btn_export = QPushButton("Export")
        ops_layout.addWidget(self.btn_plot)
        ops_layout.addWidget(self.btn_split)
        ops_layout.addWidget(self.btn_export)
        left_layout.addWidget(ops_group)

        # Debug log
        self.log_text = QTextEdit(left_panel)
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text, 1)

        # Right: plot widget
        right_layout = QVBoxLayout(right_panel)
        # Switch from Matplotlib to Plotly Dash embed
        self.plot_widget = PlotlyDashWidget(right_panel)
        right_layout.addWidget(self.plot_widget, 1)
        # Initialize DataManipulator/Dash
        self.dm = DataManipulator(1)
        # Start a simple 1x1 figure in Dash without opening browser
        self.dm.live_plot_init(
            1,
            1,
            lines_per_fig=5,
            pixel_height=700,
            pixel_width=900,
            titles=[[""]],
            axes_labels=[[["", ""]]],
            plot_types=[["scatter"]],
            browser_open=False,
            inline_jupyter=False,
        )
        # Load the Dash page into the view (respect dynamically selected port)
        url = self.dm.get_dash_url() or "http://127.0.0.1:11235"
        self.plot_widget.load_dash(url)

        # Connections
        self.btn_new_folder.clicked.connect(self.create_folder)
        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_import.clicked.connect(self.import_project)
        self.btn_plot.clicked.connect(self.plot_data)
        self.btn_split.clicked.connect(self.split_sweeps)
        self.btn_export.clicked.connect(self.export_selected)
        self.tree.itemSelectionChanged.connect(self.on_selection_changed)
        self.combo_x.currentIndexChanged.connect(self.plot_data)
        self.combo_y.currentIndexChanged.connect(self.plot_data)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)

        self._log("Initialized PyQt GUI")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        try:
            if getattr(self, "dm", None) is not None:
                self.dm.stop_dash()
        except Exception:
            pass
        super().closeEvent(event)

    # ------------------------------- UI helpers -------------------------------
    def _log(self, msg: str, log_type: str = "info") -> None:
        if log_type == "info":
            logger.info(msg)
        elif log_type == "error":
            logger.error(msg)
        elif log_type == "warning":
            logger.warning(msg)
        elif log_type == "debug":
            logger.debug(msg)
        else:
            raise ValueError(f"Invalid log type: {log_type}")
        self.log_text.append(msg)
        self.log_text.ensureCursorVisible()

    def _selected_items(self) -> list[QTreeWidgetItem]:
        return list(self.tree.selectedItems())

    def _item_name(self, item: QTreeWidgetItem) -> str | None:
        # Reverse-lookup by tree_item
        for name, node in self.project.nodes.items():
            if node.tree_item is item:
                return name
        return None

    def _update_axis_combos_from_name(self, name: str) -> None:
        df = self.project.get_current_data(name)
        if df is None:
            return
        cols = list(map(str, df.columns))
        self.combo_x.clear()
        self.combo_x.addItems(cols)
        self.combo_y.clear()
        self.combo_y.addItems(cols)
        if cols:
            self.combo_x.setCurrentIndex(0)
        if len(cols) > 1:
            self.combo_y.setCurrentIndex(1)

    # ------------------------------ Tree actions ------------------------------
    def create_folder(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "New Folder", "Folder name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Error", "Folder name cannot be empty")
            return
        if name in self.project.nodes:
            QMessageBox.warning(self, "Error", "An item with this name exists")
            return

        parent_item = self.tree.currentItem()
        parent_name = self._item_name(parent_item) if parent_item else None
        if parent_name and not self.project.nodes[parent_name].is_folder:
            parent_name = None

        node = self.project.add_folder(name, parent_name)
        item = QTreeWidgetItem([name, "Folder", "-", "-"])
        node.tree_item = item
        if parent_item:
            parent_item.addChild(item)
            parent_item.setExpanded(True)
        else:
            self.tree.addTopLevelItem(item)
        self._log(f"Created folder: {name}")

    def add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select data files",
            "",
            "Data files (*.txt *.csv *.dat *.tsv);;HDF5 files (*.h5 *.hdf5);;All files (*.*)",
        )
        if not paths:
            return
        parent_item = self.tree.currentItem()
        parent_name = self._item_name(parent_item) if parent_item else None
        if parent_name and not self.project.nodes[parent_name].is_folder:
            parent_name = None

        for p in paths:
            try:
                path = Path(p)
                if path.suffix.lower() in {".h5", ".hdf5"}:
                    self._import_hdf5(path, parent_name, parent_item)
                else:
                    df = self._load_text_file(path)
                    if df is not None:
                        node = self.project.add_file(path.name, df, parent_name)
                        item = QTreeWidgetItem([path.name, "File", "Original", f"{len(df):,} rows"])
                        node.tree_item = item
                        if parent_item:
                            parent_item.addChild(item)
                            parent_item.setExpanded(True)
                        else:
                            self.tree.addTopLevelItem(item)
                        self._log(f"Loaded: {path.name} ({len(df)} rows, {len(df.columns)} cols)")
            except Exception as e:
                self._log(f"Error loading {Path(p).name}: {e}")

    def import_project(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Import Project",
            "",
            "Project files (*.h5 *.hdf5);;All files (*.*)",
        )
        if not p:
            return
        if self.tree.topLevelItemCount() > 0:
            if (
                QMessageBox.question(self, "Import Project", "Replace current project?")
                != QMessageBox.StandardButton.Yes
            ):
                return
            self.tree.clear()
            self.project = InMemoryProject()
        self._import_hdf5(Path(p), None, None)
        self._log(f"Imported project: {Path(p).name}")

    def show_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.tree.itemAt(pos)
        menu = QMenu(self)
        act_rename = menu.addAction("Rename")
        act_delete = menu.addAction("Delete")
        act = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if act is act_rename and item:
            self._rename_item(item)
        elif act is act_delete and item:
            self._delete_items([item])

    def _rename_item(self, item: QTreeWidgetItem) -> None:
        name = self._item_name(item)
        if not name:
            return
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename", "New name:", text=name)
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == name:
            return
        if new_name in self.project.nodes:
            QMessageBox.warning(self, "Error", "An item with this name exists")
            return
        node = self.project.nodes[name]
        node.name = new_name  # type: ignore[attr-defined]
        self.project.nodes[new_name] = node
        del self.project.nodes[name]
        item.setText(0, new_name)
        self._log(f"Renamed '{name}' -> '{new_name}'")

    def _delete_items(self, items: list[QTreeWidgetItem]) -> None:
        if not items:
            return
        if len(items) == 1:
            prompt = f"Delete '{items[0].text(0)}'?"
        else:
            prompt = f"Delete {len(items)} items?"
        if QMessageBox.question(self, "Confirm Delete", prompt) != QMessageBox.StandardButton.Yes:
            return
        for it in items:
            name = self._item_name(it)
            if not name:
                continue
            self.project.delete_item(name)
            parent = it.parent()
            if parent:
                parent.removeChild(it)
            else:
                idx = self.tree.indexOfTopLevelItem(it)
                self.tree.takeTopLevelItem(idx)
        self._log("Deleted item(s)")

    # ------------------------------ Plot & update -----------------------------
    def on_selection_changed(self) -> None:
        items = self._selected_items()
        if not items:
            return
        # prefer first file to populate axes
        for it in items:
            name = self._item_name(it)
            if name and not self.project.nodes[name].is_folder:
                self._update_axis_combos_from_name(name)
                break
        self.plot_data()

    def plot_data(self) -> None:
        items = self._selected_items()
        if not items:
            # clear plot: set empty data on first trace if exists
            if getattr(self, "dm", None) and getattr(self.dm, "go_f", None):
                if len(self.dm.go_f.data) > 0:
                    self.dm.live_plot_update(0, 0, 0, [], [])
            return
        x_col = self.combo_x.currentText()
        y_col = self.combo_y.currentText()
        if not x_col or not y_col:
            return

        files: list[str] = []
        for it in items:
            name = self._item_name(it)
            if name:
                files.extend(self.project.get_files_recursive(name))
        files = list(dict.fromkeys(files))  # stable unique

        # Prepare data and update Plotly live traces
        valid = []
        for idx, fname in enumerate(files):
            df = self.project.get_current_data(fname)
            if df is None or x_col not in df.columns or y_col not in df.columns:
                continue
            valid.append((idx, fname, df[x_col], df[y_col]))

        if valid and getattr(self.dm, "go_f", None):
            # update axis labels and ensure enough traces
            try:
                # narrow type hint for linters in dynamic environment
                fig: any = self.dm.go_f  # type: ignore[assignment]
                fig.update_xaxes(title_text=x_col, row=1, col=1)
                fig.update_yaxes(title_text=y_col, row=1, col=1)
                existing = len(fig.data)
                needed = len(valid)
                for _ in range(max(0, needed - existing)):
                    fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers"), row=1, col=1)
                for i, (_, fname, _, _) in enumerate(valid):
                    fig.data[i].name = fname
            except Exception:
                pass

            # push data per-trace to satisfy type hints
            for i, (_, _, x_series, y_series) in enumerate(valid):
                self.dm.live_plot_update(0, 0, i, x_series, y_series)

        self._log(f"Plotted {len(valid)} file(s): {x_col} vs {y_col}")

    # --------------------------------- Split ---------------------------------
    def split_sweeps(self) -> None:
        items = self._selected_items()
        if not items:
            QMessageBox.information(self, "No selection", "Select at least one file/folder.")
            return

        # Use columns from first selected file
        first_name: str | None = None
        for it in items:
            n = self._item_name(it)
            if n and not self.project.nodes[n].is_folder:
                first_name = n
                break
        if not first_name:
            QMessageBox.information(self, "No files", "Please select at least one file.")
            return
        df0 = self.project.get_current_data(first_name)
        if df0 is None:
            return
        dlg = SplitDialog(columns=[str(c) for c in df0.columns], parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        field_col, param_col, param_abbrev, min_points = dlg.values()

        total_created = 0
        for it in items:
            name = self._item_name(it)
            if not name:
                continue
            for fname in self.project.get_files_recursive(name):
                df = self.project.get_current_data(fname)
                if df is None or field_col not in df.columns:
                    continue
                # Detect and split
                segs = self.splitter.detect_sweep_segments(df, field_col, min_points=min_points)
                parts = self.splitter.split_dataframe_by_segments(df, segs)
                if len(parts) <= 1:
                    continue

                # Determine naming
                base = Path(fname).stem
                ext = Path(fname).suffix.lstrip(".")
                param_val: float | None = None
                if param_col and param_col in df.columns:
                    param_val = self.splitter.estimate_param_value(df[param_col])

                # Insert new files alongside original
                parent_folder = self.project.nodes[fname].parent_folder
                parent_item = self.project.nodes[parent_folder].tree_item if parent_folder else None
                for i, (seg, part_df) in enumerate(zip(segs, parts, strict=False), start=1):
                    out_name = self.splitter.create_sweep_filenames(
                        base,
                        index=i,
                        direction=seg.direction,
                        original_ext=ext if ext else None,
                        param_label=(param_abbrev or param_col) if param_col else None,
                        param_value=param_val,
                    )
                    node = self.project.add_file(out_name, part_df, parent_folder)
                    item = QTreeWidgetItem([out_name, "File", "Original", f"{len(part_df):,} rows"])
                    node.tree_item = item
                    if parent_item:
                        parent_item.addChild(item)
                        parent_item.setExpanded(True)
                    else:
                        self.tree.addTopLevelItem(item)
                    total_created += 1

        if total_created:
            self._log(f"Created {total_created} sweep file(s)")
            self.plot_data()
        else:
            QMessageBox.information(self, "No sweeps found", "No multi-sweep files detected.")

    # --------------------------------- Export --------------------------------
    def export_selected(self) -> None:
        items = self._selected_items()
        if not items:
            QMessageBox.information(self, "No selection", "Select files or folders to export.")
            return
        files: list[str] = []
        for it in items:
            name = self._item_name(it)
            if name:
                files.extend(self.project.get_files_recursive(name))
        files = list(dict.fromkeys(files))
        if not files:
            return

        menu = QMenu(self)
        act_csv = menu.addAction("Export CSV/Text")
        act_excel = menu.addAction("Export Excel (X/Y only)")
        act_hdf5 = menu.addAction("Export Project (HDF5)")
        pos_global = QtGui.QCursor.pos()
        act = menu.exec(pos_global)
        if act is act_csv:
            self._export_csv(files)
        elif act is act_excel:
            self._export_excel(files)
        elif act is act_hdf5:
            self._export_hdf5(files)

    # --------------------------------- IO impl -------------------------------
    @staticmethod
    def _detect_data_structure(
        file_path: str | Path, max_check_rows: int = 10
    ) -> tuple[list[str] | None, int | None, str | None]:
        """Detect the structure of the data file more precisely."""
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = [f.readline().strip() for _ in range(max_check_rows)]

            # Find the line that looks like column headers
            header_line = None
            header_row_idx = None

            for i, line in enumerate(lines):
                if not line:
                    continue

                # Split by tabs first, then by spaces
                if "\t" in line:
                    parts = line.split("\t")
                else:
                    parts = line.split()

                # Skip lines that look like timestamps or metadata
                if any(
                    keyword in line.lower()
                    for keyword in ["am", "pm", "frequency", "stop at", "hz", "db/oct"]
                ):
                    continue

                # Look for lines with typical column names
                valid_column_indicators = [
                    "outputv",
                    "time",
                    "temp",
                    "volt",
                    "current",
                    "b",
                    "t1",
                    "t2",
                    "i",
                    "x1",
                    "angle",
                    "hall",
                    "gate",
                    "leak",
                    "source",
                    "sense",
                ]

                # Count how many parts look like column names
                valid_parts = 0
                for part in parts:
                    part_lower = part.lower().strip()
                    if any(indicator in part_lower for indicator in valid_column_indicators):
                        valid_parts += 1
                    elif len(part_lower) <= 10 and (
                        part_lower.isalpha()
                        or (
                            any(c.isalpha() for c in part_lower)
                            and any(c.isdigit() for c in part_lower)
                        )
                    ):
                        valid_parts += 1

                # If most parts look like column names, this is probably the header
                if valid_parts >= len(parts) * 0.7 and len(parts) > 1:
                    header_line = parts
                    header_row_idx = i
                    separator = "\t" if "\t" in line else r"\s+"
                    break

            if header_line is None or header_row_idx is None:
                return None, None, None

            # Clean the column names
            clean_columns = [col.strip() for col in header_line if col.strip()]

            # Find where the actual data starts (skip any repeated headers)
            data_start = header_row_idx + 1
            while data_start < len(lines):
                line = lines[data_start]
                if not line:
                    data_start += 1
                    continue

                parts = line.split(separator)

                # Check if this line looks like a header repetition
                if len(parts) == len(clean_columns):
                    # If most parts match our column names, skip this line
                    matches = sum(
                        1
                        for i, part in enumerate(parts)
                        if i < len(clean_columns)
                        and part.strip().lower() == clean_columns[i].lower()
                    )
                    if matches >= len(clean_columns) * 0.8:
                        data_start += 1
                        continue

                # Check if this looks like data (contains numbers)
                numeric_count = 0
                for part in parts:
                    try:
                        float(part)
                        numeric_count += 1
                    except:
                        pass

                if numeric_count >= len(parts) * 0.5:  # At least half are numbers
                    break

                data_start += 1

            return clean_columns, data_start, separator

        except Exception as e:
            raise ValueError(f"Failed to detect data structure: {e}") from e

    def _load_text_file(self, path: str | Path, if_check: bool = True) -> pd.DataFrame | None:
        try:
            if if_check:
                detected_columns, data_start_row, separator = self._detect_data_structure(path)
                msg = f"Detected columns: {detected_columns}, data start row: {data_start_row}, separator: {separator}"
                logger.info(msg)
                self._log(msg)
                if detected_columns and data_start_row is not None:
                    df = pd.read_csv(
                        path, sep=separator, header=None, skiprows=data_start_row, encoding="utf-8"
                    )
                    if len(detected_columns) < len(df.columns):
                        df = df.iloc[:, : len(detected_columns)]
                        df.columns = detected_columns
                        self._log(f"Trimmed columns: {df.columns}")
                    elif len(detected_columns) > len(df.columns):
                        df.columns = detected_columns[: len(df.columns)]
                        self._log(
                            f"detected columns more than dataframe columns: {detected_columns}"
                        )
                    else:
                        df.columns = detected_columns
                else:
                    msg = f"Failed to load {path}: No data structure detected"
                    self._log(msg)
                    return None
            else:
                df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")

            df = df.dropna(axis=1, how="all").dropna(how="all")
            df.columns = [str(c).strip() for c in df.columns]
            # best-effort numeric conversion
            for c in df.columns:
                if df[c].dtype == object:
                    try:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    except:
                        pass
            return df
        except Exception as e:
            msg = f"Failed to load {path}: {e}"
            self._log(msg, log_type="error")
            return None

    def _export_csv(self, files: list[str]) -> None:
        if len(files) == 1:
            p, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", files[0] + ".csv", "CSV files (*.csv)"
            )
            if not p:
                return
            df = self.project.get_current_data(files[0])
            if df is not None:
                df.to_csv(p, index=False)
                self._log(f"Exported: {Path(p).name}")
            return
        # multiple
        directory = QFileDialog.getExistingDirectory(self, "Select export directory")
        if not directory:
            return
        out_dir = Path(directory)
        for name in files:
            df = self.project.get_current_data(name)
            if df is not None:
                (out_dir / f"{name}.csv").write_text(df.to_csv(index=False))
        self._log(f"Exported {len(files)} files to {out_dir}")

    def _export_excel(self, files: list[str]) -> None:
        x_col = self.combo_x.currentText()
        y_col = self.combo_y.currentText()
        if not x_col or not y_col:
            QMessageBox.warning(self, "Select columns", "Please select X and Y columns first.")
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Save Excel", "combined.xlsx", "Excel files (*.xlsx)"
        )
        if not p:
            return
        try:
            col_frames: list[pd.DataFrame] = []
            for name in files:
                df = self.project.get_current_data(name)
                if df is None or x_col not in df.columns or y_col not in df.columns:
                    continue
                col_frames.append(
                    pd.DataFrame({f"{name}_{x_col}": df[x_col], f"{name}_{y_col}": df[y_col]})
                )
            if not col_frames:
                QMessageBox.information(self, "No data", "No valid data to export.")
                return
            combined = pd.concat(col_frames, axis=1)
            with pd.ExcelWriter(p, engine="openpyxl", mode="w") as writer:
                combined.to_excel(writer, sheet_name="Combined_Data", index=False)
            self._log(f"Exported Excel: {Path(p).name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_hdf5(self, files: list[str]) -> None:
        p, _ = QFileDialog.getSaveFileName(
            self, "Export Project", "project.h5", "HDF5 files (*.h5)"
        )
        if not p:
            return
        try:
            store = pd.HDFStore(p, mode="w")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to create HDF5: {e}")
            return
        with store:
            # Build metadata with order and hierarchy
            all_items = set(files)
            # include parent folders
            for name in list(files):
                node = self.project.nodes.get(name)
                if node and node.parent_folder:
                    parent = node.parent_folder
                    while parent and parent not in all_items:
                        all_items.add(parent)
                        parent_node = self.project.nodes.get(parent)
                        parent = (
                            parent_node.parent_folder
                            if parent_node and parent_node.parent_folder
                            else None
                        )

            ordered: list[str] = []

            def traverse(parent: QTreeWidgetItem | None) -> None:
                children = (
                    [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
                    if parent is None
                    else [parent.child(i) for i in range(parent.childCount())]
                )
                for it in children:
                    name = self._item_name(it)
                    if name and name in all_items:
                        ordered.append(name)
                    traverse(it)

            traverse(None)
            order_map = {n: i for i, n in enumerate(ordered)}

            meta_rows = []
            for name in all_items:
                node = self.project.nodes.get(name)
                if not node:
                    continue
                meta_rows.append(
                    {
                        "name": node.name,
                        "is_folder": node.is_folder,
                        "parent_folder": node.parent_folder or "",
                        "path": self.project.get_item_path(name),
                        "order": order_map.get(name, 1_000_000),
                    }
                )
            meta_df = pd.DataFrame(meta_rows).sort_values("order")
            store.put("__metadata__", meta_df, format="table")

            for name in files:
                df = self.project.get_current_data(name)
                if df is None:
                    continue
                safe_path = (
                    self.project.get_item_path(name)
                    .replace("\\", "/")
                    .replace("/", "__")
                    .replace(".", "_")
                    .replace(" ", "_")
                )
                key = f"/data/{safe_path}"
                store.put(key, df, format="table")
        self._log(f"Exported HDF5: {Path(p).name}")

    def _import_hdf5(
        self, path: Path, target_folder: str | None, parent_item: QTreeWidgetItem | None
    ) -> None:
        try:
            try:
                import importlib

                importlib.import_module("tables")
            except Exception:  # pragma: no cover - optional dependency
                QMessageBox.critical(
                    self,
                    "Missing Dependency",
                    "The 'tables' package is required for HDF5 support.\nInstall via: pip install tables",
                )
                return
            store = pd.HDFStore(str(path), mode="r")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to open HDF5: {e}")
            return

        with store:
            keys = store.keys()
            if "/__metadata__" in keys:
                meta = store.get("/__metadata__")
                name_to_item: dict[str, QTreeWidgetItem] = {}
                # ensure parent-first ordering
                if "path" in meta.columns:
                    meta["depth"] = meta["path"].astype(str).str.count("/") + 1
                    meta = meta.sort_values(
                        ["depth", "order" if "order" in meta.columns else "depth"]
                    )  # type: ignore[index]
                for _, row in meta.iterrows():
                    item_name = str(row["name"])  # type: ignore[index]
                    is_folder = bool(row["is_folder"])  # type: ignore[index]
                    parent_name = str(row.get("parent_folder", "") or "")
                    if is_folder:
                        node = self.project.add_folder(item_name, parent_name or target_folder)
                        item = QTreeWidgetItem([item_name, "Folder", "-", "-"])
                        node.tree_item = item
                        if parent_name and parent_name in name_to_item:
                            name_to_item[parent_name].addChild(item)
                            name_to_item[parent_name].setExpanded(True)
                        elif parent_item:
                            parent_item.addChild(item)
                            parent_item.setExpanded(True)
                        else:
                            self.tree.addTopLevelItem(item)
                        name_to_item[item_name] = item
                    else:
                        safe_path = (
                            str(row["path"])  # type: ignore[index]
                            .replace("\\", "/")
                            .replace("/", "__")
                            .replace(".", "_")
                            .replace(" ", "_")
                        )
                        data_key = f"/data/{safe_path}"
                        if data_key in keys:
                            df = store.get(data_key)
                            node = self.project.add_file(
                                item_name, df, parent_name or target_folder
                            )
                            item = QTreeWidgetItem(
                                [item_name, "File", "Original", f"{len(df):,} rows"]
                            )
                            node.tree_item = item
                            container = (
                                name_to_item[parent_name]
                                if parent_name in name_to_item
                                else parent_item
                            )
                            if container:
                                container.addChild(item)
                                container.setExpanded(True)
                            else:
                                self.tree.addTopLevelItem(item)
                            name_to_item[item_name] = item
                self._log(f"Loaded HDF5 project: {path.name}")
            else:
                # legacy: put datasets under a folder named after file
                folder = path.stem
                folder_node = self.project.add_folder(folder, target_folder)
                folder_item = QTreeWidgetItem([folder, "Folder", "-", "-"])
                folder_node.tree_item = folder_item
                if parent_item:
                    parent_item.addChild(folder_item)
                    parent_item.setExpanded(True)
                else:
                    self.tree.addTopLevelItem(folder_item)
                for key in keys:
                    if key.startswith("/_"):
                        continue
                    dataset_name = key.strip("/")
                    df = store.get(key)
                    node = self.project.add_file(dataset_name, df, folder)
                    item = QTreeWidgetItem(
                        [dataset_name, "Dataset", "Original", f"{len(df):,} rows"]
                    )
                    node.tree_item = item
                    folder_item.addChild(item)
                    folder_item.setExpanded(True)
                self._log(f"Loaded legacy HDF5: {path.name}")


def gui_easy_data() -> None:
    """
    GUI for easy data manipulation
    """
    app = QApplication.instance() or QApplication(sys.argv)
    w = EasyDataWindow()
    w.show()
    # Avoid sys.exit in interactive environments (e.g., Jupyter) to prevent
    # terminating the hosting kernel/process. Just run the event loop.
    app.exec()


if __name__ == "__main__":  # pragma: no cover
    gui_easy_data()
