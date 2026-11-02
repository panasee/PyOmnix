from collections.abc import Iterator, Sequence
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet

import pyomnix.data_process.data_manipulator as module
from pyomnix import DataManipulator
from pyomnix.data_process.data_splitter import DataSplitter
from pyomnix.utils import GridData


def test_point_row_and_offline_reconstruction_agree() -> None:
    x, y = [0., 1., 2.], [10., 20.]
    expected = np.array([[0., 1., 2.], [10., 11., 12.]])
    point, row = GridData(x, y), GridData(x, y)
    rows = []
    for iy in range(2):
        for ix in [2, 1, 0]:
            point.update(ix, iy, expected[iy, ix], indices=True)
            rows.append((x[ix], y[iy], expected[iy, ix]))
        row.update(x, y[iy], expected[iy])
    frame = pd.DataFrame(rows, columns=["x", "y", "value"])
    offline = DataManipulator(1).mapping_grid("x", "y", "value", data_df=frame,
                                               x_values=x, y_values=y)
    for grid in [point, row, offline]:
        np.testing.assert_array_equal(grid.z, expected)
        assert grid.complete and grid.coverage == 1
        np.testing.assert_array_equal(grid.counts, 1)
    np.testing.assert_array_equal(point.order, [[2, 1, 0], [5, 4, 3]])
    np.testing.assert_array_equal(row.order, [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(offline.order, point.order)
    np.testing.assert_array_equal(offline.path, point.path)


def test_grid_path_tracks_repeated_visits_and_nan_readings() -> None:
    grid = GridData([2, 1, 0], [20, 10])
    assert grid.path.shape == (0, 2)
    np.testing.assert_array_equal(grid.order, -np.ones((2, 3), dtype=int))
    grid.update(1.00001, 10, 4, tolerance=.0001)
    snapshot = grid.path
    grid.update([0, 1, 0], 1, [5, 6, np.nan], indices=True, overwrite=True)
    np.testing.assert_array_equal(grid.path, [[1, 1], [0, 1], [1, 1], [0, 1]])
    np.testing.assert_array_equal(grid.order, [[-1, -1, -1], [3, 2, -1]])
    np.testing.assert_array_equal(grid.counts, [[0, 0, 0], [2, 2, 0]])
    assert np.isnan(grid.z[1, 0]) and grid.z[1, 1] == 6
    np.testing.assert_array_equal(snapshot, [[1, 1]])
    with pytest.raises(ValueError, match="read-only"):
        grid.order[0, 0] = 10
    with pytest.raises(ValueError, match="read-only"):
        grid.path[0, 0] = 10


def test_grid_path_follows_broadcast_input_order() -> None:
    grid = GridData([0, 1, 2], [10, 20])
    grid.update([2, 0, 1], [[20], [10]], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(grid.path, [[2, 1], [0, 1], [1, 1],
                                             [2, 0], [0, 0], [1, 0]])
    np.testing.assert_array_equal(grid.order, [[4, 5, 3], [1, 2, 0]])


def test_grid_path_rejected_and_empty_updates_do_not_consume_numbers() -> None:
    grid = GridData([0, 1, 2], [0], valid=[[True, True, False]])
    grid.update(0, 0, 7)
    for x, values in [([1, 100], [8, 9]), ([1, 2], [8, 9]),
                      ([1, 1], [8, 9]), ([1, 0], [8, 9]), ([1], [np.inf])]:
        with pytest.raises(ValueError):
            grid.update(x, 0, values)
        np.testing.assert_array_equal(grid.path, [[0, 0]])
        np.testing.assert_array_equal(grid.order, [[0, -1, -1]])
        np.testing.assert_array_equal(grid.counts, [[1, 0, 0]])
        np.testing.assert_array_equal(grid.z, [[7, np.nan, np.nan]])
    grid.update([], [], [])
    grid.update(1, 0, np.nan)
    np.testing.assert_array_equal(grid.path, [[0, 0], [1, 0]])
    np.testing.assert_array_equal(grid.order, [[0, 1, -1]])
    assert grid.complete and grid.coverage == 1


@pytest.mark.parametrize("calibrated", [False, True])
def test_grid_save_load_preserves_state_and_resumes(tmp_path, calibrated) -> None:
    coordinates = np.meshgrid([20., 10., 0.], [200., 100.]) if calibrated else None
    grid = GridData([2, 1, 0], [20, 10], valid=[[True, True, False], [True, True, True]],
                    coordinates=coordinates)
    grid.update([0, 1, 0], 0, [1, 0, 3], indices=True, overwrite=True)
    grid.update(1, 1, np.nan, indices=True)
    filename = tmp_path / "扫描数据.npz"
    grid.save(filename)
    restored = GridData.load(filename)
    for name in ("x", "y", "valid", "z", "counts", "order", "path"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(grid, name))
        assert not getattr(restored, name).flags.writeable
    if calibrated:
        for actual, expected in zip(restored.coordinates, grid.coordinates):
            np.testing.assert_array_equal(actual, expected)
            assert not actual.flags.writeable
    else:
        assert restored.coordinates is None
    assert restored.coverage == grid.coverage and restored.complete == grid.complete
    np.testing.assert_array_equal(restored.order, [[2, 1, -1], [-1, 3, -1]])
    restored.update(0, 0, 9, indices=True, overwrite=True)
    restored.update(2, 1, 5, indices=True)
    assert restored.order[0, 0] == 4 and restored.order[1, 2] == 5
    assert restored.counts[0, 0] == 3 and restored.z[0, 0] == 9
    assert grid.z[0, 0] == 3 and len(grid.path) == 4
    # Saving again replaces the file and keeps the new order gaps intact.
    restored.save(filename)
    np.testing.assert_array_equal(GridData.load(filename).order, restored.order)


@pytest.mark.parametrize("valid", [True, False])
def test_grid_save_load_empty_grid_and_exact_filename(tmp_path, valid) -> None:
    grid = GridData([0], [0], valid=[[valid]])
    filename = tmp_path / "grid"
    grid.save(str(filename))
    assert filename.is_file() and not (tmp_path / "grid.npz").exists()
    restored = GridData.load(str(filename))
    assert restored.path.shape == (0, 2)
    assert restored.order[0, 0] == -1 and restored.counts[0, 0] == 0
    assert np.isnan(restored.z[0, 0])
    assert restored.complete == grid.complete and restored.coverage == grid.coverage


@pytest.mark.parametrize("field,value,message", [
    ("format_version", np.array(2), "format_version"),
    ("format_version", np.array(1.), "format_version"),
    ("z", np.zeros((2, 2)), "grid shape"),
    ("z", np.array([[np.inf, np.nan]]), "infinity"),
    ("z", np.array([[1., 7.]]), "unmeasured"),
    ("z", np.array([["1", "nan"]]), "numeric"),
    ("counts", np.array([[1., 0.]]), "signed integers"),
    ("counts", np.array([[2, 0]]), "inconsistent"),
    ("order", np.array([[5, -1]]), "inconsistent"),
    ("path", np.array([0, 0]), "shape"),
    ("path", np.array([[2, 0]]), "within"),
    ("path", np.array([[-1, 0]]), "within"),
    ("valid", np.array([[False, True]]), "excluded"),
    ("valid", np.array([[1, 1]]), "booleans"),
    ("x", np.array([0., 0.]), "monotonic"),
    ("X", np.zeros((1, 2)), "fields"),
])
def test_grid_load_rejects_invalid_state(tmp_path, field, value, message) -> None:
    filename = tmp_path / "grid.npz"
    grid = GridData([0, 1], [0])
    grid.update(0, 0, 1)
    grid.save(filename)
    with np.load(filename, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    data[field] = value
    np.savez_compressed(filename, **data)
    with pytest.raises(ValueError, match=message):
        GridData.load(filename)


def test_grid_load_rejects_missing_fields_and_pickle(tmp_path) -> None:
    filename = tmp_path / "grid.npz"
    grid = GridData([0], [0])
    grid.save(filename)
    with np.load(filename, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    del data["order"]
    np.savez_compressed(filename, **data)
    with pytest.raises(ValueError, match="fields"):
        GridData.load(filename)
    data["order"] = np.array([[-1]], dtype=object)
    np.savez_compressed(filename, **data)
    with pytest.raises(ValueError, match="allow_pickle=False"):
        GridData.load(filename)
    np.save(tmp_path / "array.npy", [1])
    with pytest.raises(ValueError, match="NPZ archive"):
        GridData.load(tmp_path / "array.npy")


def test_missing_excluded_nan_and_zero_are_distinct() -> None:
    grid = GridData([0, 1, 2], [0], valid=[[True, True, False]])
    grid.update(0, 0, 0)
    assert grid.coverage == .5 and not grid.complete
    assert grid.z[0, 0] == 0 and np.isnan(grid.z[0, 1])
    grid.update(1, 0, np.nan)
    assert grid.complete and grid.counts[0, 1] == 1
    assert grid.counts[0, 2] == 0 and not grid.valid[0, 2]
    with pytest.raises(ValueError, match="excluded"):
        grid.update(2, 0, 7)


def test_validation_is_atomic_and_repeats_are_explicit() -> None:
    grid = GridData([0, 1, 2], [0, 1])
    with pytest.raises(ValueError, match="duplicate"):
        grid.update([0, 0], 0, [1, 2])
    assert not grid.counts.any()
    grid.update([0, 0], 0, [1, 2], overwrite=True)
    assert grid.z[0, 0] == 2 and grid.counts[0, 0] == 2
    with pytest.raises(ValueError):
        grid.update([1, 100], 0, [8, 9])
    assert grid.counts.sum() == 2 and np.isnan(grid.z[0, 1])
    with pytest.raises(ValueError, match="indices"):
        grid.update(.5, 0, 1, indices=True)


def test_descending_axes_tolerance_and_ambiguous_matches() -> None:
    grid = GridData([2, 1, 0], [20, 10])
    grid.update(1.00001, 10, 4, tolerance=.0001)
    assert grid.z[1, 1] == 4
    with pytest.raises(ValueError, match="multiple"):
        grid.update(.5, 10, 7, tolerance=.51)


def test_static_grid_uses_calibrated_coordinates_and_shared_norm() -> None:
    x, y = np.meshgrid([0, 1, 2], [0, 1])
    grid = GridData([0, 1, 2], [0, 1], coordinates=(x + .2*y, y))
    grid.update([0, 1, 2], 0, [1, 2, 3])
    norm = Normalize(0, 10)
    fig, ax = DataManipulator(1).plot_mapping(grid=grid, norm=norm)
    assert ax.collections[0].norm is norm
    assert np.ma.getmaskarray(ax.collections[0].get_array()).sum() == 3
    plt.close(fig)


def test_select_scan_retains_order_and_breaks_at_directions_and_gaps() -> None:
    frame = pd.DataFrame(dict(scan_id=[0]*4 + [1]*2, n=[1, 1.1, .9, 1, 3, 3],
                              direction=[1, 1, -1, -1, 1, -1],
                              time=pd.to_datetime([0, 1, 2, 100, 101, 102], unit="s")))
    selected, meta = DataSplitter.select_scan(frame, "n", 1.1, time_col="time", max_gap=10)
    assert list(selected.index) == [0, 1, 2, 3]
    assert list(selected._plot_segment) == [0, 0, 1, 2]
    assert meta["minimum"] == .9 and meta["maximum"] == 1.1
    assert meta["directions"] == (1, -1)
    with pytest.raises(ValueError, match="equally"):
        DataSplitter.select_scan(frame, "n", 2)
    with pytest.raises(ValueError, match="outside"):
        DataSplitter.select_scan(frame, "n", 4)
    assert "_plot_segment" not in frame


@pytest.mark.parametrize("dtype", ["int64", "Int64", "string", "category"])
def test_select_scan_accepts_nullable_direction_types(dtype: str) -> None:
    frame = pd.DataFrame({
        "scan_id": [0, 0, 0], "n": [1.0, 1.0, 1.0],
        "direction": pd.Series([1, 1, -1], dtype=dtype),
    })
    frame.index = [7, 3, 9]
    original = frame.copy(deep=True)
    selected, metadata = DataSplitter.select_scan(frame, "n", scan_id=0)
    assert selected._plot_segment.tolist() == [0, 0, 1]
    assert selected.index.tolist() == frame.index.tolist()
    assert metadata["directions"] == tuple(frame.direction.unique())
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("dtype", ["Int64", "string"])
def test_select_scan_still_rejects_missing_direction(dtype: str) -> None:
    frame = pd.DataFrame({
        "scan_id": [0, 0], "n": [1.0, 1.0],
        "direction": pd.Series([1, None], dtype=dtype),
    })
    with pytest.raises(ValueError, match="direction labels must not be missing"):
        DataSplitter.select_scan(frame, "n", scan_id=0)


@pytest.mark.parametrize("unit,scale", [("s", 1), ("ms", 1000), ("us", 10**6), ("ns", 10**9)])
def test_select_scan_explicit_numeric_time_units(unit: str, scale: int) -> None:
    frame = pd.DataFrame({
        "scan_id": [0, 0, 0], "n": [1.0, 1.0, 1.0], "direction": [1, 1, 1],
        "time": np.array([0, 1, 100], dtype=float) * scale,
    })
    selected, _ = DataSplitter.select_scan(
        frame, "n", scan_id=0, time_col="time", max_gap=10, time_unit=unit,
    )
    assert selected._plot_segment.tolist() == [0, 0, 1]


@pytest.mark.parametrize("time_kind", ["datetime", "string", "numeric_ns"])
def test_select_scan_preserves_legacy_time_parsing(time_kind: str) -> None:
    timestamps = pd.to_datetime([0, 1, 100], unit="s")
    values = {
        "datetime": timestamps,
        "string": timestamps.astype(str),
        "numeric_ns": np.array([0, 1, 100], dtype=np.int64) * 10**9,
    }[time_kind]
    frame = pd.DataFrame({"scan_id": [0, 0, 0], "n": [1.0]*3, "time": values})
    selected, _ = DataSplitter.select_scan(
        frame, "n", scan_id=0, direction_col=None, time_col="time", max_gap=10,
    )
    assert selected._plot_segment.tolist() == [0, 0, 1]


@pytest.mark.parametrize("values,message", [
    ([0, 100, 1], "acquisition order"),
    ([0, np.nan, 100], "timestamps must not be missing"),
    (["2026-01-01", "2026-01-02", "2026-01-03"], "numeric time column"),
])
def test_select_scan_explicit_time_unit_keeps_validation(values: list, message: str) -> None:
    frame = pd.DataFrame({"scan_id": [0, 0, 0], "n": [1.0]*3, "time": values})
    with pytest.raises(ValueError, match=message):
        DataSplitter.select_scan(
            frame, "n", scan_id=0, direction_col=None,
            time_col="time", max_gap=10, time_unit="s",
        )


@pytest.mark.parametrize("render", ["pcolormesh", "contourf"])
def test_dataframe_and_grid_mapping_share_orientation_and_renderer(render: str) -> None:
    frame = pd.DataFrame({"x": [10, 10, 20, 20], "y": [1, 2, 1, 2], "z": [3, 4, 5, 6]})
    plot = DataManipulator(1)
    grid = plot.mapping_grid("x", "y", "z", data_df=frame)
    np.testing.assert_array_equal(grid.z, [[3, 5], [4, 6]])
    # Exercise the default renderer as well as the explicit contour option.
    kwargs = {} if render == "pcolormesh" else {"render": render}
    for data_kwargs in [{"data_df": frame}, {"grid": grid}]:
        selectors = () if "grid" in data_kwargs else ("x", "y", "z")
        fig, ax = plot.plot_mapping(*selectors, **data_kwargs, **kwargs)
        try:
            if render == "pcolormesh":
                assert isinstance(ax.collections[0], QuadMesh)
                assert ax.get_xlim() == (5, 25) and ax.get_ylim() == (.5, 2.5)
                np.testing.assert_array_equal(ax.collections[0].get_array(), grid.z)
            else:
                assert isinstance(ax.collections[0], QuadContourSet)
                assert ax.get_xlim() == (10, 20) and ax.get_ylim() == (1, 2)
        finally:
            plt.close(fig)


@pytest.mark.parametrize("shape", [(1, 3), (2, 1), (1, 1)])
@pytest.mark.parametrize("calibrated", [False, True])
def test_static_singleton_grid_rejected_before_plot_mutation(
    shape: tuple[int, int], calibrated: bool,
) -> None:
    rows, cols = shape
    x, y = np.arange(cols), np.arange(rows)
    coordinates = np.meshgrid(x + 10, y + 20) if calibrated else None
    grid = GridData(x, y, coordinates=coordinates)
    grid.update(x, y[:, None], 7)
    plot = DataManipulator(1)
    fig, ax = plt.subplots()
    try:
        figures_before = plt.get_fignums()
        for kwargs in [{}, {"fig": fig, "ax": ax}]:
            with pytest.raises(ValueError, match="at least two"):
                plot.plot_mapping(grid=grid, **kwargs)
            assert plt.get_fignums() == figures_before
            assert not ax.collections and len(fig.axes) == 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize("selection", [{"scan_id": ("run", 0)}, {"target": 1.1}])
@pytest.mark.parametrize("tuple_columns", [False, True])
def test_select_scan_supports_tuple_ids_and_columns(selection: dict, tuple_columns: bool) -> None:
    parameter = ("gate", "V") if tuple_columns else "n"
    scan_col = ("acquisition", "scan") if tuple_columns else "scan_id"
    frame = pd.DataFrame({
        scan_col: [("run", 0), ("run", 0), ("run", 1), ("run", 1)],
        parameter: [1.0, 1.2, 2.0, 2.2],
    }, index=[7, 3, 9, 1])
    original = frame.copy(deep=True)
    selected, metadata = DataSplitter.select_scan(
        frame, parameter, scan_col=scan_col, direction_col=None, **selection,
    )
    assert selected.index.tolist() == [7, 3]
    assert selected[scan_col].tolist() == [("run", 0), ("run", 0)]
    assert metadata["scan_id"] == ("run", 0)
    assert metadata["actual"] == pytest.approx(1.1)
    assert metadata["minimum"] == 1.0 and metadata["maximum"] == 1.2
    assert selected["_plot_segment"].tolist() == [0, 0]
    pd.testing.assert_frame_equal(frame, original)
