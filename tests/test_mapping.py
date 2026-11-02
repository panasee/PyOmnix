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


def live(types: Sequence[str], **kwargs: Any) -> DataManipulator:
    plot = DataManipulator(1)
    plot.live_plot_init(1, len(types), 1, plot_types=[types], show=False, **kwargs)
    return plot


@pytest.mark.parametrize("plot_type", ["heatmap", "contour"])
@pytest.mark.parametrize("shape", [(2, 3), (1, 3), (3, 1), (1, 1)])
def test_live_map_accepts_matrix(
    plot_type: str, shape: tuple[int, int],
) -> None:
    plot = live([plot_type])
    rows, cols = shape
    x, y = np.arange(cols), np.arange(rows)
    z = np.arange(rows * cols).reshape(shape)
    plot.live_plot_update(0, 0, 0, x, y, z)
    np.testing.assert_array_equal(plot.go_f.data[0].z, z)
    before = plot.go_f.to_json()
    with pytest.raises(ValueError, match="shape"):
        plot.live_plot_update(0, 0, 0, x, y, np.zeros((rows + 1, cols)))
    assert plot.go_f.to_json() == before


def test_live_ragged_maps_and_lines_share_atomic_batch() -> None:
    plot = live(["heatmap", "contour", "scatter"])
    plot.live_plot_update((0, 0, 0), (0, 1, 2), (0, 0, 0),
                          [[0, 1, 2], [0, 1], [0, 1, 2, 3]],
                          [[0, 1], [0, 1, 2], [2, 3, 4, 5]],
                          [np.ones((2, 3)), np.ones((3, 2))])
    assert np.shape(plot.go_f.data[0].z) == (2, 3)
    before = plot.go_f.to_json()
    with pytest.raises(ValueError, match="match"):
        plot.live_plot_update((0, 0), (0, 2), (0, 0), [[0, 1], [0, 1]],
                              [[0, 1], [0]], np.zeros((2, 2)))
    assert plot.go_f.to_json() == before


def test_live_grid_preserves_holes_and_does_not_smooth() -> None:
    a, b = GridData([0, 1, 2], [0, 1]), GridData([0, 1], [0, 1])
    a.update(1, 1, 5)
    plot = live(["heatmap", "contour"])
    plot.live_plot_update((0, 0), (0, 1), (0, 0), [a, b])
    assert np.isnan(plot.go_f.data[0].z[0, 0])
    assert plot.go_f.data[0].z[1, 1] == 5
    assert plot.go_f.data[0].zsmooth is False
    assert plot.go_f.data[1].connectgaps is False


def test_line_window_does_not_disable_map_hover() -> None:
    plot = live(["heatmap", "scatter"], disable_hover_threshold=2)
    grid = GridData([0, 1], [0, 1])
    grid.update([0, 1], 0, [2, 3])
    plot.live_plot_update(0, 0, 0, grid)
    grid.update(0, 1, 4)
    plot.live_plot_update(0, 0, 0, grid)
    np.testing.assert_array_equal(plot.go_f.data[0].z, [[2, 3], [4, np.nan]])
    plot.live_plot_update(0, 1, 0, [0, 1, 2], [3, 4, 5])
    plot.live_plot_update(0, 1, 0, [3, 4], [6, 7], incremental=True, max_points=3)
    np.testing.assert_array_equal(plot.go_f.data[1].x, [2, 3, 4])
    assert plot.go_f.layout.hovermode is not False
    assert plot.go_f.data[0].hoverinfo != "skip"


def test_grid_widget_uses_null_for_missing_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "is_notebook", lambda: True)
    monkeypatch.setattr("IPython.display.display", lambda figure: None)
    plot = DataManipulator(1)
    plot.live_plot_init(1, 1, 1, plot_types=[["heatmap"]])
    grid = GridData([0, 1], [0, 1])
    grid.update(0, 0, 2)
    plot.live_plot_update(0, 0, 0, grid)
    assert plot.go_f.data[0].z[1][1] is None
    assert plot.go_f.data[0].z[0][0] == 2
    assert np.isnan(grid.z[1, 1])
    plot.go_f.close()


def test_map_colorbars_follow_subplot_domains() -> None:
    plot = DataManipulator(1)
    plot.live_plot_init(2, 2, 1, plot_types=[["heatmap", "heatmap"], ["contour", "contour"]], show=False)
    bars = [trace.colorbar for trace in plot.go_f.data]
    assert bars[0].x < bars[1].x
    assert bars[0].y > bars[2].y
    assert all(bar.len < .5 for bar in bars)


def test_legacy_multiline_strings_and_clearing() -> None:
    plot = live(["scatter", "scatter"])
    plot.live_plot_update((0, 0), (0, 1), (0, 0), [["a", "b"], ["c"]],
                          [[1, 2], [3]], with_str=True)
    plot.live_plot_update((0, 0), (0, 1), (0, 0), ["d", "e"], [4, 5],
                          incremental=True, with_str=True)
    assert list(plot.go_f.data[0].x) == ["a", "b", "d"]
    plot.live_plot_update(0, 0, 0, [], [])
    assert len(plot.go_f.data[0].x) == 0


def test_dash_snapshot_retains_direct_edits_and_zoom(monkeypatch: pytest.MonkeyPatch) -> None:
    class IdleThread:
        def __init__(self, **kwargs: Any) -> None:
            pass
        def start(self) -> None:
            pass
    plot = live(["heatmap"])
    monkeypatch.setattr(module, "threading", SimpleNamespace(Thread=IdleThread))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    plot.create_dash(plot.go_f, refresh_interval_ms=350)
    callback = next(iter(plot._dash_app.callback_map.values()))["callback"].__wrapped__
    plot.go_f.data[0].update(x=[0, 1], y=[0, 1], z=[[1, 2], [3, 4]])
    result = callback(1, {"xaxis.range[0]": .2, "xaxis.range[1]": .8})
    np.testing.assert_array_equal(result.data[0].z, [[1, 2], [3, 4]])
    assert tuple(result.layout.xaxis.range) == (.2, .8)
    assert plot._dash_app.layout.children[1].interval == 350
    # A later external mutation remains visible without a managed-dirty flag.
    plot.live_dfs[0][0][0].z = [[9, 2], [3, 4]]
    assert callback(2, None).data[0].z[0][0] == 9


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


def test_live_grid_batch_can_shrink_to_one_trace() -> None:
    plot = live(["heatmap", "contour"])
    first, second = GridData([0, 1], [0]), GridData([0, 1], [0])
    first.update([0, 1], 0, [1, 2])
    plot.live_plot_update((0, 0), (0, 1), (0, 0), [first, second])
    second.update([0, 1], 0, [3, 4])
    plot.live_plot_update((0,), (1,), (0,), second)
    np.testing.assert_array_equal(plot.go_f.data[0].z, first.z)
    np.testing.assert_array_equal(plot.go_f.data[1].z, second.z)


@pytest.mark.parametrize("webgl", ["off", "on"])
def test_single_line_wrappers_append_window_and_clear(webgl: str) -> None:
    plot = live(["scatter"], webgl=webgl)
    plot.live_plot_update((0,), (0,), (0,), [np.array([0, 1])], [[10, 11]])
    plot.live_plot_update(0, 0, 0, 2, 12, incremental=True)
    plot.live_plot_update((0,), (0,), (0,), [[3, 4]], [[13, 14]],
                          incremental=True, max_points=3)
    np.testing.assert_array_equal(plot.go_f.data[0].x, [2, 3, 4])
    np.testing.assert_array_equal(plot.go_f.data[0].y, [12, 13, 14])
    plot.live_plot_update(0, 0, 0, [], [])
    assert len(plot.go_f.data[0].x) == 0


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("with_str", [False, True])
def test_line_missing_value_vectors_remain_valid(wrapped: bool, with_str: bool) -> None:
    plot = live(["scatter"])
    x, y = ([[0]], [[None]]) if wrapped else ([0], [None])
    plot.live_plot_update((0,), (0,), (0,), x, y, with_str=with_str)
    plot.live_plot_update(0, 0, 0, [1], [None], incremental=True, with_str=with_str)
    np.testing.assert_array_equal(plot.go_f.data[0].x, [0, 1])
    assert np.isnan(np.asarray(plot.go_f.data[0].y, dtype=float)).all()


def candle_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=2),
        "open": [2, 3], "high": [4, 5], "low": [1, 2], "close": [3, 4],
    })


def test_candlestick_replacement_and_wrappers_remain_compatible() -> None:
    plot = live(["candlestick", "candlestick"])
    frame = candle_frame()
    plot.live_plot_update(0, 0, 0, frame)
    plot.live_plot_update((0,), (0,), (0,), [frame], incremental=True)
    assert len(plot.go_f.data[0].x) == len(frame)
    plot.live_plot_update((0, 0), (0, 1), (0, 0), [[frame], [frame]])
    for trace in plot.go_f.data:
        np.testing.assert_array_equal(trace.close, frame.close)


def test_mixed_grid_raw_map_line_and_candle_share_input_normalization() -> None:
    plot = live(["scatter", "heatmap", "contour", "candlestick"])
    grid = GridData([0, 1], [0])
    grid.update([0, 1], 0, [5, 6])
    matrix = np.arange(6).reshape(3, 2)
    frame = candle_frame()
    plot.live_plot_update(
        (0, 0, 0, 0), (0, 1, 2, 3), (0, 0, 0, 0),
        [[0, 1], grid, [0, 1], frame],
        [[2, 3], None, [0, 1, 2], None], matrix,
    )
    np.testing.assert_array_equal(plot.go_f.data[0].y, [2, 3])
    np.testing.assert_array_equal(plot.go_f.data[1].z, grid.z)
    np.testing.assert_array_equal(plot.go_f.data[2].z, matrix)
    np.testing.assert_array_equal(plot.go_f.data[3].close, frame.close)


@pytest.fixture(params=[False, True], ids=["figure", "widget"])
def metadata_plot(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[DataManipulator]:
    monkeypatch.setattr(module, "is_notebook", lambda: True)
    monkeypatch.setattr("IPython.display.display", lambda figure: None)
    plot = DataManipulator(1)
    plot.live_plot_init(1, 1, 1, plot_types=[["heatmap"]], show=request.param)
    yield plot
    if request.param:
        plot.go_f.close()


@pytest.mark.parametrize("replacement", ["grid", "raw"])
def test_grid_axes_follow_data_replacement(
    metadata_plot: DataManipulator, replacement: str,
) -> None:
    plot = metadata_plot
    calibrated = GridData([0, 1], [0, 1], coordinates=np.meshgrid([10, 20], [30, 40]))
    plot.live_plot_update(0, 0, 0, calibrated)
    np.testing.assert_array_equal(plot.go_f.data[0].x, [10, 20])
    np.testing.assert_array_equal(plot.go_f.data[0].y, [30, 40])
    assert plot.go_f.data[0].customdata is None
    next_grid = GridData([4, 5, 6], [7])
    next_grid.update([4, 5, 6], 7, [8, 9, 10])
    if replacement == "grid":
        plot.live_plot_update(0, 0, 0, next_grid)
    else:
        plot.live_plot_update(0, 0, 0, next_grid.x, next_grid.y, next_grid.z)
    assert plot.go_f.data[0].customdata is None
    np.testing.assert_array_equal(plot.go_f.data[0].x, next_grid.x)
    np.testing.assert_array_equal(plot.go_f.data[0].y, next_grid.y)
    np.testing.assert_array_equal(plot.go_f.data[0].z, next_grid.z)
    recalibrated = GridData([4, 5, 6], [7], coordinates=np.meshgrid([40, 50, 60], [70]))
    plot.live_plot_update(0, 0, 0, recalibrated)
    np.testing.assert_array_equal(plot.go_f.data[0].x, [40, 50, 60])
    np.testing.assert_array_equal(plot.go_f.data[0].y, [70])
    assert plot.go_f.data[0].customdata is None


@pytest.mark.parametrize("after_calibration", [False, True])
def test_grid_updates_preserve_caller_customdata(
    metadata_plot: DataManipulator, after_calibration: bool,
) -> None:
    plot = metadata_plot
    if after_calibration:
        grid = GridData([0, 1], [0, 1], coordinates=np.meshgrid([10, 20], [30, 40]))
        plot.live_plot_update(0, 0, 0, grid)
    manual = np.array([["a", "b"], ["c", "d"]])
    plot.go_f.data[0].customdata = manual
    calibrated = GridData([0, 1], [0, 1], coordinates=np.meshgrid([10, 20], [30, 40]))
    plot.live_plot_update(0, 0, 0, calibrated)
    np.testing.assert_array_equal(plot.go_f.data[0].customdata, manual)
    plot.live_plot_update(0, 0, 0, GridData([0, 1], [0, 1]))
    np.testing.assert_array_equal(plot.go_f.data[0].customdata, manual)
    plot.live_plot_update(0, 0, 0, [0, 1], [0, 1], np.ones((2, 2)))
    np.testing.assert_array_equal(plot.go_f.data[0].customdata, manual)


def test_failed_batch_preserves_calibrated_grid() -> None:
    plot = live(["heatmap", "scatter"])
    grid = GridData([0, 1], [0, 1], coordinates=np.meshgrid([10, 20], [30, 40]))
    plot.live_plot_update(0, 0, 0, grid)
    before = plot.go_f.to_json()
    replacement = GridData([0, 1], [0, 1])
    with pytest.raises(ValueError, match="line x/y lengths must match"):
        plot.live_plot_update((0, 0), (0, 1), (0, 0), [replacement, [0, 1]], [None, [0]])
    assert plot.go_f.to_json() == before
    plot.live_plot_update(0, 0, 0, replacement)
    np.testing.assert_array_equal(plot.go_f.data[0].x, replacement.x)
    np.testing.assert_array_equal(plot.go_f.data[0].y, replacement.y)


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


@pytest.mark.parametrize("plot_type", ["heatmap", "contour"])
@pytest.mark.parametrize("widget", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_live_grid_uses_calibrated_axes(
    monkeypatch: pytest.MonkeyPatch, plot_type: str, widget: bool, descending: bool,
) -> None:
    monkeypatch.setattr(module, "is_notebook", lambda: True)
    monkeypatch.setattr("IPython.display.display", lambda figure: None)
    plot = DataManipulator(1)
    plot.live_plot_init(1, 1, 1, plot_types=[[plot_type]], show=widget)
    calibrated_x = [30., 20., 10.] if descending else [10., 20., 30.]
    calibrated_y = [50., 40.] if descending else [40., 50.]
    grid = GridData([0, 1, 2], [0, 1],
                    coordinates=np.meshgrid(calibrated_x, calibrated_y))
    grid.update(1, 0, 7)
    try:
        plot.live_plot_update(0, 0, 0, grid)
        trace = plot.go_f.data[0]
        np.testing.assert_array_equal(trace.x, calibrated_x)
        np.testing.assert_array_equal(trace.y, calibrated_y)
        np.testing.assert_array_equal(np.asarray(trace.z, dtype=float), grid.z)
        # Calibration affects plotting only, not how measurements find cells.
        np.testing.assert_array_equal(grid.x, [0, 1, 2])
        np.testing.assert_array_equal(grid.y, [0, 1])
    finally:
        if widget:
            plot.go_f.close()


@pytest.mark.parametrize("plot_type", ["heatmap", "contour"])
@pytest.mark.parametrize("invalid", ["sheared_x", "sheared_y", "folded_x", "collapsed_y"])
def test_live_grid_rejects_unrepresentable_calibration_atomically(
    plot_type: str, invalid: str,
) -> None:
    plot = live(["scatter", plot_type])
    calibrated = GridData([0, 1, 2], [0, 1],
                          coordinates=np.meshgrid([10., 20., 30.], [40., 50.]))
    plot.live_plot_update(0, 1, 0, calibrated)
    before = plot.go_f.to_json()
    x, y = (a.copy() for a in calibrated.mesh())
    if invalid == "sheared_x":
        x[1] += .2
    elif invalid == "sheared_y":
        y[:, 1] += .2
    elif invalid == "folded_x":
        x[:] = [10., 30., 20.]
    else:
        y[:] = 40.
    invalid_grid = GridData([0, 1, 2], [0, 1], coordinates=(x, y))
    with pytest.raises(ValueError, match="calibrated coordinates"):
        plot.live_plot_update((0, 0), (0, 1), (0, 0),
                              [[0, 1], invalid_grid], [[1, 2], None])
    assert plot.go_f.to_json() == before
    # A subsequent valid update must still replace the previous calibration.
    plot.live_plot_update(0, 1, 0, GridData([0, 1, 2], [0, 1]))
    np.testing.assert_array_equal(plot.go_f.data[1].x, [0, 1, 2])
    np.testing.assert_array_equal(plot.go_f.data[1].y, [0, 1])


@pytest.mark.parametrize("webgl", ["off", "on"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_legacy_incremental_strings_work_without_with_str(webgl: str, wrapped: bool) -> None:
    plot = live(["scatter"], webgl=webgl)
    for i in range(3):
        x, y = [f"2026-01-0{i + 1}"], [i]
        if wrapped:
            x, y = [x], [y]
        plot.live_plot_update(0, 0, 0, x, y, incremental=True, max_points=2)
    np.testing.assert_array_equal(plot.go_f.data[0].x, ["2026-01-02", "2026-01-03"])
    np.testing.assert_array_equal(plot.go_f.data[0].y, [1, 2])


def test_legacy_incremental_line_numeric_strings() -> None:
    plot = live(["scatter"])
    for value in ["1.5", "2.5"]:
        plot.live_plot_update(0, 0, 0, [value], [value], incremental=True)
    np.testing.assert_array_equal(plot.go_f.data[0].x, [1.5, 2.5])
    np.testing.assert_array_equal(plot.go_f.data[0].y, [1.5, 2.5])
    before = plot.go_f.to_json()
    with pytest.raises(TypeError):
        plot.live_plot_update(0, 0, 0, [object()], [1], incremental=True)
    assert plot.go_f.to_json() == before


@pytest.mark.parametrize("plot_type", ["heatmap", "contour"])
@pytest.mark.parametrize("invalid", ["wrapped_z", "triplets", "scalar_z", "wrapped_axes", "edges"])
def test_maps_reject_ambiguous_or_nonmatrix_inputs(plot_type: str, invalid: str) -> None:
    plot = live(["scatter", plot_type])
    x, y, z = [0, 1], [0, 1], np.ones((2, 2))
    if invalid == "wrapped_z":
        z = [z]
    elif invalid == "triplets":
        z = [1, 2]
    elif invalid == "scalar_z":
        x, y, z = [0], [0], 1
    elif invalid == "wrapped_axes":
        x = [x]
    else:
        x = [0, 1, 2]
    before = plot.go_f.to_json()
    with pytest.raises(ValueError, match="map"):
        plot.live_plot_update((0, 0), (0, 1), (0, 0), [[0, 1], x], [[2, 3], y], z)
    assert plot.go_f.to_json() == before


@pytest.mark.parametrize("plot_type", ["heatmap", "contour"])
@pytest.mark.parametrize("use_grid", [False, True])
def test_maps_replace_data_with_fixed_orientation(plot_type: str, use_grid: bool) -> None:
    plot = live([plot_type])
    grid = GridData([0, 1, 2], [0, 1])
    grid.update([0, 1, 2], 0, [1, 2, 3])
    args = (grid,) if use_grid else (grid.x, grid.y, grid.z)
    before = plot.go_f.to_json()
    with pytest.raises(ValueError, match="incremental=False"):
        plot.live_plot_update(0, 0, 0, *args, incremental=True)
    assert plot.go_f.to_json() == before
    plot.go_f.data[0].transpose = True
    plot.live_plot_update(0, 0, 0, *args)
    assert plot.go_f.data[0].transpose is False
    np.testing.assert_array_equal(plot.go_f.data[0].z, grid.z)


@pytest.mark.parametrize("webgl", ["off", "on"])
@pytest.mark.parametrize("with_str", [False, True])
@pytest.mark.parametrize("kind", ["date", "datetime", "timestamp", "numpy"])
def test_incremental_line_date_coordinates(kind: str, with_str: bool, webgl: str) -> None:
    plot = live(["scatter"], webgl=webgl)
    expected = pd.date_range("2026-01-01", periods=3)
    for i, timestamp in enumerate(expected):
        value = {
            "date": date(2026, 1, i + 1),
            "datetime": datetime(2026, 1, i + 1),
            "timestamp": timestamp,
            "numpy": timestamp.to_datetime64(),
        }[kind]
        x = np.array([value]) if kind == "numpy" else [value]
        plot.live_plot_update(0, 0, 0, x, [i], incremental=True,
                              with_str=with_str, max_points=2)
    np.testing.assert_array_equal(pd.to_datetime(plot.go_f.data[0].x), expected[1:])
    np.testing.assert_array_equal(plot.go_f.data[0].y, [1, 2])
    assert "2026-01-02" in plot.go_f.to_json()


@pytest.mark.parametrize("numpy_first", [False, True])
@pytest.mark.parametrize("with_str", [False, True])
@pytest.mark.parametrize("object_array", [False, True])
def test_incremental_lines_mix_date_representations_without_integer_timestamps(
    numpy_first: bool, with_str: bool, object_array: bool,
) -> None:
    plot = live(["scatter"])
    dates = pd.date_range("2026-01-01T00:00:00.000123", periods=2)
    for i, timestamp in enumerate(dates):
        use_numpy = (i == 0) == numpy_first
        x = np.array([timestamp.to_datetime64()]) if use_numpy else [timestamp]
        if use_numpy and object_array:
            x = np.fromiter(x, dtype=object)
        plot.live_plot_update(0, 0, 0, x, [i], incremental=True, with_str=with_str)
    assert all(isinstance(value, (date, np.datetime64)) for value in plot.go_f.data[0].x)
    np.testing.assert_array_equal(pd.to_datetime(plot.go_f.data[0].x), dates)
    serialized = plot.go_f.to_json()
    assert "2026-01-01T00:00:00.000123" in serialized
    assert "2026-01-02T00:00:00.000123" in serialized


def test_incremental_line_accepts_existing_numpy_date_trace() -> None:
    plot = live(["scatter"])
    plot.go_f.data[0].update(x=np.array(["2026-01-01"], dtype="datetime64[ns]"), y=[0])
    plot.live_plot_update(0, 0, 0, [pd.Timestamp("2026-01-02")], [1], incremental=True)
    assert all(isinstance(value, (date, np.datetime64)) for value in plot.go_f.data[0].x)


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
