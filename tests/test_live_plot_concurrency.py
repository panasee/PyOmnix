"""Regression tests for live updates racing with Plotly figure snapshots."""
from queue import Queue
from threading import Event, Lock, Thread

import numpy as np
import plotly.graph_objects as go
import pytest
from _plotly_utils.basevalidators import BaseDataValidator

from pyomnix import DataManipulator


@pytest.mark.parametrize("webgl", ["off", "on"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_live_update_waits_for_snapshot_type_restoration(monkeypatch, webgl, wrapped):
    plot = DataManipulator(1)
    plot.live_plot_init(1, 1, 1, show=False, webgl=webgl)
    props = plot.go_f.data[0]._props
    copying, resume = Event(), Event()
    progress, errors = Queue(), Queue()
    original_get_class = BaseDataValidator.get_trace_class

    class ObservedLock:
        def __init__(self):
            self.lock = Lock()

        def __enter__(self):
            if copying.is_set():
                progress.put("waiting")
            self.lock.acquire()
            return self

        def __exit__(self, *args):
            self.lock.release()

    plot._figure_lock = ObservedLock()

    def pause_copy(validator, trace_name):
        # Plotly temporarily removes the source trace's type during copying.
        if "type" not in props and not copying.is_set():
            copying.set()
            if not resume.wait(5):
                raise TimeoutError("snapshot was not released")
        return original_get_class(validator, trace_name)

    monkeypatch.setattr(BaseDataValidator, "get_trace_class", pause_copy)

    def snapshot():
        try:
            with plot._figure_lock:
                go.Figure(plot.go_f)
        except Exception as exc:
            errors.put(exc)

    def update():
        try:
            x, y = ([[1.]], [[2.]]) if wrapped else (1., 2.)
            plot.live_plot_update(0, 0, 0, x, y, incremental=True)
        except Exception as exc:
            errors.put(exc)
            progress.put(exc)

    saver = Thread(target=snapshot, daemon=True)
    updater = Thread(target=update, daemon=True)
    saver.start()
    try:
        assert copying.wait(5), "snapshot did not reach the missing-type window"
        updater.start()
        assert progress.get(timeout=5) == "waiting"
    finally:
        resume.set()
        saver.join(5)
        if updater.ident is not None:
            updater.join(5)

    assert not saver.is_alive() and not updater.is_alive(), "plot lock deadlocked"
    assert errors.empty(), list(errors.queue)
    assert plot.go_f.data[0].type == ("scattergl" if webgl == "on" else "scatter")
    np.testing.assert_array_equal(plot.go_f.data[0].x, [1.])
    np.testing.assert_array_equal(plot.go_f.data[0].y, [2.])


def test_invalid_live_update_releases_figure_lock():
    plot = DataManipulator(1)
    plot.live_plot_init(1, 1, 1, show=False)
    before = plot.go_f.to_json()

    with pytest.raises(ValueError, match="lengths"):
        plot.live_plot_update(0, 0, 0, [1., 2.], [3.])

    assert plot.go_f.to_json() == before
    acquired = plot._figure_lock.acquire(blocking=False)
    assert acquired, "validation error left the figure locked"
    plot._figure_lock.release()
    plot.live_plot_update(0, 0, 0, 1., 2.)
    np.testing.assert_array_equal(plot.go_f.data[0].y, [2.])
