from __future__ import annotations

import unittest

from pyomnix.data_process.data_manipulator import DataManipulator


class _StillAliveThread:
    def __init__(self) -> None:
        self.join_timeout = None

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout


class TestPlotSaving(unittest.TestCase):
    def test_start_saving_raises_if_previous_thread_does_not_stop(self) -> None:
        dm = DataManipulator(1)
        existing_thread = _StillAliveThread()
        dm._thread = existing_thread

        with self.assertRaisesRegex(RuntimeError, "Previous save thread did not stop"):
            dm.start_saving("unused.png", time_interval=60)

        self.assertEqual(existing_thread.join_timeout, 10)
        self.assertTrue(dm._stop_event.is_set())
        self.assertIs(dm._thread, existing_thread)


if __name__ == "__main__":
    unittest.main()
