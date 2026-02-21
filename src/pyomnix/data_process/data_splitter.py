"""Data splitting utilities for field sweep datasets.

This module extracts reusable, non-GUI logic from the previous tkinter
prototype and adapts it into a typed, testable utility class that can be
imported by PyQt GUIs or other backends.

Key features:
- Detect sweep direction changes from a field-like column
- Split a long dataframe into multiple sweeps (segments)
- Provide helpers to label direction and parameter values for filenames

Design notes:
- Pure logic with no UI or filesystem side effects
- Verbose logging to assist debugging and unit tests

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..omnix_logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SweepSegment:
    """A detected sweep segment.

    Attributes:
        start_index: Inclusive start row index (integer position) in the dataframe
        end_index: Exclusive end row index (integer position) in the dataframe
        direction: "up" if the field increases overall, otherwise "down"
    """

    start_index: int
    end_index: int
    direction: str


class DataSplitter:
    """Detect and split multiple field sweeps in tabular datasets.

    Typical usage:
        splitter = DataSplitter()
        segments = splitter.detect_sweep_segments(df, field_col="B", min_points=10)
        split_dfs = splitter.split_dataframe_by_segments(df, segments)
    """

    def detect_sweep_segments(
        self,
        df: pd.DataFrame,
        field_col: str,
        *,
        min_points: int = 10,
        smoothing_window: int | None = None,
    ) -> list[SweepSegment]:
        """Detect sweep direction changes using a smoothed derivative sign.

        Args:
            df: Input dataframe containing the field column
            field_col: Column name for the field (e.g., magnetic field B)
            min_points: Minimum length (rows) for each resulting segment
            smoothing_window: Optional window for rolling mean smoothing of
                first-order differences; if None, an adaptive window is used

        Returns:
            List of SweepSegment entries covering the dataframe end-to-end.

        Notes:
            - NaNs in the field column are dropped before processing
            - If no robust direction change is detected, returns a single
              segment spanning the whole dataframe
        """

        if field_col not in df.columns:
            logger.error("Field column '%s' not found in dataframe", field_col)
            return []

        work = df[[field_col]].copy()
        work[field_col] = pd.to_numeric(work[field_col], errors="coerce")
        col = work[field_col]
        vals = np.asarray(col)
        mask = pd.notna(vals)
        work = work.loc[mask].reset_index(drop=True)

        n_rows = len(work)
        if n_rows < max(2 * min_points, 5):
            # Not enough rows to form multiple segments; single segment only
            logger.debug(
                "Too few rows (%d) for multi-sweep detection; returning one segment",
                n_rows,
            )
            direction = self._overall_direction(work[field_col].to_numpy())
            return [SweepSegment(0, n_rows, direction)]

        # First derivative and smoothing to reduce noise
        diff = work[field_col].diff()
        if smoothing_window is None:
            smoothing_window = max(3, min(5, n_rows // 10))
        diff_smooth = diff.rolling(window=smoothing_window, center=True).mean()

        # Detect direction sign changes robustly
        direction_changes: list[int] = []
        cur_sign: int | None = None
        values = diff_smooth.to_numpy()
        for i in range(1, len(values)):
            v = values[i]
            if np.isnan(v) or v == 0:
                continue
            new_sign = 1 if v > 0 else -1
            if cur_sign is None:
                cur_sign = new_sign
                continue
            if new_sign != cur_sign:
                direction_changes.append(i)
                cur_sign = new_sign

        # Seed boundaries with start and end, then filter out short segments
        boundaries = [0, *direction_changes, n_rows]
        filtered = [boundaries[0]]
        for idx in range(1, len(boundaries)):
            if boundaries[idx] - filtered[-1] >= min_points:
                filtered.append(boundaries[idx])
            else:
                logger.debug(
                    "Filtered small segment: [%d, %d) length=%d < min_points=%d",
                    filtered[-1],
                    boundaries[idx],
                    boundaries[idx] - filtered[-1],
                    min_points,
                )

        # Ensure last boundary covers the tail adequately
        if filtered[-1] != n_rows:
            if n_rows - filtered[-1] >= min_points:
                filtered.append(n_rows)
            else:
                # Merge tail into the previous segment
                filtered[-1] = n_rows

        # Build segments with direction labels
        segments: list[SweepSegment] = []
        for start, end in zip(filtered[:-1], filtered[1:], strict=False):
            direction = self._segment_direction(work[field_col].to_numpy(), start, end)
            segments.append(SweepSegment(start, end, direction))

        if not segments:
            # Fallback: whole range as one segment
            direction = self._overall_direction(work[field_col].to_numpy())
            segments = [SweepSegment(0, n_rows, direction)]

        logger.debug("Detected %d sweep segment(s)", len(segments))
        return segments

    def split_dataframe_by_segments(
        self,
        df: pd.DataFrame,
        segments: Sequence[SweepSegment],
        *,
        reset_index: bool = True,
    ) -> list[pd.DataFrame]:
        """Split dataframe into multiple dataframes using provided segments.

        Args:
            df: Original dataframe (not modified)
            segments: Segment definitions w.r.t. the dataframe order after
                NaN drop and index reset performed in detection
            reset_index: If True, reset index for each segment dataframe

        Returns:
            List of dataframes, each corresponding to a segment
        """

        # To keep the behavior predictable, mirror the detection preprocessing
        work = df.copy()
        work = work.reset_index(drop=True)
        split_dfs: list[pd.DataFrame] = []
        for seg in segments:
            part = work.iloc[seg.start_index : seg.end_index]
            if reset_index:
                part = part.reset_index(drop=True)
            split_dfs.append(part)
        return split_dfs

    def create_sweep_filenames(
        self,
        base_name: str,
        *,
        index: int,
        direction: str,
        original_ext: str | None = None,
        param_label: str | None = None,
        param_value: float | None = None,
    ) -> str:
        """Create a readable filename for a split sweep.

        Examples:
            base_name="sample", index=1, direction="up" -> sample_sweep1_up
            with param: T=4.20_sample_sweep1_down.csv
        """

        prefix = ""
        if param_label is not None and param_value is not None:
            # Pretty formatting across scales
            if abs(param_value) >= 1000 or (abs(param_value) < 0.01 and param_value != 0):
                prefix = f"{param_label}={param_value:.2e}_"
            else:
                prefix = f"{param_label}={param_value:.3f}_"

        name = f"{prefix}{base_name}_sweep{index}_{direction}"
        if original_ext:
            name = f"{name}.{original_ext}"
        return name

    @staticmethod
    def estimate_param_value(series: pd.Series) -> float | None:
        """Estimate a representative parameter value for a segment.

        Uses mean of numeric coercion; returns None if not meaningful.
        """

        try:
            vals = pd.Series(pd.to_numeric(series, errors="coerce")).dropna()
            return float(vals.mean()) if not vals.empty else None
        except Exception:
            return None

    @staticmethod
    def _overall_direction(values: np.ndarray) -> str:
        return "up" if float(values[-1]) > float(values[0]) else "down"

    @staticmethod
    def _segment_direction(values: np.ndarray, start: int, end: int) -> str:
        start_v = float(values[start])
        end_v = float(values[end - 1])
        return "up" if end_v > start_v else "down"


__all__ = ["DataSplitter", "SweepSegment"]


