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

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

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


class ScanMetadata(TypedDict):
    """Description of the acquired path returned by ``select_scan``.

    ``requested`` is the target parameter, or None when selecting by scan ID.
    ``actual`` is the scan mean; ``minimum``/``maximum`` describe its drift.
    ``directions`` retains first-seen label order; ``segments`` counts plot runs.
    """

    scan_id: Hashable
    requested: float | None
    actual: float
    minimum: float
    maximum: float
    directions: tuple[Hashable, ...]
    segments: int


class DataSplitter:
    """Detect and split multiple field sweeps in tabular datasets.

    Typical usage:
        splitter = DataSplitter()
        segments = splitter.detect_sweep_segments(df, field_col="B", min_points=10)
        split_dfs = splitter.split_dataframe_by_segments(df, segments)
    """

    @staticmethod
    def select_scan(
        df: pd.DataFrame,
        parameter: Hashable,
        target: float | None = None,
        *,
        scan_col: Hashable = "scan_id",
        scan_id: Hashable | None = None,
        direction_col: Hashable | None = "direction",
        time_col: Hashable | None = None,
        max_gap: float | None = None,
        segment_col: Hashable = "_plot_segment",
        time_unit: Literal["D", "h", "m", "s", "ms", "us", "ns"] | None = None,
    ) -> tuple[pd.DataFrame, ScanMetadata]:
        """Select an actual acquired scan, without interpolation or relabeling direction.

        Select by ID or the nearest scan-mean parameter (ties and out-of-range
        requests raise). Return (copied rows, metadata). Plot segments split at
        direction changes and optional time gaps; input order/index are retained.
        Parameter min/max describe drift along the selected physical path.

        ``max_gap`` is in seconds. For a numeric time column, pass its unit
        explicitly, e.g. ``time_unit="s"`` for elapsed seconds. The default
        preserves pandas datetime parsing, including nanoseconds for numeric
        values; no unit is inferred from column names or magnitudes. A supplied
        time_unit requires a numeric column. Metadata remains a plain dict.
        """
        if (target is None) == (scan_id is None):
            raise ValueError("provide either target or scan_id")
        if segment_col in df:
            raise ValueError(f"{segment_col} already exists")
        if df.empty or df[scan_col].isna().any():
            raise ValueError("scan data must be nonempty and scan IDs must not be missing")
        values = df[parameter].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("scan parameter must be finite")
        means = df[parameter].groupby(df[scan_col], sort=False, observed=True).mean()
        if target is not None:
            if not np.isfinite(target) or not means.min() <= target <= means.max():
                raise ValueError("target is outside the measured scan-mean range")
            distances = (means - target).abs()
            matches = distances.index[np.isclose(distances, distances.min(), rtol=1e-12, atol=0)]
            if len(matches) != 1:
                raise ValueError("multiple scans are equally close; select scan_id explicitly")
            scan_id = matches[0]
        elif scan_id not in means.index:
            raise ValueError("scan_id is not present")
        selected = df.loc[df[scan_col] == scan_id].copy()
        breaks = np.zeros(len(selected), dtype=bool)
        directions: tuple[Hashable, ...] = ()
        if direction_col is not None:
            if selected[direction_col].isna().any():
                raise ValueError("direction labels must not be missing")
            # shift() creates a missing first comparison for nullable dtypes.
            # Compare only subsequent rows; the first row starts a segment below.
            changes = selected[direction_col].ne(selected[direction_col].shift())
            breaks[1:] = changes.iloc[1:].to_numpy(dtype=bool)
            directions = tuple(selected[direction_col].unique())
        if max_gap is not None:
            if time_col is None or not np.isfinite(max_gap) or max_gap <= 0:
                raise ValueError(
                    "max_gap requires time_col and a positive finite seconds threshold"
                )
            if time_unit is not None and not pd.api.types.is_numeric_dtype(selected[time_col]):
                raise ValueError("time_unit requires a numeric time column")
            times = pd.to_datetime(selected[time_col], unit=time_unit, errors="raise")
            if times.isna().any():
                raise ValueError("timestamps must not be missing")
            gaps = times.diff().dt.total_seconds().to_numpy()
            if np.any(gaps[1:] < 0):
                raise ValueError("timestamps must follow acquisition order")
            breaks |= gaps > max_gap
        breaks[0] = True
        selected[segment_col] = np.cumsum(breaks) - 1
        return selected, dict(scan_id=scan_id, requested=target,
                              actual=float(means.at[scan_id]),
                              minimum=float(selected[parameter].min()),
                              maximum=float(selected[parameter].max()),
                              directions=directions, segments=int(breaks.sum()))

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
        valid_positions = np.flatnonzero(mask)
        work = work.loc[mask].reset_index(drop=True)

        n_rows = len(work)
        if n_rows == 0:
            return []

        def original_positions(segments):
            # Keep positions relative to the original frame passed to split().
            # Invalid rows remain in the corresponding segment, not silently lost.
            return [SweepSegment(
                0 if seg.start_index == 0 else int(valid_positions[seg.start_index]),
                len(df) if seg.end_index == n_rows else int(valid_positions[seg.end_index]),
                seg.direction,
            ) for seg in segments]

        if n_rows < max(2 * min_points, 5):
            # Not enough rows to form multiple segments; single segment only
            logger.debug(
                "Too few rows (%d) for multi-sweep detection; returning one segment",
                n_rows,
            )
            direction = self._overall_direction(work[field_col].to_numpy())
            return original_positions([SweepSegment(0, n_rows, direction)])

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
        return original_positions(segments)

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
            segments: Segment definitions in original dataframe row positions.
                Invalid coordinate rows ignored during detection remain here.
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


