"""Event-preserving statistics and explicit single-branch alignment."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)


def _finite_column(df: pd.DataFrame, column: str) -> np.ndarray:
    values = df[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        logger.raise_error(f"{column} contains nonfinite values", ValueError, include_traceback=False)
    return values


def label_plateaus(df: pd.DataFrame, column: str, *, atol: float = 0,
                   event_col: str = "pulse_id") -> pd.DataFrame:
    """Label consecutive plateaus, without merging later repeats of a value.

    Values are compared to each plateau's first value (not chained neighbors).
    Equal consecutive pulses cannot be recovered from amplitude alone; use
    acquisition event IDs when available. Existing IDs are never overwritten.
    """
    if not np.isfinite(atol) or atol < 0:
        logger.raise_error("atol must be finite and nonnegative", ValueError, include_traceback=False)
    if event_col in df:
        logger.raise_error(f"{event_col} already exists", ValueError, include_traceback=False)
    values = _finite_column(df, column)
    ids = np.zeros(len(values), dtype=int)
    if len(values):
        anchor, event = values[0], 0
        for i, value in enumerate(values):
            if abs(value - anchor) > atol:
                anchor, event = value, event + 1
            ids[i] = event
    result = df.copy()
    result[event_col] = ids
    return result


def summarize_events(df: pd.DataFrame, *, value_col: str = "I_read",
                     group_cols: Sequence[str] = ("cycle_id", "pulse_id"),
                     metadata_cols: Sequence[str] = ("pulse_value", "read_bias"),
                     tail: int | None = None,
                     time_window: tuple[float, float] | None = None,
                     time_col: str = "time_since_pulse_s") -> pd.DataFrame:
    """Return mean, sample std (ddof=1), and count per event in acquisition order.

    Select either the last N rows or an inclusive time window, or all rows.
    Short tail windows and empty time windows raise rather than silently using
    less data. A single sample has undefined std (NaN). Metadata must be
    constant across the entire event. Resistance is deliberately not inferred:
    read_bias / I_read_mean differs from mean(read_bias / I_read).
    """
    groups, metadata = list(group_cols), list(metadata_cols)
    if not groups or len(set(groups + metadata)) != len(groups + metadata):
        logger.raise_error("group and metadata columns must be unique and disjoint", ValueError, include_traceback=False)
    outputs = [f"{value_col}_mean", f"{value_col}_std", "n_samples"]
    if set(outputs) & set(groups + metadata):
        logger.raise_error("output names collide with group/metadata columns", ValueError, include_traceback=False)
    if tail is not None and (not isinstance(tail, int) or tail < 1):
        logger.raise_error("tail must be a positive integer", ValueError, include_traceback=False)
    if tail is not None and time_window is not None:
        logger.raise_error("choose tail or time_window, not both", ValueError, include_traceback=False)
    if time_window is not None:
        if len(time_window) != 2 or not np.isfinite(time_window).all() or time_window[0] > time_window[1]:
            logger.raise_error("time_window must be finite and ordered", ValueError, include_traceback=False)
        _finite_column(df, time_col)
    _finite_column(df, value_col)
    if df[groups + metadata].isna().any().any():
        logger.raise_error("event IDs and metadata must not be missing", ValueError, include_traceback=False)
    records = []
    for key, event in df.groupby(groups, sort=False, observed=True):
        keys = key if isinstance(key, tuple) else (key,)
        record = dict(zip(groups, keys, strict=True))
        for column in metadata:
            if event[column].nunique(dropna=False) != 1:
                logger.raise_error(f"{column} varies within event {keys}", ValueError, include_traceback=False)
            record[column] = event[column].iloc[0]
        selected = event
        if tail is not None:
            if len(event) < tail:
                logger.raise_error(f"event {keys} contains fewer than {tail} samples", ValueError, include_traceback=False)
            selected = event.iloc[-tail:]
        elif time_window is not None:
            selected = event[event[time_col].between(*time_window)]
        if selected.empty:
            logger.raise_error(f"no samples in selected window for event {keys}", ValueError, include_traceback=False)
        values = pd.Series(_finite_column(selected, value_col))
        record.update({outputs[0]: values.mean(), outputs[1]: values.std(ddof=1),
                       outputs[2]: len(values)})
        records.append(record)
    return pd.DataFrame(records, columns=groups + metadata + outputs)


def branch_difference(left: pd.DataFrame, right: pd.DataFrame, *, x: str,
                      y: str, method: str = "exact", tolerance: float = 0) -> pd.DataFrame:
    """Subtract two explicitly selected, strictly monotonic branches.

    Preserve the left branch's order and columns; append '<y>_reference' and
    '<y>_difference'. exact requires identical coordinate sets; nearest requires
    an unambiguous one-to-one match for every left point within tolerance.
    linear uses actual coordinate distances and rejects extrapolation. Duplicate
    x values raise: callers must choose their endpoint/repeat policy explicitly.
    Never pass a whole multi-branch loop to this function.
    """
    if method not in {"exact", "nearest", "linear"}:
        logger.raise_error("method must be exact, nearest or linear", ValueError, include_traceback=False)
    if not np.isfinite(tolerance) or tolerance < 0:
        logger.raise_error("tolerance must be finite and nonnegative", ValueError, include_traceback=False)
    if method != "nearest" and tolerance != 0:
        logger.raise_error("tolerance applies only to nearest matching", ValueError, include_traceback=False)
    for frame in (left, right):
        coords = _finite_column(frame, x)
        _finite_column(frame, y)
        if not len(coords):
            logger.raise_error("branches must not be empty", ValueError, include_traceback=False)
        if len(coords) > 1 and not (np.all(np.diff(coords) > 0) or np.all(np.diff(coords) < 0)):
            logger.raise_error("select a strictly monotonic branch with unique coordinates", ValueError, include_traceback=False)
    names = (f"{y}_reference", f"{y}_difference")
    if any(name in left for name in names):
        logger.raise_error("difference output columns already exist", ValueError, include_traceback=False)
    lx = _finite_column(left, x)
    reference = right.sort_values(x)
    rx, ry = _finite_column(reference, x), _finite_column(reference, y)
    if method == "exact":
        if not np.array_equal(np.sort(lx), rx):
            logger.raise_error("exact matching requires identical coordinate sets", ValueError, include_traceback=False)
        matched = ry[np.searchsorted(rx, lx)]
    elif method == "linear":
        if len(rx) < 2 or lx.min() < rx[0] or lx.max() > rx[-1]:
            logger.raise_error("linear matching requires two reference points and no extrapolation", ValueError, include_traceback=False)
        matched = np.interp(lx, rx, ry)
    else:
        used, indices = set(), []
        for value in lx:
            distances = np.abs(rx - value)
            candidates = np.flatnonzero(distances == distances.min())
            if len(candidates) != 1 or distances[candidates[0]] > tolerance:
                logger.raise_error("missing or ambiguous nearest coordinate", ValueError, include_traceback=False)
            index = int(candidates[0])
            if index in used:
                logger.raise_error("nearest matching would reuse a reference point", ValueError, include_traceback=False)
            used.add(index)
            indices.append(index)
        matched = ry[indices]
    result = left.copy()
    result[names[0]] = matched
    result[names[1]] = _finite_column(left, y) - matched
    return result
