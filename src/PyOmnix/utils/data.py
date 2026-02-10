from __future__ import annotations

import copy
from collections.abc import Sequence
from itertools import groupby
from typing import Any, Literal, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    import pandas as pd

from ..omnix_logger import get_logger

logger = get_logger(__name__)


class ObjectArray:
    """
    This class is used to store the objects as a multi-dimensional array.

    The ObjectArray provides a way to store objects in a multi-dimensional structure
    with methods for accessing, manipulating, and extending the array.
    NOTE: getter supports flattened indexing, but setter does not

    Attributes:
        shape (tuple[int, ...]): The dimensions of the array
        fill_value (Any): The default value used to fill the array
        unique (bool): If True, ensures all elements in the array are unique
    """

    def __init__(self, *dims: int, fill_value: Any = None, unique: bool = False) -> None:
        """
        initialize the ObjectArray with certain dimensions and fill value (note to copy the fill value except for special cases)

        Args:
            *dims: The dimensions of the objects
            fill_value: The value to fill the array with
            unique: If True, ensures all elements in the array are unique
        """
        self.shape = dims
        self.size = np.prod(dims)
        self.fill_value = fill_value
        self.unique = unique
        self.objects = self._create_objects(dims) # customized initialization can be implemented by overriding this method

    def __repr__(self) -> str:
        """
        Return a string representation of the ObjectArray.

        This method is called when the instance is directly referenced in a print statement
        or when it's evaluated in an interactive shell.

        Returns:
            str: A formatted string representation of all elements in the ObjectArray.
        """
        flat_objects = self._flatten(self.objects)

        # Create a formatted representation
        result = f"ObjectArray(shape={self.shape})\n"

        # Add elements with their indices
        for i, obj in enumerate(flat_objects):
            # Calculate multi-dimensional indices
            indices = []
            remaining = i
            for dim in reversed(self.shape):
                indices.insert(0, remaining % dim)
                remaining //= dim

            # Format the element representation
            obj_repr = str(obj).replace("\n", "\n  ")  # Indent any multi-line representations
            result += f"  {tuple(indices)}: {obj_repr}\n"

        return result

    def _create_objects(self, dims: tuple[int, ...]) -> list[Any]:
        """
        create the list of objects
        override this method to customize the initialization
        (for details, see the MRO of Python)

        Args:
        - dims: the dimensions of the objects
        """
        if len(dims) == 1:
            return [copy.deepcopy(self.fill_value) for _ in range(dims[0])]
        else:
            return [self._create_objects(dims[1:]) for _ in range(dims[0])]

    def _flatten(self, lst):
        """
        Flatten a multi-dimensional list using recursion
        """
        return [
            item
            for sublist in lst
            for item in (self._flatten(sublist) if isinstance(sublist, list) else [sublist])
        ]

    def __getitem__(self, index: tuple[int, ...] | int) -> dict:
        """
        get the objects assignated by the index

        Args:
        - index: the index of the object to be get
        """
        if isinstance(index, int):
            index = np.unravel_index(index, self.shape)
        arr = self.objects
        for idx in index:
            arr = arr[idx]
        return arr

    def __setitem__(self, index: tuple[int, ...] | int, value: Any) -> None:
        """
        Set the value at the specified index.

        Args:
            index: The index where to set the value
            value: The value to set

        Raises:
            ValueError: If unique is True and the value already exists in the array
        """
        if isinstance(index, int):
            if index == -1:
                logger.validate(self.pointer_next is not None, "No space to set value")
                self.__setitem__(self.pointer_next, value)
                return
            index = np.unravel_index(index, self.shape)
        arr = self.objects
        for idx in index[:-1]:
            arr = arr[idx]
        arr[index[-1]] = copy.deepcopy(value)

    def _are_equal(self, obj1: Any, obj2: Any) -> bool:
        """
        Compare two objects for equality using appropriate method based on type.

        Args:
            obj1: First object to compare
            obj2: Second object to compare

        Returns:
            bool: True if objects are considered equal, False otherwise
        """
        # Handle None values
        if obj1 is None and obj2 is None:
            return True
        if obj1 is None or obj2 is None:
            return False

        # Handle numpy arrays
        if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
            return np.array_equal(obj1, obj2)

        # Handle pandas objects
        obj1_type = type(obj1).__name__
        obj2_type = type(obj2).__name__
        if "DataFrame" in obj1_type and "DataFrame" in obj2_type:
            if hasattr(obj1, "equals"):
                return obj1.equals(obj2)
        if "Series" in obj1_type and "Series" in obj2_type:
            if hasattr(obj1, "equals"):
                return obj1.equals(obj2)

        # For other objects, try equality comparison
        try:
            return obj1 == obj2
        except Exception as e:
            logger.warning("Equality comparison failed: %s. Using identity comparison.", e)
            return obj1 is obj2

    def _validate_uniqueness(self, value: Any, current_index: tuple[int, ...]) -> bool:
        """
        Validate that the new value maintains uniqueness in the array.

        Args:
            value: The value to check for uniqueness
            current_index: The index where the value would be inserted

        Returns:
            bool: True if the value is unique (or not required uniqueness), False otherwise
        """
        # if not required uniqueness, return True
        if not self.unique:
            return True

        locations = self.find(value)
        # Filter out the current index from locations if it exists
        other_locations = [loc for loc in locations if loc != current_index]
        
        if other_locations:
            return False
        else:
            return True

    @property
    def pointer_next(self) -> int | None:
        """
        return the index of the next element that is not filled values by default
        """
        for i in range(self.size):
            if self._are_equal(self[i], self.fill_value):
                return i
        return None

    def extend(self, *dims: int) -> None:
        """
        Extend the array to a new shape, filling extended elements with None. Not the dimensions of the array should be the same.

        This method extends the array to the specified dimensions while preserving
        the existing elements. If any of the new dimensions is smaller than the
        current dimensions, an error is raised.

        Args:
            *dims: The new dimensions for the array
        """
        # Check if new dimensions are valid (not smaller than current)
        logger.validate(
            len(dims) == len(self.shape),
            f"Expected {len(self.shape)} dimensions, got {len(dims)}",
        )

        for i, (current, new) in enumerate(zip(self.shape, dims, strict=False)):
            logger.validate(
                new >= current,
                f"New dimension {i} ({new}) is smaller than current dimension ({current})",
            )

        # If dimensions are the same, no need to extend
        if dims == self.shape:
            return

        # Create a new array with the extended dimensions
        new_objects = self._create_objects(dims)

        # Copy existing elements to the new array
        self._copy_elements(self.objects, new_objects, self.shape)

        # Update shape and objects
        self.shape = dims
        self.objects = new_objects

    def clear(self) -> None:
        """
        Clear the array
        """
        self.objects = self._create_objects(self.shape)

    def _copy_elements(
        self,
        source: list,
        target: list,
        source_shape: tuple[int, ...],
        source_idx: tuple = (),
        target_idx: tuple = (),
    ) -> None:
        """
        Recursively copy elements from source array to target array.

        Args:
            source: Source array to copy from
            target: Target array to copy to
            source_shape: Shape of the source array
            source_idx: Current index in source array (for recursion)
            target_idx: Current index in target array (for recursion)
        """
        if len(source_idx) == len(source_shape):
            # We've reached the elements, copy the value
            self._set_subarray(target, target_idx, self._get_subarray(source, source_idx))
            return

        # Get current dimension
        dim_idx = len(source_idx)

        # Recursively copy elements for this dimension
        for i in range(source_shape[dim_idx]):
            self._copy_elements(source, target, source_shape, source_idx + (i,), target_idx + (i,))

    def find(self, search_value: Any) -> list[tuple[int, ...]]:
        """
        Find all locations of a given object in the array. Only supports one object at a time.

        Args:
            search_value: The object to search for in the array.

        Returns:
            list[tuple[int, ...]]: A list of tuples containing the indices where the value was found.
                                 Each tuple represents the multi-dimensional index location.
        """
        flat_objects = self._flatten(self.objects)
        found_indices = []

        # Find all matching indices in flattened array
        for i, obj in enumerate(flat_objects):
            if self._are_equal(obj, search_value):
                # Calculate multi-dimensional indices
                indices = []
                remaining = i
                for dim in reversed(self.shape):
                    indices.insert(0, remaining % dim)
                    remaining //= dim
                found_indices.append(tuple(indices))

        return found_indices

    def find_objs(self, search_values: Sequence[Any] | Any) -> list[tuple[int, ...]]:
        """
        Find locations of given objects in the tuple or list(if multiple locations are found, only the first one will be returned). Supports multiple objects at a time.
        """
        if not isinstance(search_values, (tuple, list)):
            search_values = (search_values,)
        flat_objects = self._flatten(self.objects)
        found_indices = []
        for search_value in search_values:
            for i, obj in enumerate(flat_objects):
                if self._are_equal(obj, search_value):
                    # Calculate multi-dimensional indices
                    indices = []
                    remaining = i
                    for dim in reversed(self.shape):
                        indices.insert(0, remaining % dim)
                        remaining //= dim
                    found_indices.append(tuple(indices))
                    break
        return found_indices


class CacheArray:
    """
    A class working as dynamic cache with max length and if-stable status
    """

    def __init__(self, cache_length: int = 60, *, var_crit: float = 1e-4, least_length: int = 3):
        """
        Args:
            cache_length: the max length of the cache
            var_crit: the criterion of the variance
            least_length: the least length of the cache to judge the stability(smaller cache will be considered unstable)
        """
        self.cache_length = cache_length
        self.cache = np.array([])
        self.var_crit = var_crit
        self.least_length = least_length

    @property
    def mean(self) -> float | None:
        """return the mean of the cache"""
        if self.cache.size == 0:
            logger.warning("Cache is empty")
            return None
        else:
            return self.cache.mean()

    def update_cache(
        self, new_value: float | Sequence[float]
    ) -> None:
        """
        update the cache using newest values
        """
        if isinstance(new_value, (int, float)):
            new_value = [new_value]

        self.cache = np.append(self.cache, new_value)[-self.cache_length :]

    def get_status(
        self, *, require_cache: bool = False, var_crit: float | None = None
    ) -> dict[str, float | Sequence[float] | bool] | None:
        """
        return the cache, mean value, and whether the cache is stable

        Args:
            require_cache (bool): whether to return the cache array
            var_crit (float): the criterion of the variance
        """
        if self.cache.size <= self.least_length:
            logger.debug("Cache is not enough to judge the stability")
            var_stable = False
        else:
            if var_crit is None:
                var_stable = self.cache.var() < self.var_crit
            else:
                var_stable = self.cache.var() < var_crit

        if require_cache:
            return {"cache": self.cache, "mean": self.mean, "if_stable": var_stable}
        return {"mean": self.mean, "if_stable": var_stable}


def rename_duplicates(columns: list[str]) -> list[str]:
    """
    rename the duplicates with numbers (like ["V","V"] to ["V1","V2"])
    """
    count_dict = {}
    renamed_columns = []
    for col in columns:
        if col in count_dict:
            count_dict[col] += 1
            renamed_columns.append(f"{col}{count_dict[col]}")
        else:
            count_dict[col] = 1
            renamed_columns.append(col)
    return renamed_columns


def match_with_tolerance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    target_axis: Any,
    tolerance: float,
    suffixes: tuple[str] = ("_1", "_2"),
) -> pd.DataFrame:
    """
    Merge two dataframes according to the target_axis and only keep the rows within tolerance, unmatched rows will be dropped. Suffixes will be added to distinguish the columns from different dataframes.
    e.g.
    | A | B | and |  A  | C |   =>      | A_1 | B_1 | A_2 | C_2 |
    |---|---|     |-----|---| tole=0.2  |-----|-----|-----|-----|
    | 1 | 2 |     | 1.1 | 2 | axis="A"  |  1  |  2  | 1.1 |  2  |
    | 3 | 4 |     | 3.2 | 4 |           |  3  |  4  | 3.2 |  4  |
    | 5 | 6 |     | 5.3 | 6 |           (row 5 is dropped)


    Args:
    - df1: the first dataframe
    - df2: the second dataframe
    - on: the column to merge on
    - tolerance: the tolerance for the merge
    - suffixes: the suffixes for the columns of the two dataframes
    """
    import pandas as pd
    df1 = df1.sort_values(by=target_axis).reset_index(drop=True)
    df2 = df2.sort_values(by=target_axis).reset_index(drop=True)

    i = 0
    j = 0

    result = []

    while i < len(df1) and j < len(df2):
        if abs(df1.loc[i, target_axis] - df2.loc[j, target_axis]) <= tolerance:
            row = pd.concat(
                [df1.loc[i].add_suffix(suffixes[0]), df2.loc[j].add_suffix(suffixes[1])]
            )
            result.append(row)
            i += 1
            j += 1
        elif df1.loc[i, target_axis] < df2.loc[j, target_axis]:
            i += 1
        else:
            j += 1

    return pd.DataFrame(result).copy()


def symmetrize(
    ori_df: pd.DataFrame,
    index_col: str | float | int,
    obj_col: str | float | int | list[str | float | int],
    *,
    neutral_point: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    do symmetrization to the dataframe w.r.t. the index col and return the symmetric and antisymmetric DataFrames,
    note that this function is dealing with only one dataframe, meaning the positive and negative parts
    are to be combined first (no need to sort)
    e.g. idx col is [-1,-2,-3,0,4,2,1], obj cols corresponding to -1 will add/minus the corresponding obj cols of 1
        that of -3 will be added/minus that interpolated by 2 and 4, etc. (positive - negative)/2 for antisym

    Args:
    - ori_df: the original dataframe
    - index_col: the name of the index column for symmetrization
    - obj_col: a list of the name(s) of the objective column for symmetrization
    - neutral_point: the neutral point for symmetrization

    Returns:
    - pd.DataFrame[0]: the symmetric part (col names are suffixed with "_sym")
    - pd.DataFrame[1]: the antisymmetric part (col names are suffixed with "_antisym")
    """
    import pandas as pd
    if not isinstance(obj_col, (tuple, list)):
        obj_col = [obj_col]
    # Separate the negative and positive parts for interpolation
    df_negative = ori_df[ori_df[index_col] < neutral_point][[index_col] + obj_col].copy()
    df_positive = ori_df[ori_df[index_col] > neutral_point][[index_col] + obj_col].copy()
    # For symmetrization, we need to flip the negative part and make positions positive
    df_negative[index_col] = -df_negative[index_col]
    # sort them
    df_negative = df_negative.sort_values(by=index_col).reset_index(drop=True)
    df_positive = df_positive.sort_values(by=index_col).reset_index(drop=True)
    # do interpolation for the union of the two parts
    index_union = np.union1d(df_negative[index_col], df_positive[index_col])
    pos_interpolated = np.array(
        [
            np.interp(index_union, df_positive[index_col], df_positive[obj_col[i]])
            for i in range(len(obj_col))
        ]
    )
    neg_interpolated = np.array(
        [
            np.interp(index_union, df_negative[index_col], df_negative[obj_col[i]])
            for i in range(len(obj_col))
        ]
    )
    # Symmetrize and save to DataFrame
    sym = (pos_interpolated + neg_interpolated) / 2
    sym_df = pd.DataFrame(
        np.transpose(np.append([index_union], sym, axis=0)),
        columns=[index_col] + [f"{obj_col[i]}_sym" for i in range(len(obj_col))],
    )
    antisym = (pos_interpolated - neg_interpolated) / 2
    antisym_df = pd.DataFrame(
        np.transpose(np.append([index_union], antisym, axis=0)),
        columns=[index_col] + [f"{obj_col[i]}_antisym" for i in range(len(obj_col))],
    )

    # return pd.concat([sym_df, antisym_df], axis = 1)
    return sym_df, antisym_df

def extract_longest_monotonic_segment(
    df: pd.DataFrame, col: str = "I_source"
) -> pd.DataFrame:
    """Extract the longest monotonically increasing or decreasing segment from a DataFrame.

    The function identifies contiguous segments where the values in ``col``
    move in one direction (increasing or decreasing, with flat regions
    treated as continuing the previous trend) and returns the longest one.

    Args:
        df: The input DataFrame.
        col: The column name whose monotonicity is evaluated.

    Returns:
        A copy of the longest monotonic segment (original columns preserved).
    """
    # 1. Compute the sign of the first-order difference.
    #    np.sign(0) returns 0; replace 0 with NaN then forward-fill so that
    #    flat regions inherit the direction of the preceding trend.
    #    Back-fill handles the leading NaN (from diff) so the first row is
    #    grouped with the opening trend instead of being isolated.
    diff_sign = np.sign(df[col].diff()).replace(0, np.nan).ffill().bfill()

    # 2. Detect direction-change points.
    #    A change occurs when the current sign differs from the previous one.
    direction_changes = diff_sign != diff_sign.shift()

    # 3. Assign a segment ID to every row.
    #    Each direction reversal increments the cumulative sum by 1,
    #    thereby separating distinct monotonic intervals.
    segment_id = direction_changes.cumsum()

    # 4. Find the segment ID with the most rows (i.e. the longest segment).
    longest_segment_id = segment_id.value_counts().idxmax()

    # 5. Extract and return the longest segment.
    return df[segment_id == longest_segment_id].copy()


def difference(
    ori_df: Sequence[pd.DataFrame],
    index_col: str | float | int | Sequence[str | float | int],
    target_col: str
    | float
    | int
    | Sequence[str | float | int]
    | Sequence[Sequence[str | float | int]],
    *,
    relative: bool = False,
    interpolate_method: str = "linear",
) -> pd.DataFrame:
    """
    Calculate the difference between the values in the columns(should have the same name) of two dataframes
    the final df will use the names of the first df
    NOTE the interpolation will cause severe error for extension outside the original range
    the overlapped values will be AVERAGED
    e.g. ori_df = [df1, df2], index_col = ["B1", "B2"] (if given "B", it equals to ["B", "B"]), target_col = [["I1", "I2"], ["I3", "I4"]] (same as above, low-dimension will be expanded to high-dimension), the result will be df["B1"] = df1["B1"] - df2["B2"], df["I1"] = df1["I1"] - df2["I3"], df["I2"] = df1["I2"] - df2["I4"]

    Args:
    - ori_df: the original dataframe(s)
    - index_col: the name of the index column for symmetrization
    - target_col: the name of the target column for difference calculation
    - relative: whether to calculate the relative difference
    - interpolate_method: the method for interpolation, default is "linear"
    """
    import pandas as pd
    logger.validate(len(ori_df) == 2, "ori_df should be a sequence of two elements")
    if isinstance(index_col, (str, float, int)):
        return difference(
            ori_df,
            [index_col, index_col],
            target_col,
            relative=relative,
            interpolate_method=interpolate_method,
        )
    logger.validate(len(index_col) == 2, "index_col should be a sequence of two elements")
    if isinstance(target_col, (str, float, int)):
        return difference(
            ori_df,
            index_col,
            [[target_col], [target_col]],
            relative=relative,
            interpolate_method=interpolate_method,
        )
    elif isinstance(target_col[0], (str, float, int)):
        return difference(
            ori_df,
            index_col,
            [target_col, target_col],
            relative=relative,
            interpolate_method=interpolate_method,
        )
    logger.validate(
        len(target_col) == 2 and len(target_col[0]) == len(target_col[1]),
        "target_col should be a sequence of two equally long sequences",
    )

    rename_dict = {index_col[1]: index_col[0]}
    for i in range(len(target_col[0])):
        rename_dict[target_col[1][i]] = target_col[0][i]
    df_1 = ori_df[0][[index_col[0]] + target_col[0]].copy()
    df_2 = ori_df[1][[index_col[1]] + target_col[1]].copy()
    df_1.set_index(index_col[0], inplace=True)
    df_2.set_index(index_col[1], inplace=True)
    df_2.rename(columns=rename_dict, inplace=True)

    common_idx = sorted(set(df_1.index).union(set(df_2.index)))
    df_1_reindexed = (
        df_1.groupby(df_1.index)
        .mean()
        .reindex(common_idx)
        .interpolate(method=interpolate_method)
        .sort_index()
    )
    df_2_reindexed = (
        df_2.groupby(df_2.index)
        .mean()
        .reindex(common_idx)
        .interpolate(method=interpolate_method)
        .sort_index()
    )
    diff = df_1_reindexed - df_2_reindexed
    if relative:
        diff = diff / df_2_reindexed
    diff[index_col[0]] = diff.index
    diff.reset_index(drop=True, inplace=True)
    return diff


def loop_diff(
    ori_df: pd.DataFrame,
    vary_col: str | float | int,
    target_col: str | float | int | list[str | float | int],
    *,
    relative: bool = False,
    interpolate_method: str = "linear",
) -> pd.DataFrame:
    """
    Calculate the difference within a hysteresis loop (increasing minus decreasing direction)

    Args:
    - ori_df: the original dataframe
    - vary_col: the name of the column to vary
    - target_col: the name of the column to calculate the difference
    - relative: whether to calculate the relative difference
    - interpolate_method: the method for interpolation, default is "linear"
    """
    if not isinstance(target_col, (tuple, list)):
        target_col = [target_col]
    df_1 = ori_df[[vary_col] + target_col].copy()
    df_1 = identify_direction(df_1, vary_col)
    return difference(
        [df_1[df_1["direction"] == 1], df_1[df_1["direction"] == -1]],
        vary_col,
        target_col,
        relative=relative,
        interpolate_method=interpolate_method,
    )


def identify_direction(
    ori_df: pd.DataFrame, idx_col: str | float | int, min_count: int = 17
):
    """
    Identify the direction of the sweeping column and add another direction column
    (1 for increasing, -1 for decreasing)

    Args:
    - ori_df: the original dataframe
    - idx_col: the name of the index column
    - min_count: the min number of points for each direction (used to avoid fluctuation at ends)
    """
    df_in = ori_df.copy()
    df_in["direction"] = np.sign(np.gradient(df_in[idx_col]))
    directions = df_in["direction"].tolist()
    # Perform run-length encoding
    rle = [(direction, len(list(group))) for direction, group in groupby(directions)]
    # Initialize filtered directions list
    filtered_directions = []
    for idx, (direction, length) in enumerate(rle):
        if length >= min_count and direction != 0:
            # Accept the run as is
            filtered_directions.extend([direction] * length)
        else:
            # Replace short runs with the previous direction
            if filtered_directions:
                replaced_direction = filtered_directions[-1]
            else:
                lookahead_idx = idx + 1
                while lookahead_idx < len(rle) and (
                    rle[lookahead_idx][1] < min_count or rle[lookahead_idx][0] == 0
                ):
                    lookahead_idx += 1
                assert lookahead_idx < len(rle), "The direction for starting is not clear"
                replaced_direction = rle[lookahead_idx][0]
            filtered_directions.extend([replaced_direction] * length)

    # Assign the filtered directions back to the DataFrame
    df_in["direction"] = filtered_directions
    return df_in

def sph_to_cart(r: float, theta: float, phi: float) -> tuple[float, float, float]:
    """
    Convert spherical coordinates to Cartesian coordinates
    r: radius
    theta: polar angle
    phi: azimuthal angle
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def scatter_to_candle(
    scatter_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    interval_mode: Literal["count", "val"] = "val",
    interval: int | float | str | pd.Timedelta | None = None,
    num_bins: int | None = None,
    closed: Literal["left", "right"] = "left",
) -> pd.DataFrame:
    """
    Convert scatter data to candlestick OHLC data.

    Returns a DataFrame with columns: ``time``, ``open``, ``high``, ``low``, ``close``.

    Behavior:
    - interval_mode == "count": group sequentially by fixed point counts per candle.
      ``interval`` should be an int (points per candle). If None, an automatic size
      is chosen to target ~100 candles.
    - interval_mode == "val": group by value ranges of ``x_col`` (numeric) or by
      time windows (datetime-like). For time windows, set ``interval`` to a pandas
      offset alias (e.g. "1T", "5min") or a ``pd.Timedelta``. If ``interval`` is
      None, the range is split into ``num_bins`` (default ~100) equal bins.

    Handles numeric and timestamp x-values. Rows that cannot be parsed (NaN/NaT)
    are dropped.
    """
    import pandas as pd
    logger.validate(
        x_col in scatter_df.columns and y_col in scatter_df.columns,
        "x_col and y_col must exist in DataFrame",
    )

    df = scatter_df[[x_col, y_col]].copy()
    df = df.dropna(subset=[x_col, y_col])
    if df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"]).copy()

    def _agg_group(g: pd.DataFrame, x_name: str, y_name: str) -> dict[str, Any]:
        g_sorted = g.sort_values(by=x_name)
        return {
            "time": g_sorted[x_name].iloc[0],
            "open": g_sorted[y_name].iloc[0],
            "high": g_sorted[y_name].max(),
            "low": g_sorted[y_name].min(),
            "close": g_sorted[y_name].iloc[-1],
        }

    if interval_mode == "count":
        # Group sequentially by point count (after sorting by x)
        df_sorted = df.sort_values(by=x_col).reset_index(drop=True)
        n = len(df_sorted)
        if isinstance(interval, int) and interval > 0:
            window_size = interval
        else:
            target_candles = num_bins if (isinstance(num_bins, int) and num_bins > 0) else 100
            window_size = max(1, int(np.ceil(n / target_candles)))

        group_ids = (np.arange(n) // window_size).astype(int)
        grouped = df_sorted.groupby(group_ids, dropna=False)
        records = [_agg_group(g, x_col, y_col) for _, g in grouped if not g.empty]
        result = pd.DataFrame.from_records(records)  # type: ignore
        result = result.sort_values(by="time").reset_index(drop=True)
        result.rename(columns={"time": "time"}, inplace=True)
        return result[["time", "open", "high", "low", "close"]]

    # interval_mode == "val": determine whether x is datetime-like or numeric
    is_datetime = pd.api.types.is_datetime64_any_dtype(df[x_col])
    x_dt = None
    if not is_datetime:
        # Try coercing to datetime to detect timestamp-like values
        x_try = pd.to_datetime(df[x_col], errors="coerce", utc=False, infer_datetime_format=True)
        if not x_try.isna().all():
            is_datetime = True
            x_dt = x_try
    else:
        x_dt = df[x_col]

    if is_datetime:
        # Time-based grouping
        if x_dt is None:
            x_dt = pd.to_datetime(df[x_col], errors="coerce", utc=False)
        valid_mask = ~x_dt.isna()
        df_t = df.loc[valid_mask].copy()
        df_t["__x"] = x_dt.loc[valid_mask].astype("datetime64[ns]")
        if df_t.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close"]).copy()

        df_t = df_t.sort_values(by="__x")
        if interval is not None:
            # Use resample on a datetime index
            freq = interval
            s = df_t.set_index("__x")[y_col]
            ohlc = s.resample(freq, origin="start_day").ohlc()
            ohlc.index.name = "time"
            ohlc = ohlc.dropna(how="all")
            out = ohlc.reset_index()
            return out.rename(
                columns={"open": "open", "high": "high", "low": "low", "close": "close"}
            )[["time", "open", "high", "low", "close"]]
        else:
            # Bin the overall time span into equal bins
            target_bins = num_bins if (isinstance(num_bins, int) and num_bins > 0) else 100
            start = df_t["__x"].min()
            end = df_t["__x"].max()
            if start == end:
                # Single timestamp: one candle
                rec = _agg_group(df_t, "__x", y_col)
                rec["time"] = start
                return pd.DataFrame([rec], columns=["time", "open", "high", "low", "close"]).copy()
            bins = pd.date_range(start=start, end=end, periods=target_bins + 1)
            cats = pd.cut(
                df_t["__x"],
                bins=bins,
                right=(closed == "right"),
                include_lowest=True,
            )
            df_t["__bin"] = cats
            records = []
            for _, g in df_t.groupby("__bin", dropna=True):
                if g.empty:
                    continue
                rec = _agg_group(g, "__x", y_col)
                # Label time by bin left or right edge
                interval_obj = g["__bin"].iloc[0]
                rec["time"] = interval_obj.left if closed == "left" else interval_obj.right
                records.append(rec)
            out = pd.DataFrame.from_records(records)
            return out.sort_values(by="time").reset_index(drop=True)[
                ["time", "open", "high", "low", "close"]
            ]

    # Numeric value-based grouping
    x_num = pd.to_numeric(df[x_col], errors="coerce")
    df_n = df.loc[~x_num.isna()].copy()
    if df_n.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"]).copy()
    df_n["__x"] = x_num.loc[df_n.index]
    df_n = df_n.sort_values(by="__x")

    if interval is not None:
        logger.validate(
            isinstance(interval, (int, float)) and interval > 0,
            "For interval_mode='val' with numeric x, interval must be a positive number",
        )
        width = float(interval)
        x_min = df_n["__x"].min()
        x_max = df_n["__x"].max()
        if x_min == x_max:
            rec = _agg_group(df_n, "__x", y_col)
            rec["time"] = x_min
            return pd.DataFrame([rec], columns=["time", "open", "high", "low", "close"]).copy()
        start_edge = np.floor(x_min / width) * width
        end_edge = np.ceil(x_max / width) * width
        # Ensure inclusive end by adding one step
        edges = np.arange(start_edge, end_edge + width, width)
        cats = pd.cut(
            df_n["__x"],
            bins=edges,
            right=(closed == "right"),
            include_lowest=True,
        )
        df_n["__bin"] = cats
        records = []
        for _, g in df_n.groupby("__bin", dropna=True):
            if g.empty:
                continue
            rec = _agg_group(g, "__x", y_col)
            interval_obj = g["__bin"].iloc[0]
            rec["time"] = float(interval_obj.left if closed == "left" else interval_obj.right)
            records.append(rec)
        out = pd.DataFrame.from_records(records)
        return out.sort_values(by="time").reset_index(drop=True)[
            ["time", "open", "high", "low", "close"]
        ]
    else:
        target_bins = num_bins if (isinstance(num_bins, int) and num_bins > 0) else 100
        x_min = df_n["__x"].min()
        x_max = df_n["__x"].max()
        if x_min == x_max:
            rec = _agg_group(df_n, "__x", y_col)
            rec["time"] = x_min
            return pd.DataFrame([rec], columns=["time", "open", "high", "low", "close"]).copy()
        edges = np.linspace(x_min, x_max, target_bins + 1)
        cats = pd.cut(
            df_n["__x"],
            bins=edges,
            right=(closed == "right"),
            include_lowest=True,
        )
        df_n["__bin"] = cats
        records = []
        for _, g in df_n.groupby("__bin", dropna=True):
            if g.empty:
                continue
            rec = _agg_group(g, "__x", y_col)
            interval_obj = g["__bin"].iloc[0]
            rec["time"] = float(interval_obj.left if closed == "left" else interval_obj.right)
            records.append(rec)
        out = pd.DataFrame.from_records(records)
        return out.sort_values(by="time").reset_index(drop=True)[
            ["time", "open", "high", "low", "close"]
        ]

if __name__ == "__main__":
    from pyomnix.utils.data import scatter_to_candle
    import pandas as pd
    import numpy as np
    test_df = pd.DataFrame({"x": np.linspace(0, 10, 100), "y": np.sin(np.linspace(0, 10, 100))})
    scatter_to_candle(test_df, x_col="x", y_col="y", interval_mode="val", interval=None)