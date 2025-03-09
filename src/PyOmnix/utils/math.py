import re
from datetime import datetime
from typing import Generator, Literal, Sequence, Optional
import numpy as np
import pandas as pd
from ..pltconfig import color_preset as colors


# define constants
CM_TO_INCH = 0.3937
HPLANCK = 6.626 * 10 ** (-34)  # SI unit: J*s
HBAR = HPLANCK / 2 / np.pi
HBAR_THZ = HBAR * 10 ** 12  # SI unit: J*s -> THz
KB = 1.38 * 10 ** (-23)  # SI unit: J/K
UNIT_FACTOR_FROMSI = {"": 1, "f": 1E15, "p": 1E12,
                      "n": 1E9, "u": 1E6, "m": 1E3,
                      "k": 1E-3, "M": 1E-6, "G": 1E-9,
                      "T": 1E-12, "P": 1E-15}
UNIT_FACTOR_TO_SI = {"": 1, "f": 1E-15, "p": 1E-12,
                     "n": 1E-9, "u": 1E-6, "m": 1E-3,
                     "k": 1E3, "M": 1E6, "G": 1E9,
                     "T": 1E12,"P": 1E15}

#define plotting default settings
DEFAULT_PLOT_DICT = {"color": colors.Presets["Nl"][0],
                     "linewidth": 1,
                     "linestyle": "-",
                     "marker": "o",
                     "markersize": 1.5,
                     "markerfacecolor": "None",
                     "markeredgecolor": "black",
                     "markeredgewidth": 0.3,
                     "label": "", "alpha": 0.77}

SWITCH_DICT = {"on": True, "off": False, "ON": True, "OFF": False}


def split_no_str(s: str | int | float) -> tuple[float | None, str | None]:
    """
    split the string into the string part and the float part.

    Args:
        s (str): the string to split

    Returns:
        tuple[float,str]: the string part and the integer part
    """
    if isinstance(s, (int, float)):
        return s, ""
    match = re.match(r"([+-]?[0-9.]+)([a-zA-Z]*)", s, re.I)

    if match:
        items = match.groups()
        return float(items[0]), items[1]
    else:
        return None, None

def factor(unit: str, mode: str = "from_SI"):
    """
    Transform the SI unit to targeted unit or in the reverse order.

    Args:
    unit: str
        The unit to be transformed.
    mode: str
        The direction of the transformation. "from_SI" means transforming from SI unit to the targeted unit, and "to_SI" means transforming from the targeted unit to SI unit.
    """
    # add judgement for the length to avoid m (meter) T (tesla) to be recognized as milli
    if len(unit) <= 1:
        return 1
    if mode == "from_SI":
        if unit[0] in UNIT_FACTOR_FROMSI:
            return UNIT_FACTOR_FROMSI.get(unit[0])
        else:
            return 1
    if mode == "to_SI":
        if unit[0] in UNIT_FACTOR_TO_SI:
            return UNIT_FACTOR_TO_SI.get(unit[0])
        else:
            return 1

def convert_unit(before: float | int | str | list[float | int | str, ...] | tuple[float | int | str, ...] | np.ndarray,
                 target_unit: str = "") -> tuple[float, str] | tuple[list[float], list[str]]:
    """
    Convert the value with the unit to the SI unit.

    Args:
        before (float | str): the value with the unit
        target_unit (str): the target unit

    Returns:
        tuple[float, str]: the value in the target unit and the whole str with final unit
    """
    if isinstance(before, (int, float, str)):
        value, unit = split_no_str(before)
        value_SI = value * factor(unit, mode="to_SI")
        new_value = value_SI * factor(target_unit, mode="from_SI")
        return new_value, f"{new_value}{target_unit}"
    elif isinstance(before, (np.integer, np.floating)):
        return convert_unit(float(before), target_unit)
    elif isinstance(before, (list, tuple, np.ndarray)):
        return [convert_unit(i, target_unit)[0] for i in before], [convert_unit(i, target_unit)[1] for i in before]

def gen_seq(start, end, step):
    """
    double-ended bi-direction sequence generator
    """
    if step == 0:
        raise ValueError("step should not be zero")
    if step * (end - start) < 0:
        step *= -1
    value = start
    while (value - end) * step < 0:
        yield value
        value += step
    yield end

def constant_generator(value, repeat: int | Literal["inf"] = "inf"):
    """
    generate a constant value infinitely
    """
    if repeat == "inf":
        while True:
            yield value
    else:
        idx = 0
        while idx < repeat:
            idx += 1
            yield value


def time_generator(format_str: str = "%Y-%m-%d_%H:%M:%S"):
    """
    generate current time always

    Args:
        format_str (str): the format of the time
    """
    while True:
        yield datetime.now().isoformat(sep="_", timespec="milliseconds")


def combined_generator_list(lst_gens: list[Generator]):
    """
    combine a list of generators into one generator generating a whole list
    """
    while True:
        try:
            list_ini = [next(i) for i in lst_gens]
            list_fin = []
            for i in list_ini:
                if isinstance(i, list | tuple):
                    list_fin.extend(i)
                else:
                    list_fin.append(i)
            yield list_fin
        except StopIteration:
            break


def next_lst_gen(lst_gens: list[Generator]):
    """
    get the next value of the generators in the list ONCE
    """
    try:
        list_ini = [next(i) for i in lst_gens]
        list_fin = []
        for i in list_ini:
            if isinstance(i, list | tuple):
                list_fin.extend(i)
            else:
                list_fin.append(i)
        return list_fin
    except StopIteration:
        return None

def timestr_convert(t: pd.Series | Sequence[str] | np.ndarray, format_str: str = "%Y-%m-%d_%H:%M:%S.%f", *,
                    elapsed: Optional[Literal["sec", "min", "hour"]] = None) -> list[datetime] | list[float]:
    """
    Convert the time to datetime object, used to split time series without day information

    Args:
    t : pd.Series
        The time series to be converted, format should be like "11:30 PM"
    format_str : str
        The format string for the time, e.g. "%I:%M %p"
        the meaning of each character and optional characters is as follows:
        %H : Hour (24-hour clock) as a zero-padded decimal number. 00, 01, ..., 23
        %I : Hour (12-hour clock) as a zero-padded decimal number. 01, 02, ..., 12
        %p : Locale's equivalent of either AM or PM.
        %M : Minute as a zero-padded decimal number. 00, 01, ..., 59
        %S : Second as a zero-padded decimal number. 00, 01, ..., 59
        %f : Microsecond as a decimal number, zero-padded on the left. 000000, 000001, ..., 999999
        %a : Weekday as locale's abbreviated name.
        %A : Weekday as locale's full name.
        %w : Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.
        %d : Day of the month as a zero-padded decimal number. 01, 02, ..., 31
        %b : Month as locale's abbreviated name.
        %B : Month as locale's full name.
        %m : Month as a zero-padded decimal number. 01, 02, ..., 12
        %y : Year without century as a zero-padded decimal number. 00, 01, ..., 99
        %Y : Year with century as a decimal number. 0001, 0002, ..., 2013, 2014, ..., 9998, 9999
    elapsed : Literal["sec", "min", "hour"]
        Whether to return the time past from first time points instead of return datetime list
    Returns:
    list[datetime] | list[float]
        The datetime list or the time past from the first time points
    """
    datetime_lst = [datetime.strptime(ts, format_str) for ts in t]
    if not datetime_lst:
        raise ValueError("The input time series is empty")
    if elapsed is not None:
        time_start = datetime_lst[0]
        match elapsed:
            case "sec":
                elapsed_times = [(dt - time_start).total_seconds() for dt in datetime_lst]
            case "min":
                elapsed_times = [(dt - time_start).total_seconds() / 60 for dt in datetime_lst]
            case "hour":
                elapsed_times = [(dt - time_start).total_seconds() / 3600 for dt in datetime_lst]
            case _:
                raise ValueError("The elapsed argument should be one of 'sec', 'min', 'hour'")
        return elapsed_times
    else:
        return datetime_lst
