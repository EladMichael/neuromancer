"""
    Clock for simulation, to ensure that *all* components are using the same  
    clock, and that all time calculations and references are consistent.

    Time is measured in float64 seconds with t=0 referring to Unix Epoch

    Currently there is no yearly variations, so weather signals are repeated
    from year to year (if simulations are run multi-year), but clock will
    still measure time across years correctly.
"""
from datetime import datetime
from torch import vmap
import os
if os.environ.get("RUNTIME_TYPING", 1):
    from beartype import beartype
else:
    # passthrough for no type checking
    def beartype(fn): return fn
    

@beartype
def day_of_year(
    t_sec: int | float,
) -> int:
    t_date = datetime.fromtimestamp(t_sec)
    day_zero = datetime(t_date.year, 1, 1, 0, 0, 0)
    return 1 + (t_date - day_zero).days


days_of_year = vmap(day_of_year)


@beartype
def hour_of_day(
    t_sec: int | float,
) -> int:
    return datetime.fromtimestamp(t_sec).hour


hours_of_day = vmap(hour_of_day)


@beartype
def hour_str(
    t_sec: int | float,
    am_pm: bool = True,
) -> str:
    h_of_day = hour_of_day(t_sec)

    if am_pm:
        is_am = h_of_day < 12
        h_12 = ((h_of_day-1) % 12) + 1

        m_str = " AM" if is_am else " PM"
        return str(h_12) + m_str
    else:
        return str(h_of_day)


@beartype
def sec_from_date(
    *,
    year: int = 1970,
    month: int = 1,
    day: int = 1,
    hour: int = 10,
    minute: int = 0,
    second: int = 0
) -> float:
    return datetime(year, month, day, hour, minute, second).timestamp()
