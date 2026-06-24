"""
    Clock for simulation, to ensure that *all* components are using the same
    clock, and that all time calculations and references are consistent.

    Time is measured in float64 seconds with t=0 referring to the Unix Epoch.
    All conversions use UTC so results are reproducible across machines and
    timezones (the wall-clock timezone of the host must not change behaviour).

    Currently there is no yearly variation, so weather signals repeat from year
    to year (if simulations are run multi-year), but the clock still measures
    time across years correctly.
"""
from datetime import datetime, timezone

from ._runtime import beartype


@beartype
def day_of_year(
    t_sec: int | float,
) -> int:
    t_date = datetime.fromtimestamp(t_sec, tz=timezone.utc)
    day_zero = datetime(t_date.year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    return 1 + (t_date - day_zero).days


@beartype
def hour_of_day(
    t_sec: int | float,
) -> float:
    """Hour of day in [0, 24) as a float (includes minutes/seconds) for smooth
    diurnal signals and well-behaved derivatives."""
    t_date = datetime.fromtimestamp(t_sec, tz=timezone.utc)
    return t_date.hour + t_date.minute / 60.0 + t_date.second / 3600.0


@beartype
def hour_str(
    t_sec: int | float,
    am_pm: bool = True,
) -> str:
    h_of_day = int(hour_of_day(t_sec))

    if am_pm:
        is_am = h_of_day < 12
        h_12 = ((h_of_day - 1) % 12) + 1

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
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp()
