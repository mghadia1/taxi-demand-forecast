"""Features for a next-day forecast.

The forecast is made at midnight for all 24 hours of the coming day. Every
feature of a target hour on day D must therefore be computable from data
before D 00:00. Lags of 24 hours or more satisfy that for every hour of the
day; a 1-hour lag would not (at midnight, 17:00 of the same day is unknown).

Lags are looked up **by timestamp**, not by shifting rows. The grid is missing
the DST spring-forward hour, so a row shift of 24 would drift by one hour
for the rest of the year; a timestamp lookup returns NaN for that one hour
instead, which the model handles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# US federal holidays in 2024: calendar facts, known in advance.
HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19", "2024-07-04",
    "2024-09-02", "2024-10-14", "2024-11-11", "2024-11-28", "2024-12-25",
])

LAG_HOURS = (24, 48, 168, 336, 504, 672)
NUMERIC = [
    "hour_of_day", "day_of_week", "is_holiday", "day_after_holiday",
    "lag_24", "lag_48", "lag_168", "lag_336", "lag_504", "lag_672",
    "same_hour_mean_4w", "level_7d",
]
CATEGORICAL = ["borough"]
FEATURES = NUMERIC + CATEGORICAL


def build(grid: pd.DataFrame) -> pd.DataFrame:
    frame = grid.sort_values(["borough", "hour"]).reset_index(drop=True)
    series = frame.set_index(["borough", "hour"])["trips"].astype(float)
    # the DST double hour is not a real one-hour count: never use it as a lag
    series[frame.set_index(["borough", "hour"])["dst_irregular"].to_numpy()] = np.nan

    for k in LAG_HOURS:
        keys = pd.MultiIndex.from_arrays([frame["borough"], frame["hour"] - pd.Timedelta(hours=k)])
        frame[f"lag_{k}"] = series.reindex(keys).to_numpy()
    frame["same_hour_mean_4w"] = frame[["lag_168", "lag_336", "lag_504", "lag_672"]].mean(axis=1, skipna=True)

    # mean hourly trips over the 7 full days before the forecast day: the level
    # the day starts from, which tracks slow changes (summer dip, autumn rise)
    frame["day"] = frame["hour"].dt.normalize()
    daily = frame.groupby(["borough", "day"])["trips"].mean()
    level = daily.groupby(level="borough").transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    frame["level_7d"] = level.reindex(pd.MultiIndex.from_arrays([frame["borough"], frame["day"]])).to_numpy()

    frame["hour_of_day"] = frame["hour"].dt.hour
    frame["day_of_week"] = frame["hour"].dt.dayofweek
    frame["is_holiday"] = frame["day"].isin(HOLIDAYS).astype(int)
    frame["day_after_holiday"] = (frame["day"] - pd.Timedelta(days=1)).isin(HOLIDAYS).astype(int)
    return frame


def usable(frame: pd.DataFrame) -> pd.Series:
    """Rows fit to train on or score: all four weekly lags present and a real hour."""
    return frame[["lag_672", "level_7d"]].notna().all(axis=1) & ~frame["dst_irregular"]
