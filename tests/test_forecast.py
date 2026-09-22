"""Offline tests: the grid and DST handling, no-lookahead features, the rolling
backtest, metrics, and conformal intervals on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxiforecast import backtest as bt, data, features

ZONES = pd.DataFrame({"LocationID": [1, 2, 3], "Borough": ["Manhattan", "Bronx", "Staten Island"],
                      "Zone": ["a", "b", "c"], "service_zone": ["x"] * 3})


def synthetic_grid(start="2024-01-01", days=70, seed=0, level=1.0) -> pd.DataFrame:
    """Four boroughs with a daily and weekly cycle plus Poisson noise."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, periods=days * 24, freq="h")
    hours = hours[hours != data.DST_GAP]
    scale = {"Manhattan": 3000, "Queens": 400, "Brooklyn": 60, "Bronx": 12}
    rows = []
    for borough, base in scale.items():
        shape = 0.3 + np.sin(np.pi * hours.hour / 24) ** 2 * (1 - 0.3 * (hours.dayofweek >= 5))
        rows.append(pd.DataFrame({"hour": hours, "borough": borough,
                                  "trips": rng.poisson(level * base * shape)}))
    grid = pd.concat(rows, ignore_index=True)
    grid["dst_irregular"] = grid["hour"] == data.DST_DOUBLE
    return grid


# ---- data -----------------------------------------------------------------------

def test_hourly_counts_filters_boroughs_and_bad_durations():
    t = pd.Timestamp("2024-02-01 08:15")
    trips = pd.DataFrame({
        "tpep_pickup_datetime": [t, t, t, t, t],
        "tpep_dropoff_datetime": [t + pd.Timedelta(minutes=m) for m in (10, 10, 10, 0, 300)],
        "PULocationID": [1, 1, 3, 1, 1],
    })
    counts = data.hourly_counts(trips, ZONES)
    assert counts.to_dict("records") == [{"hour": pd.Timestamp("2024-02-01 08:00"), "borough": "Manhattan", "trips": 2}]


def test_grid_fills_zeros_and_skips_the_dst_gap():
    counts = pd.DataFrame({"hour": [pd.Timestamp("2024-03-10 01:00")], "borough": ["Bronx"], "trips": [3]})
    grid = data.complete_grid(counts, "2024-03-10", "2024-03-11")
    assert len(grid) == 23 * 4                                   # 10 March had 23 hours
    assert data.DST_GAP not in set(grid["hour"])
    assert grid["trips"].sum() == 3
    assert (grid[grid["borough"] == "Manhattan"]["trips"] == 0).all()


def test_dst_double_hour_is_flagged():
    grid = data.complete_grid(pd.DataFrame(columns=["hour", "borough", "trips"]), "2024-11-03", "2024-11-04")
    assert grid.loc[grid["dst_irregular"], "hour"].unique().tolist() == [data.DST_DOUBLE]


# ---- features ---------------------------------------------------------------------

def test_lags_are_looked_up_by_timestamp_across_the_dst_gap():
    frame = features.build(synthetic_grid(start="2024-03-01", days=20)).set_index(["borough", "hour"])
    grid = synthetic_grid(start="2024-03-01", days=20).set_index(["borough", "hour"])["trips"]
    # 11 Mar 03:00 looks back to 10 Mar 03:00, not to the row 24 positions earlier
    assert frame.loc[("Bronx", pd.Timestamp("2024-03-11 03:00")), "lag_24"] == grid[("Bronx", pd.Timestamp("2024-03-10 03:00"))]
    # 11 Mar 02:00 looks back to an hour that never existed
    assert np.isnan(frame.loc[("Bronx", pd.Timestamp("2024-03-11 02:00")), "lag_24"])


def test_features_never_see_the_forecast_day():
    """Change every count from day D onwards; no feature of any hour on day D may move."""
    grid = synthetic_grid(days=60)
    day = pd.Timestamp("2024-02-20")
    before = features.build(grid)
    changed = grid.copy()
    changed.loc[changed["hour"] >= day, "trips"] *= 7
    after = features.build(changed)
    on_day = (before["hour"] >= day) & (before["hour"] < day + pd.Timedelta(days=1))
    cols = [c for c in features.NUMERIC]
    pd.testing.assert_frame_equal(before.loc[on_day, cols].reset_index(drop=True),
                                  after.loc[on_day, cols].reset_index(drop=True))


def test_level_uses_the_seven_previous_days():
    grid = synthetic_grid(days=20)
    frame = features.build(grid)
    row = frame[(frame["borough"] == "Queens") & (frame["hour"] == pd.Timestamp("2024-01-15 10:00"))].iloc[0]
    window = grid[(grid["borough"] == "Queens") & (grid["hour"] >= "2024-01-08") & (grid["hour"] < "2024-01-15")]
    assert row["level_7d"] == pytest.approx(window["trips"].mean())


def test_holiday_flags():
    frame = features.build(synthetic_grid(start="2024-01-01", days=3))
    day = frame.drop_duplicates("hour").set_index("hour")
    assert day.loc["2024-01-01 12:00", "is_holiday"] == 1
    assert day.loc["2024-01-02 12:00", "day_after_holiday"] == 1
    assert day.loc["2024-01-03 12:00", ["is_holiday", "day_after_holiday"]].sum() == 0


def test_usable_needs_four_weeks_of_history():
    frame = features.build(synthetic_grid(days=40))
    first = frame.loc[features.usable(frame), "hour"].min()
    assert first == pd.Timestamp("2024-01-29")


# ---- backtest ---------------------------------------------------------------------

def test_rolling_origin_trains_only_on_the_past(monkeypatch):
    frame = features.build(synthetic_grid(days=70))
    seen = []
    real_fit = bt.fit

    def spy(train, variant, seed=0):
        seen.append(train["hour"].max())
        return real_fit(train, variant, seed)

    monkeypatch.setattr(bt, "fit", spy)
    period = (pd.Timestamp("2024-02-26"), pd.Timestamp("2024-03-11"))
    out = bt.run(frame, bt.VARIANTS[0], period)
    folds = bt.weeks(period)
    assert len(seen) == len(folds) == 2
    for latest, start in zip(seen, folds):
        assert latest < start
    assert out["hour"].min() == period[0] and out["hour"].max() < period[1]


def test_model_beats_last_week_on_a_learnable_series():
    frame = features.build(synthetic_grid(days=84))
    period = (pd.Timestamp("2024-03-11"), pd.Timestamp("2024-03-25"))
    model = bt.metrics(bt.run(frame, bt.VARIANTS[0], period)).set_index("borough").loc["ALL", "wape_pct"]
    naive = bt.metrics(bt.baselines(frame, period)["same hour last week"]).set_index("borough").loc["ALL", "wape_pct"]
    assert model < naive


def test_metrics_arithmetic():
    f = pd.DataFrame({"borough": ["A", "A", "B"], "trips": [10, 20, 100], "forecast": [12, 18, 90]})
    ref = f.assign(forecast=[10, 30, 80])
    m = bt.metrics(f, ref).set_index("borough")
    assert m.loc["A", "wape_pct"] == pytest.approx(100 * 4 / 30)
    assert m.loc["ALL", "wape_pct"] == pytest.approx(100 * 14 / 130)
    assert m.loc["A", "skill_vs_reference"] == pytest.approx(1 - 2 / 5)


# ---- conformal --------------------------------------------------------------------

def _forecasts(days=90, seed=0, night_noise=0.3, day_noise=0.1):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    noise = np.where(hours.hour < 6, night_noise, day_noise)
    forecast = np.full(len(hours), 500.0)
    actual = np.expm1(np.log1p(forecast) + rng.normal(0, noise))
    return pd.DataFrame({"hour": hours, "borough": "Queens", "trips": actual, "forecast": forecast})


def test_conformal_first_window_has_no_interval_and_later_covers():
    iv = bt.conformal(_forecasts(night_noise=0.1))
    first_days = iv["hour"] < pd.Timestamp("2024-01-29")
    assert iv.loc[first_days, "halfwidth"].isna().all()
    assert iv.loc[~first_days, "covered"].mean() == pytest.approx(0.90, abs=0.02)


def test_conformal_uses_only_past_days():
    f = _forecasts()
    base = bt.conformal(f)
    later = f.copy()
    later.loc[later["hour"] >= "2024-03-01", "trips"] *= 5           # wreck the future
    changed = bt.conformal(later)
    today = (base["hour"] >= "2024-02-29") & (base["hour"] < "2024-03-01")
    assert np.allclose(base.loc[today, "halfwidth"], changed.loc[today, "halfwidth"])


def test_mondrian_fixes_time_of_day_coverage():
    f = _forecasts(night_noise=0.3, day_noise=0.1)
    for by_block, night_ok in ((False, False), (True, True)):
        iv = bt.conformal(f, by_hour_block=by_block).dropna(subset=["halfwidth"])
        night = iv[iv["hour"].dt.hour < 6]["covered"].mean()
        assert (abs(night - 0.90) < 0.03) == night_ok


def test_checksum_mismatch(tmp_path):
    path = tmp_path / "bad.parquet"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.verify(path, next(iter(data.FILES.values())))
