"""Rolling-origin backtest, baselines, and online conformal intervals.

**Rolling origin.** For every week W in the backtest, the model is trained
only on hours before W's Monday 00:00, then forecasts each day of W. Nothing
from W or later can reach training. A random train/test split of hours would
let the model train on 14:00 and be tested on 13:00 of the same day.

**Two periods.** Model variants are compared on the *selection* period
(July-September 2024). The chosen variant is then reported on the
*evaluation* period (October-December 2024), which played no part in any
choice. Same idea as a validation and a test set, laid out in time.

**Intervals.** Split conformal, run online: for each forecast day, take the
out-of-sample errors (on a log scale) of the previous 28 days, and use their
90% quantile as the half-width. Only errors from days already over at the
forecast origin are used, so the interval is as honest as the forecast.

The errors are pooled per borough *and* per 6-hour block of the day
("Mondrian" conformal). One width per borough gave about 90% coverage on
average but only ~80% overnight and ~97% at midday: right on average and
wrong at every time of day. Both versions are reported; see docs/results.md
for how this was found.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from taxiforecast.features import CATEGORICAL, FEATURES, NUMERIC, usable

SELECTION = (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30"))    # 13 weeks
EVALUATION = (pd.Timestamp("2024-09-30"), pd.Timestamp("2024-12-30"))   # 13 weeks
COVERAGE = 0.90
CONFORMAL_DAYS = 28


@dataclass(frozen=True)
class Variant:
    name: str
    loss: str                 # "poisson" on counts, or "squared_error" on log1p(counts)


VARIANTS = (Variant("boosting, Poisson loss", "poisson"), Variant("boosting, squared loss on log counts", "squared_error"))


def _matrix(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[FEATURES].copy()
    X["borough"] = X["borough"].astype("category")
    return X


def fit(train: pd.DataFrame, variant: Variant, seed: int = 0) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss=variant.loss, learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        categorical_features="from_dtype", random_state=seed,
    )
    y = train["trips"] if variant.loss == "poisson" else np.log1p(train["trips"])
    model.fit(_matrix(train), y)
    return model


def predict(model, frame: pd.DataFrame, variant: Variant) -> np.ndarray:
    raw = model.predict(_matrix(frame))
    return np.clip(raw if variant.loss == "poisson" else np.expm1(raw), 0, None)


def weeks(period: tuple[pd.Timestamp, pd.Timestamp]) -> list[pd.Timestamp]:
    return list(pd.date_range(period[0], period[1], freq="7D", inclusive="left"))


def run(frame: pd.DataFrame, variant: Variant, period: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """Forecast every usable hour in `period`, retraining at each week start."""
    ok = usable(frame)
    out = []
    for start in weeks(period):
        end = start + pd.Timedelta(days=7)
        train = frame[ok & (frame["hour"] < start)]
        target = frame[ok & (frame["hour"] >= start) & (frame["hour"] < end)].copy()
        model = fit(train, variant)
        target["forecast"] = predict(model, target, variant)
        target["fold"] = start
        out.append(target)
    return pd.concat(out, ignore_index=True)


def baselines(frame: pd.DataFrame, period: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, pd.DataFrame]:
    ok = usable(frame) & (frame["hour"] >= period[0]) & (frame["hour"] < period[1])
    part = frame[ok]
    return {
        "same hour yesterday": part.assign(forecast=part["lag_24"]),
        "same hour last week": part.assign(forecast=part["lag_168"]),
        "same hour, mean of last 4 weeks": part.assign(forecast=part["same_hour_mean_4w"]),
    }


def metrics(forecasts: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per borough and overall: MAE, WAPE, and skill against a reference forecast.

    WAPE = sum|error| / sum(actual), comparable across boroughs whose volumes
    differ 300-fold. Skill = 1 - MAE / MAE(reference): 0 means no better than
    the reference, and negative means worse.
    """
    f = forecasts.assign(abs_error=(forecasts["forecast"] - forecasts["trips"]).abs())
    rows = []
    for borough, part in list(f.groupby("borough")) + [("ALL", f)]:
        row = {"borough": borough, "hours": len(part), "mean_trips": part["trips"].mean(),
               "mae": part["abs_error"].mean(), "wape_pct": 100 * part["abs_error"].sum() / part["trips"].sum()}
        if reference is not None:
            ref = reference if borough == "ALL" else reference[reference["borough"] == borough]
            ref_mae = (ref["forecast"] - ref["trips"]).abs().mean()
            row["skill_vs_reference"] = 1 - row["mae"] / ref_mae
        rows.append(row)
    return pd.DataFrame(rows)


def hour_block(hours: pd.Series) -> pd.Series:
    return pd.cut(hours.dt.hour, [0, 6, 12, 18, 24], right=False,
                  labels=["00-05", "06-11", "12-17", "18-23"]).astype(str)


def conformal(forecasts: pd.DataFrame, coverage: float = COVERAGE, window_days: int = CONFORMAL_DAYS,
              by_hour_block: bool = False) -> pd.DataFrame:
    """Online split-conformal intervals on the log1p scale, per borough
    (and per 6-hour block of the day when `by_hour_block`).

    For a forecast of day D, the half-width is the `coverage` quantile of
    |log1p(actual) - log1p(forecast)| over that borough's previous
    `window_days` days of forecasts. The log scale makes one width serve both
    an hour of 5 trips and an hour of 8,000. Days without a full window of
    history get no interval (NaN).
    """
    f = forecasts.sort_values(["borough", "hour"]).copy()
    f["day"] = f["hour"].dt.normalize()
    f["log_error"] = (np.log1p(f["trips"]) - np.log1p(f["forecast"])).abs()
    f["cell"] = f["borough"] + ("|" + hour_block(f["hour"]) if by_hour_block else "")
    halfwidth = pd.Series(np.nan, index=f.index)
    for _, part in f.groupby("cell"):
        days = part["day"].unique()
        for day in days:
            history = part[(part["day"] < day) & (part["day"] >= day - pd.Timedelta(days=window_days))]
            if history["day"].nunique() < window_days:
                continue
            # finite-sample correction: the ceil((n+1) q)/n quantile guarantees
            # >= q coverage when errors are exchangeable
            n = len(history)
            level = min(1.0, np.ceil((n + 1) * coverage) / n)
            halfwidth[part.index[part["day"] == day]] = np.quantile(history["log_error"], level)
    f["halfwidth"] = halfwidth
    f["lower"] = np.clip(np.expm1(np.log1p(f["forecast"]) - f["halfwidth"]), 0, None)
    f["upper"] = np.expm1(np.log1p(f["forecast"]) + f["halfwidth"])
    f["covered"] = (f["trips"] >= f["lower"]) & (f["trips"] <= f["upper"])
    return f.drop(columns=["day", "cell"])


def coverage_table(intervals: pd.DataFrame, by: str) -> pd.DataFrame:
    scored = intervals.dropna(subset=["halfwidth"])
    return (scored.groupby(by)
            .agg(hours=("covered", "size"), coverage=("covered", "mean"),
                 median_width=("upper", lambda s: float(np.median(s - scored.loc[s.index, "lower"]))))
            .reset_index())
