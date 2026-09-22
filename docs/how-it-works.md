# How it works

```
12 monthly 2024 parquet files (SHA-256) ──DuckDB──▶ hourly pickups × 4 boroughs
        ▼
complete grid: real zeros filled, DST gap hour dropped, DST double hour flagged
        ▼
features (all from data before the forecast day's midnight):
  lags 24/48/168/336/504/672 h (by timestamp), 4-week same-hour mean,
  7-day level, hour, weekday, holiday, day after holiday, borough
        ▼
selection period Jul–Sep: 13 weekly folds × {Poisson, log-squared} + 3 baselines ─▶ pick lowest WAPE
        ▼
evaluation period Oct–Dec: 13 weekly folds, chosen model, same baselines
        ▼
online conformal intervals (per borough × 6-hour block, 28-day window) ─▶ coverage checks
```

## Choices I would defend

| Choice | Why |
|---|---|
| **Forecast at midnight for the whole next day** | It is a realistic planning horizon (staffing tomorrow), and it fixes which data is allowed: anything before 00:00 |
| **Lags of 24 hours or more only** | At midnight, 17:00 of the same day is unknown. A 1-hour lag would be leakage |
| **Look up lags by timestamp, not by row position** | The grid is missing the DST gap hour. A row shift of 24 would be off by an hour for the rest of the year |
| **Rolling origin, weekly retraining** | Each week's model sees only the past, like it would in production. A random split of hours would train on 14:00 and test on 13:00 of the same day |
| **Selection period and evaluation period** | The same role as validation and test sets, laid out in time |
| **Baselines first** | "Same hour last week" is a strong forecast for strongly weekly data. A model that can't beat it has no reason to exist |
| **One global model across boroughs** | The boroughs share daily and weekly shapes, so the model learns those from all of them. Borough is a categorical feature |
| **Poisson loss** | The targets are counts. It was also chosen by the rule, on the selection period |
| **WAPE as the headline metric** | It compares fairly across boroughs whose volumes differ 300-fold. MAPE would explode on Bronx hours with 0–2 trips |
| **Conformal intervals on the log scale, per borough × time block** | Log scale: one relative width works for 5 trips and for 8,000. Time blocks: errors are larger at night, relative to volume |

## Tests (15)

The tests cover:
- **SQL aggregation:** the borough and duration filters.
- **The grid:** zero-filling, the missing spring-forward hour (a 23-hour day), and the flagged fall-back hour.
- **Lag lookup across the DST gap:** by timestamp, with NaN for the hour that never existed.
- **No lookahead:** multiply every count from day D onwards by 7, and no feature of any hour on day D changes.
- **Other features:** the 7-day level is the mean of exactly the previous 7 days; holiday flags; four weeks of history are required.
- **The rolling origin:** a spy on `fit` confirms every training set ends before its fold starts.
- **The model beats "last week"** on a learnable synthetic series.
- **Metrics:** WAPE and skill arithmetic.
- **Conformal intervals:** no interval before a full window; about 90% coverage on exchangeable errors; wrecking future data does not change today's interval; per-block widths fix night coverage where pooled widths do not.
- **Checksum rejection.**

## Things to be able to explain

- Why can't the model use a 1-hour lag?
- What is a rolling-origin backtest, and why not a random split?
- Why does "same hour yesterday" lose to "same hour last week"?
- Why was the model's advantage larger in Oct–Dec than in Jul–Sep?
- What does 88.8% coverage mean, and why is it a little under 90%?
- How was the time-of-day fix found, and why are both versions reported?
- How would you improve holiday forecasts?
