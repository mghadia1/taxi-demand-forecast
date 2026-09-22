"""Command line: `taxiforecast download | run`.

`run` keeps the order that makes the evaluation honest: compare variants on
the selection period, pick one by a rule fixed in advance (lowest overall
WAPE), and only then forecast the evaluation period.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from taxiforecast import backtest as bt, charts, data, features

REFERENCE = "same hour last week"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="taxiforecast")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="fetch and checksum the 2024 TLC files (~700 MB)")
    dl.add_argument("--destination", type=Path, default=Path("data/raw"))
    rn = sub.add_parser("run", help="select on Jul-Sep, evaluate on Oct-Dec, add intervals; write outputs/")
    rn.add_argument("--raw", type=Path, default=Path("data/raw"))
    rn.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    if args.command == "download":
        for path in data.download(args.destination):
            print(path)
    else:
        print(json.dumps(run(args.raw, args.out), indent=2, default=float))


def run(raw: Path, out: Path) -> dict:
    for name, expected in data.FILES.items():
        data.verify(raw / name, expected)
    out.mkdir(parents=True, exist_ok=True)

    parquet = sorted(raw / n for n in data.FILES if n.endswith(".parquet"))
    counts = data.hourly_counts(parquet, raw / "taxi_zone_lookup.csv")
    frame = features.build(data.complete_grid(counts, "2024-01-01", "2025-01-01"))

    # 1. selection period: baselines and model variants
    sel_base = bt.baselines(frame, bt.SELECTION)
    sel_ref = sel_base[REFERENCE]
    selection_rows, sel_forecasts = [], {}
    for name, forecast in sel_base.items():
        selection_rows.append({"forecast": name, **_overall(bt.metrics(forecast, sel_ref))})
    for variant in bt.VARIANTS:
        sel_forecasts[variant.name] = bt.run(frame, variant, bt.SELECTION)
        selection_rows.append({"forecast": variant.name, **_overall(bt.metrics(sel_forecasts[variant.name], sel_ref))})
    selection = pd.DataFrame(selection_rows).sort_values("wape_pct")
    chosen = min(bt.VARIANTS, key=lambda v: bt.metrics(sel_forecasts[v.name]).set_index("borough").loc["ALL", "wape_pct"])

    # 2. evaluation period: chosen model vs the same baselines, never used for a choice
    eval_base = bt.baselines(frame, bt.EVALUATION)
    eval_ref = eval_base[REFERENCE]
    evaluated = bt.run(frame, chosen, bt.EVALUATION)
    evaluation = {name: bt.metrics(f, eval_ref) for name, f in eval_base.items()}
    evaluation[chosen.name] = bt.metrics(evaluated, eval_ref)

    # 3. intervals: conformal windows warm up on selection-period errors
    both = pd.concat([sel_forecasts[chosen.name], evaluated], ignore_index=True)

    def eval_intervals(by_hour_block: bool) -> pd.DataFrame:
        iv = bt.conformal(both, by_hour_block=by_hour_block)
        iv = iv[(iv["hour"] >= bt.EVALUATION[0]) & (iv["hour"] < bt.EVALUATION[1])].copy()
        iv["hour_block"] = bt.hour_block(iv["hour"])
        return iv

    in_eval = eval_intervals(by_hour_block=True)            # the method used
    pooled = eval_intervals(by_hour_block=False)             # the first version, kept for comparison
    cover_borough = bt.coverage_table(in_eval, "borough")
    cover_block = bt.coverage_table(in_eval, "hour_block")
    cover_block_pooled = bt.coverage_table(pooled, "hour_block")

    # 4. where the model fails: holidays vs ordinary days (evaluation period)
    evaluated["day"] = evaluated["hour"].dt.normalize()
    evaluated["abs_error"] = (evaluated["forecast"] - evaluated["trips"]).abs()
    daily = (evaluated.groupby("day")
             .agg(trips=("trips", "sum"), forecast=("forecast", "sum"), abs_error=("abs_error", "sum"),
                  is_holiday=("is_holiday", "max"), day_after=("day_after_holiday", "max"))
             .reset_index())
    daily["wape_pct"] = 100 * daily["abs_error"] / daily["trips"]
    daily["kind"] = np.where(daily["is_holiday"] == 1, "holiday",
                             np.where(daily["day_after"] == 1, "day after holiday", "ordinary"))
    by_kind = daily.groupby("kind").agg(days=("day", "size"), wape_pct=("wape_pct", "mean")).reset_index()
    worst_days = daily.sort_values("wape_pct", ascending=False).head(8)

    selection.to_csv(out / "selection.csv", index=False)
    pd.concat([t.assign(forecast=name) for name, t in evaluation.items()]).to_csv(out / "evaluation.csv", index=False)
    cover_borough.to_csv(out / "coverage_by_borough.csv", index=False)
    cover_block.to_csv(out / "coverage_by_hour_block.csv", index=False)
    daily.to_csv(out / "evaluation_daily.csv", index=False)
    in_eval.drop(columns=["dst_irregular"]).to_csv(out / "evaluation_forecasts.csv", index=False)

    charts.skill_by_borough(evaluation, chosen.name, REFERENCE, out / "skill_by_borough.png")
    charts.week_with_intervals(in_eval, "Manhattan", pd.Timestamp("2024-11-25"), out / "thanksgiving_week_manhattan.png")
    charts.week_with_intervals(in_eval, "Bronx", pd.Timestamp("2024-11-25"), out / "thanksgiving_week_bronx.png")
    charts.coverage_by_block(cover_block_pooled, cover_block, bt.COVERAGE, out / "coverage_by_hour_block.png")
    cover_block_pooled.to_csv(out / "coverage_by_hour_block_pooled.csv", index=False)

    ev = {name: t.set_index("borough") for name, t in evaluation.items()}
    scored = in_eval.dropna(subset=["halfwidth"])
    summary = {
        "hours_in_grid": int(len(frame) // len(data.BOROUGHS)),
        "trips_counted": int(frame["trips"].sum()),
        "selection_period": [str(d.date()) for d in bt.SELECTION],
        "evaluation_period": [str(d.date()) for d in bt.EVALUATION],
        "selection": selection.round(4).to_dict("records"),
        "chosen_variant": chosen.name,
        "evaluation_overall": {name: {k: round(float(v), 4) for k, v in t.loc["ALL"].items() if k != "hours"}
                               for name, t in ev.items()},
        "evaluation_by_borough": ev[chosen.name].round(4).reset_index().to_dict("records"),
        "interval_target": bt.COVERAGE,
        "interval_coverage_overall": round(float(scored["covered"].mean()), 4),
        "interval_coverage_by_borough": cover_borough.round(4).to_dict("records"),
        "interval_coverage_by_hour_block": cover_block.round(4).to_dict("records"),
        "pooled_interval_coverage_overall": round(float(pooled.dropna(subset=["halfwidth"])["covered"].mean()), 4),
        "pooled_interval_coverage_by_hour_block": cover_block_pooled.round(4).to_dict("records"),
        "daily_wape_by_kind": by_kind.round(3).to_dict("records"),
        "worst_days": worst_days[["day", "trips", "forecast", "wape_pct", "kind"]]
        .assign(day=lambda f: f["day"].dt.date.astype(str)).round(1).to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def _overall(table: pd.DataFrame) -> dict:
    row = table.set_index("borough").loc["ALL"]
    return {"wape_pct": float(row["wape_pct"]), "mae": float(row["mae"]),
            "skill_vs_last_week": float(row.get("skill_vs_reference", np.nan))}


if __name__ == "__main__":
    main()
