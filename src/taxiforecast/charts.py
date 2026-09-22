"""Matplotlib figures for the README and results document."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#1f2937"
ACCENT = "#0e6a70"
MUTED = "#9ca3af"
WARN = "#b45309"


def _finish(fig, ax, path: Path) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def skill_by_borough(evaluation: dict[str, pd.DataFrame], chosen: str, reference: str, path: Path) -> None:
    names = [n for n in evaluation if n != reference]
    boroughs = ["Manhattan", "Queens", "Brooklyn", "Bronx", "ALL"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        table = evaluation[name].set_index("borough")
        values = [100 * table.loc[b, "skill_vs_reference"] for b in boroughs]
        color = ACCENT if name == chosen else MUTED if "mean" in name else "#d1d5db"
        ax.bar([x + i * width for x in range(len(boroughs))], values, width=width, color=color, label=name)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks([x + width * (len(names) - 1) / 2 for x in range(len(boroughs))], boroughs)
    ax.set_ylabel(f"% lower error than '{reference}'")
    ax.set_title("Evaluation period (Oct-Dec 2024): improvement over the weekly baseline", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _finish(fig, ax, path)


def week_with_intervals(forecasts: pd.DataFrame, borough: str, monday: pd.Timestamp, path: Path) -> None:
    part = forecasts[(forecasts["borough"] == borough) & (forecasts["hour"] >= monday)
                     & (forecasts["hour"] < monday + pd.Timedelta(days=7))].sort_values("hour")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.fill_between(part["hour"], part["lower"], part["upper"], color=ACCENT, alpha=0.18, linewidth=0,
                    label="90% interval")
    ax.plot(part["hour"], part["forecast"], color=ACCENT, linewidth=1.2, label="forecast (made at midnight)")
    ax.plot(part["hour"], part["trips"], color=INK, linewidth=1, label="actual")
    missed = part[~part["covered"]]
    ax.scatter(missed["hour"], missed["trips"], color=WARN, s=10, zorder=3, label="outside interval")
    ax.set_ylabel("Pickups per hour")
    ax.set_title(f"{borough}, week of {monday:%d %b %Y} (Thanksgiving is Thursday)", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    _finish(fig, ax, path)


def coverage_by_block(pooled: pd.DataFrame, per_block: pd.DataFrame, target: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = range(len(per_block))
    ax.bar([i - 0.2 for i in x], 100 * pooled["coverage"], width=0.4, color=MUTED, label="one width per borough")
    ax.bar([i + 0.2 for i in x], 100 * per_block["coverage"], width=0.4, color=ACCENT,
           label="width per borough and time of day (used)")
    ax.axhline(100 * target, color=WARN, linestyle="--", linewidth=1)
    ax.set_xticks(list(x), per_block["hour_block"])
    ax.set_ylim(50, 100)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("% of actuals inside the 90% interval")
    ax.set_title("Interval coverage by time of day, evaluation period (dashed = target)", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _finish(fig, ax, path)
