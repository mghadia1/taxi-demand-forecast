"""Download and verify the 2024 TLC files; aggregate them to an hourly grid."""

from __future__ import annotations

import hashlib
import urllib.request
from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd

BASE = "https://d37ci6vzurychx.cloudfront.net"
BOROUGHS = ("Manhattan", "Queens", "Brooklyn", "Bronx")

# SHA-256 recorded 2026-09-22.
FILES = {
    "yellow_tripdata_2024-01.parquet": "c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510",
    "yellow_tripdata_2024-02.parquet": "c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20",
    "yellow_tripdata_2024-03.parquet": "2d4cdc8fb96726cdd3803b13b02d2e61e71d45720aff0ebc693a8bdd1f249823",
    "yellow_tripdata_2024-04.parquet": "ce37f736c7cf9e92164c850ef2b502a8f7bb590fc1804b6edbcb80293da01b89",
    "yellow_tripdata_2024-05.parquet": "1974230be2b5a3ea92e47b5eb77f32dbf1de896d81ad8df4ab6faba52749ad23",
    "yellow_tripdata_2024-06.parquet": "677cf14c8347f745b583f012fbdba072334c6c4efa17bfd6a64369f2ba30329c",
    "yellow_tripdata_2024-07.parquet": "af12d6b04daeb78550799ae99933d495b27d715b6dbb0fad9bb750a4537d3070",
    "yellow_tripdata_2024-08.parquet": "7315643dd16bafc03e27d9903edbfd35404c2550a695d0da0a3e73852d940cfb",
    "yellow_tripdata_2024-09.parquet": "2f42e4a383a6635de803001462848d0aee2b47f6ce899126805d2d05361fb843",
    "yellow_tripdata_2024-10.parquet": "d19fe05aa0b259af24eb47051735293287f1e6dc263ba04030ec1ab4c6ec2650",
    "yellow_tripdata_2024-11.parquet": "5ef321876de5007a7c147a347389133fc94fd7ac28f82eb1d57d8f2cf05cfd3b",
    "yellow_tripdata_2024-12.parquet": "41ebf7db80bebde60c58e5143c14cdf38ad04a0f3e3ff44215b3e240d55f6c78",
    "taxi_zone_lookup.csv": "1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed",
}

# TLC timestamps are local wall-clock time. On 10 Mar 2024 the clock jumped
# from 02:00 to 03:00, so that hour does not exist; on 3 Nov 2024 01:00-02:00
# happened twice, so that hour holds two hours of trips. Neither is a real
# demand observation: both are kept out of training targets and scoring.
DST_GAP = pd.Timestamp("2024-03-10 02:00")
DST_DOUBLE = pd.Timestamp("2024-11-03 01:00")


def url_for(name: str) -> str:
    folder = "misc" if name.endswith(".csv") else "trip-data"
    return f"{BASE}/{folder}/{name}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path.name}: {actual}")


def download(destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, expected in FILES.items():
        path = destination / name
        if not path.exists():
            urllib.request.urlretrieve(url_for(name), path)
        verify(path, expected)
        paths.append(path)
    return paths


def hourly_counts(trips: list[Path] | pd.DataFrame, zones: Path | pd.DataFrame,
                  start: str = "2024-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET VARIABLE window_start = ?::TIMESTAMP", [start])
    con.execute("SET VARIABLE window_end = ?::TIMESTAMP", [end])
    if isinstance(trips, pd.DataFrame):
        con.register("raw_trips", trips)
    else:
        con.read_parquet([str(p) for p in trips]).create_view("raw_trips")
    if isinstance(zones, pd.DataFrame):
        con.register("raw_zones", zones)
    else:
        con.read_csv(str(zones)).create_view("raw_zones")
    sql = resources.files("taxiforecast").joinpath("sql", "hourly_counts.sql").read_text()
    return con.execute(sql).df()


def complete_grid(counts: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Every (hour, borough) in [start, end), with 0 where no trip started.

    An hour with no pickups in the Bronx is a real zero and must be in the data;
    leaving it out would make lag features skip to an older hour. The one hour
    that never existed (the spring-forward gap) is dropped.
    """
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    hours = hours[hours != DST_GAP]
    grid = pd.MultiIndex.from_product([hours, BOROUGHS], names=["hour", "borough"]).to_frame(index=False)
    merged = grid.merge(counts, on=["hour", "borough"], how="left")
    merged["trips"] = merged["trips"].fillna(0).astype(int)
    merged["dst_irregular"] = merged["hour"] == DST_DOUBLE
    return merged
