import datetime
import os
from typing import Dict, Iterable, List, Tuple

import pathway as pw


class _DailyFileSchema(pw.Schema):
    ts: pw.DateTimeNaive
    path: str


def _iter_daily_files(base_dir: str) -> Iterable[Tuple[datetime.datetime, str]]:
    """Yield parsed daily pickle files with a date prefix (YYYY-MM-DD)."""
    for name in os.listdir(base_dir):
        if not name.endswith(".pkl"):
            continue
        date_part = name[:10]
        try:
            ts = datetime.datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            continue
        yield (ts, os.path.join(base_dir, name))


def discover_monthly_shards_with_pathway(
    base_dir: str, min_days: int = 10
) -> List[Dict[str, object]]:
    """
    Group daily pickle files into month buckets using Pathway's temporal.session window.

    Args:
        base_dir: Directory containing daily *.pkl files (filenames start with YYYY-MM-DD).
        min_days: Minimum number of daily files required to keep a discovered month.

    Returns:
        List of shard dicts compatible with fine_tune_month, each containing:
          - month (YYYY-MM)
          - month_start / month_end
          - shard_path (points back to base_dir)
          - daily_paths (sorted list of file paths in that month)
          - processed (default False)
    """
    raw_rows: List[Tuple[datetime.datetime, str] | Dict[str, object]] = list(
        _iter_daily_files(base_dir)
    )
    if not raw_rows:
        return []

    rows: List[Tuple[datetime.datetime, str]] = []
    for idx, item in enumerate(raw_rows):
        normalized: Tuple[datetime.datetime, str]
        if isinstance(item, dict):
            normalized = (item["ts"], item["path"])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            normalized = (item[0], item[1])
        else:
            try:
                unpacked = tuple(item)
            except Exception as exc:
                raise TypeError(
                    f"Unsupported row type at index {idx}: {type(item)}"
                ) from exc
            if len(unpacked) != 2:
                raise TypeError(
                    f"Row at index {idx} has length {len(unpacked)}; expected 2-tuple (ts, path)"
                )
            normalized = (unpacked[0], unpacked[1])

        if not isinstance(normalized, tuple) or len(normalized) != 2:
            raise TypeError(
                f"Normalized row at index {idx} is invalid type {type(normalized)} with length {len(normalized)}"
            )
        rows.append(normalized)

    table = pw.debug.table_from_rows(_DailyFileSchema, rows)

    # Session window: keep grouping as long as consecutive days stay in the same month.
    monthly = (
        table.windowby(
            table.ts,
            window=pw.temporal.session(
                predicate=lambda a, b: a.year == b.year and a.month == b.month
            ),
        )
        .reduce(
            month_start=pw.reducers.min(table.ts),
            month_end=pw.reducers.max(table.ts),
            days=pw.reducers.count(),
            paths=pw.reducers.sorted_tuple(table.path),
        )
    )

    pdf = pw.debug.table_to_pandas(monthly)
    pdf["month"] = pdf["month_start"].dt.strftime("%Y-%m")

    shards: List[Dict[str, object]] = []
    for _, row in pdf.iterrows():
        if row["days"] < min_days:
            continue
        shards.append(
            {
                "month": row["month"],
                "month_start": row["month_start"].strftime("%Y-%m-%d"),
                "month_end": row["month_end"].strftime("%Y-%m-%d"),
                "shard_path": base_dir,
                "daily_paths": list(row["paths"]),
                "processed": False,
            }
        )

    shards.sort(key=lambda s: s["month"])
    return shards
