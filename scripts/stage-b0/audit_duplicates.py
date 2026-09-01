#!/usr/bin/env python3
"""Audit exact, story-level, normalized, and heuristic graph/template duplicates."""

from __future__ import annotations

import json

import pandas as pd

from common import ARTIFACTS


def group_summary(frame: pd.DataFrame, column: str) -> dict:
    sizes = frame.groupby(column, dropna=False).size()
    duplicate_sizes = sizes[sizes > 1]
    return {
        "group_key": column,
        "total_groups": int(len(sizes)),
        "duplicate_groups": int(len(duplicate_sizes)),
        "rows_in_duplicate_groups": int(duplicate_sizes.sum()),
        "excess_duplicate_rows": int((duplicate_sizes - 1).sum()),
        "largest_group": int(sizes.max()),
    }


def main():
    path = ARTIFACTS / "stepgame_metadata.csv"
    frame = pd.read_csv(path)
    assert frame.row_id.is_unique
    assert frame.row_id.tolist() == list(range(len(frame)))
    keys = [
        "exact_duplicate_group", "story_group_id", "normalized_duplicate_group",
        "graph_template_group",
    ]
    summary = {key: group_summary(frame, key) for key in keys}
    for key in keys:
        sizes = frame.groupby(key)[key].transform("size")
        frame[f"{key}_size"] = sizes.astype("int32")
    frame.to_csv(path, index=False)

    parquet_status = "not_attempted"
    parquet_error = None
    try:
        import pyarrow  # noqa: F401
        frame.to_parquet(ARTIFACTS / "stepgame_metadata.parquet", index=False, compression="zstd")
        parquet_status = "written"
    except Exception as exc:  # dependency absence is reported, never hidden
        parquet_status = "unavailable"
        parquet_error = f"{type(exc).__name__}: {exc}"

    story_query_counts = frame.groupby("story_group_id").question.nunique()
    summary["story_groups_with_multiple_queries"] = int((story_query_counts > 1).sum())
    summary["rows"] = int(len(frame))
    summary["parquet_status"] = parquet_status
    summary["parquet_error"] = parquet_error
    (ARTIFACTS / "duplicate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
