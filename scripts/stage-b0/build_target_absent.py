#!/usr/bin/env python3
"""Primary-membership-preserving Derived Target-Absent subset."""

import pandas as pd

from common import RELATIONS, SPLITS, distribution, split_group_overlap


def build_target_absent(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    subset = primary.loc[
        primary["computed_hop"].between(2, 5)
        & primary["target_absent_anywhere"].astype(bool)
    ].copy()
    dist = distribution(subset, [2, 3, 4, 5])
    hop_counts = {
        str(h): int(subset["computed_hop"].eq(h).sum()) for h in [2, 3, 4, 5]
    }
    info = {
        "total": int(len(subset)),
        "sizes": {s: int(subset["split"].eq(s).sum()) for s in SPLITS},
        "hop_counts": hop_counts,
        "min_relation_cell_overall": int(
            subset.groupby(["computed_hop", "canonical_relation"], observed=True).size()
            .reindex(pd.MultiIndex.from_product([[2, 3, 4, 5], RELATIONS]), fill_value=0).min()
        ),
        "min_cells_by_split": {
            s: int(dist.loc[dist["split"].eq(s), "count"].min()) for s in SPLITS
        },
        "h5_sizes_by_split": {
            s: int((subset["split"].eq(s) & subset["computed_hop"].eq(5)).sum()) for s in SPLITS
        },
        "story_group_overlap": split_group_overlap(subset, "story_group_id"),
    }
    return subset, info

