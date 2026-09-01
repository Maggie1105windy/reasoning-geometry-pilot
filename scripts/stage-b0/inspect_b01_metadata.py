#!/usr/bin/env python3
"""Read-only B0.1 metadata profile and parsing-status audit."""

from common import OUT, add_parsing_status, load_metadata, primary_candidates, write_json


def main() -> None:
    df = add_parsing_status(load_metadata())
    candidates, excluded = primary_candidates(df)
    required = [
        "row_id", "story_group_id", "exact_duplicate_group", "computed_hop",
        "canonical_relation", "template_ids", "template_families",
        "unordered_composition_signature", "target_absent_anywhere",
    ]
    audit = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "row_id_unique": bool(df["row_id"].is_unique),
        "required_columns_present": {c: c in df.columns for c in required},
        "parsing_status": {k: int(v) for k, v in df["parsing_status"].value_counts().items()},
        "primary_before_exact_dedup": int(len(candidates) + len(excluded)),
        "primary_after_exact_dedup": int(len(candidates)),
        "primary_exact_duplicates_excluded": int(len(excluded)),
        "candidate_story_groups": int(candidates["story_group_id"].nunique()),
        "candidate_exact_groups_unique": bool(candidates["exact_duplicate_group"].is_unique),
    }
    assert audit["rows"] == 210000
    assert audit["row_id_unique"]
    assert all(audit["required_columns_present"].values())
    assert audit["parsing_status"] == {"FULL_OK": 190237, "PATH_ONLY": 10574, "FAILED": 9189}
    write_json(OUT / "metadata_inspection.json", audit)
    print(audit)


if __name__ == "__main__":
    main()
