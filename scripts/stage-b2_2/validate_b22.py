#!/usr/bin/env python3
"""Independent Stage B2.2 validator; intentionally imports no run helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


ROOT = Path(os.environ.get("REASONING_GEOMETRY_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = Path(os.environ.get("STAGE_B22_OUT", ROOT / "artifacts" / "stage_b2_2_qwen3")).resolve()
B1 = ROOT / "artifacts" / "stage_b1"
if not B1.exists():
    B1 = ROOT / "results" / "stage-b1"
RELATIONS = ["ABOVE", "BELOW", "LEFT", "RIGHT", "UPPER_LEFT", "UPPER_RIGHT", "LOWER_LEFT", "LOWER_RIGHT"]
HOPS = [1, 2, 3, 4, 5]
EXPECTED_COUNTS = {1: 712, 2: 505, 3: 422, 4: 282, 5: 142}
EXPECTED_HASH = "dbc5bf84903b9d177c7f02f32e4e6226082c61be7498e466b7e3f8e0c84ee950"
SOURCES = {
    "Raw-Sum": (OUT / "raw_sum" / "row_scores.csv", "sum_log_probability"),
    "Raw-Mean": (OUT / "raw_sum" / "row_scores.csv", "mean_log_probability"),
    "Native-Sum": (OUT / "native_sum" / "row_scores.csv", "sum_log_probability"),
    "Native-Mean": (OUT / "native_sum" / "row_scores.csv", "mean_log_probability"),
    "EqualLength-Raw": (OUT / "scoring_sensitivity" / "equal_length_raw" / "row_scores.csv", "sum_log_probability"),
    "EqualLength-Native": (OUT / "scoring_sensitivity" / "equal_length_native" / "row_scores.csv", "sum_log_probability"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def row_hash(values) -> str:
    return hashlib.sha256(("\n".join(str(int(value)) for value in values) + "\n").encode()).hexdigest()


def reconstruct(scores: pd.DataFrame, column: str) -> pd.DataFrame:
    order = {relation: index for index, relation in enumerate(RELATIONS)}
    ordered = scores.assign(_order=scores.candidate_relation.map(order)).sort_values(
        ["row_id", column, "_order"], ascending=[True, False, True]
    )
    return ordered.drop_duplicates("row_id")[["row_id", "candidate_relation"]].rename(columns={"candidate_relation": "prediction"})


def main() -> None:
    errors, limitations, checks = [], [], {}
    union = pd.read_csv(B1 / "union_manifest.csv")
    primary = union.loc[union.primary_split.eq("test")].sort_values("row_id").reset_index(drop=True)
    primary["row_id"] = primary.row_id.astype("int64")
    primary["computed_hop"] = primary.computed_hop.astype(int)
    checks["frozen_rows_2063"] = len(primary) == 2063
    checks["frozen_hop_counts_exact"] = primary.groupby("computed_hop").size().to_dict() == EXPECTED_COUNTS
    checks["frozen_row_hash_exact"] = row_hash(primary.row_id) == EXPECTED_HASH
    checks["frozen_rows_unique"] = primary.row_id.is_unique
    checks["frozen_relations_exact"] = set(primary.canonical_relation) == set(RELATIONS)
    for audit_name in ["preflight_summary", "model_load_smoke", "audit_summary", "tokenizer_audit", "equal_length_gate"]:
        audit_path = OUT / "audit" / f"{audit_name}.json"
        checks[f"{audit_name}_pass"] = audit_path.is_file() and json.loads(audit_path.read_text(encoding="utf-8")).get("status") == "PASS"
    run_meta_path = OUT / "audit" / "run_full_metadata.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.is_file() else {}
    checks["full_run_pass"] = run_meta.get("status") == "PASS"
    checks["all_four_components_pass"] = all(
        run_meta.get("components", {}).get(component, {}).get("status") == "PASS"
        for component in ["raw_natural", "native_natural", "equal_raw", "equal_native"]
    )

    stored_metrics = pd.read_csv(OUT / "statistics" / "all_hop_metrics.csv")
    all_predictions = {}
    for protocol, (path, column) in SOURCES.items():
        if not path.is_file():
            errors.append(f"missing score file: {path}")
            continue
        scores = pd.read_csv(path)
        prefix = protocol.lower().replace("-", "_")
        checks[f"{prefix}_candidate_rows"] = len(scores) == 2063 * 8
        checks[f"{prefix}_every_frozen_row_once"] = sorted(scores.row_id.astype(int).unique().tolist()) == primary.row_id.tolist()
        checks[f"{prefix}_eight_candidates"] = bool(scores.groupby("row_id").size().eq(8).all())
        checks[f"{prefix}_candidate_set"] = bool(scores.groupby("row_id").candidate_relation.apply(set).map(lambda value: value == set(RELATIONS)).all())
        checks[f"{prefix}_finite"] = bool(np.isfinite(scores[["sum_log_probability", "mean_log_probability"]].to_numpy(dtype=float)).all())
        checks[f"{prefix}_no_truncation"] = bool((scores.prompt_token_length > 0).all() and (scores.candidate_token_count > 0).all())
        prediction = primary[["row_id", "story_group_id", "computed_hop", "canonical_relation"]].merge(
            reconstruct(scores, column), on="row_id", validate="one_to_one"
        )
        all_predictions[protocol] = prediction
        for hop in HOPS:
            part = prediction.loc[prediction.computed_hop.eq(hop)]
            values = {
                "accuracy": accuracy_score(part.canonical_relation, part.prediction),
                "macro_f1": f1_score(part.canonical_relation, part.prediction, labels=RELATIONS, average="macro", zero_division=0),
                "balanced_accuracy": recall_score(part.canonical_relation, part.prediction, labels=RELATIONS, average="macro", zero_division=0),
            }
            stored = stored_metrics.loc[
                (stored_metrics.model.eq("qwen3_8b")) & (stored_metrics.protocol.eq(protocol)) & (stored_metrics.hop.eq(hop))
            ]
            if len(stored) != 1:
                errors.append(f"missing/duplicate stored metric: {protocol}/h{hop}")
                continue
            for metric, value in values.items():
                if abs(float(stored.iloc[0][metric]) - float(value)) > 1e-15:
                    errors.append(f"metric mismatch: {protocol}/h{hop}/{metric}")
            matrix = confusion_matrix(part.canonical_relation, part.prediction, labels=RELATIONS)
            matrix_path = OUT / "statistics" / "confusion_matrices" / protocol.lower().replace("-", "_") / f"h{hop}_count.csv"
            if not matrix_path.is_file():
                errors.append(f"missing confusion matrix: {matrix_path}")
            else:
                stored_matrix = pd.read_csv(matrix_path, index_col=0).reindex(index=RELATIONS, columns=RELATIONS).to_numpy()
                if not np.array_equal(matrix, stored_matrix):
                    errors.append(f"confusion mismatch: {protocol}/h{hop}")

    checks["independent_argmax_metrics_confusions"] = not any("mismatch" in item for item in errors)
    protected = json.loads((OUT / "audit" / "protected_files_audit.json").read_text(encoding="utf-8"))
    changed = []
    for relative, item in protected["files"].items():
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            changed.append(relative)
    checks["protected_b0_to_b21_unchanged"] = not changed
    if changed:
        errors.append(f"protected files changed: {changed[:20]}")

    expected_outputs = [
        OUT / "fitness_decision.json", OUT / "STAGE_B2_2_QWEN3_REPORT.md",
        OUT / "statistics" / "bootstrap_ci.csv", OUT / "statistics" / "paired_bootstrap.csv",
        OUT / "statistics" / "derived_summary.csv", OUT / "statistics" / "tulu_reference_reproduction.json",
        OUT / "scoring_sensitivity" / "cardinal_diagonal_metrics.csv",
    ]
    missing = [str(path) for path in expected_outputs if not path.is_file()]
    checks["required_outputs_present"] = not missing
    if missing:
        errors.append(f"missing outputs: {missing}")
    tulu_check_path = OUT / "statistics" / "tulu_reference_reproduction.json"
    checks["tulu_reference_reproduced"] = tulu_check_path.is_file() and json.loads(tulu_check_path.read_text(encoding="utf-8")).get("status") == "PASS"
    figures = sorted((OUT / "figures").glob("figure_*.png"))
    dimensions = {path.name: Image.open(path).size for path in figures}
    checks["five_nonempty_figures"] = len(figures) >= 5 and all(width >= 1200 and height >= 700 for width, height in dimensions.values())

    environment = json.loads((OUT / "audit" / "environment.json").read_text(encoding="utf-8"))
    if environment.get("transformers") != "4.57.6":
        limitations.append(f"transformers={environment.get('transformers')} instead of preregistered 4.57.6")
    checks = {key: bool(value) for key, value in checks.items()}
    errors.extend([f"failed check: {key}" for key, value in checks.items() if not value])
    status = "FAIL" if errors else ("PASS_WITH_LIMITATION" if limitations else "PASS")
    summary = {
        "status": status, "checks": checks, "errors": errors, "limitations": limitations,
        "protected_changed_files": changed, "figure_dimensions": dimensions,
    }
    (OUT / "validation").mkdir(parents=True, exist_ok=True)
    for path in [OUT / "validation_summary.json", OUT / "validation" / "validation_summary.json"]:
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Stage B2.2 independent validation", "", f"Final status: **{status}**", "", "## Checks", ""]
    lines += [f"- {'PASS' if value else 'FAIL'}: `{key}`" for key, value in checks.items()]
    if limitations:
        lines += ["", "## Limitations", ""] + [f"- {item}" for item in limitations]
    if errors:
        lines += ["", "## Errors", ""] + [f"- {item}" for item in errors]
    for path in [OUT / "validation_report.md", OUT / "validation" / "validation_report.md"]:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("Independent validation FAIL: " + "; ".join(errors))

    metrics = stored_metrics.loc[stored_metrics.model.eq("qwen3_8b")]
    derived = pd.read_csv(OUT / "statistics" / "derived_summary.csv")
    paired = pd.read_csv(OUT / "statistics" / "paired_bootstrap.csv")
    decision = json.loads((OUT / "fitness_decision.json").read_text(encoding="utf-8"))
    raw = metrics.loc[metrics.protocol.eq("Raw-Sum")].set_index("hop")
    native = metrics.loc[metrics.protocol.eq("Native-Sum")].set_index("hop")
    native_derived = derived.loc[(derived.model.eq("qwen3_8b")) & (derived.protocol.eq("Native-Sum"))].iloc[0]
    comparison = paired.loc[(paired.protocol.eq("Native-Sum")) & (paired.scope.eq("h2_h5_derived")) & (paired.metric.eq("accuracy"))].iloc[0]
    print("Stage B2.2 Qwen3-8B\n")
    print(f"Frozen data: {status if status == 'FAIL' else 'PASS'}")
    print(f"Model audit: {json.loads((OUT/'audit'/'model_manifest.json').read_text())['status']}")
    print(f"Boundary audit: {json.loads((OUT/'audit'/'tokenizer_audit.json').read_text())['status']}\n")
    print("Qwen Raw-Sum:")
    for hop in HOPS: print(f"h{hop}={raw.loc[hop, 'accuracy']:.4f}")
    print(f"h2-h5 derived={raw.loc[[2,3,4,5], 'accuracy'].mean():.4f}\n")
    print("Qwen Native-Sum:")
    for hop in HOPS: print(f"h{hop}={native.loc[hop, 'accuracy']:.4f}")
    print(f"h2-h5 derived={native.loc[[2,3,4,5], 'accuracy'].mean():.4f}\n")
    print("Tülu Final Native-Sum derived=0.1829")
    print(f"Qwen - Tülu difference={comparison.point_difference:+.4f}")
    print(f"95% CI=[{comparison.ci95_low:.4f}, {comparison.ci95_high:.4f}]\n")
    print(f"Qwen Native Macro-F1 h2-h5={native_derived.macro_f1_h2_h5:.4f}")
    print(f"Qwen Native BalancedAcc h2-h5={native_derived.balanced_accuracy_h2_h5:.4f}\n")
    print(f"Prediction collapse:\nRaw={decision['prediction_collapse']['raw']}\nNative={decision['prediction_collapse']['native']}\n")
    print(f"Protocol sensitivity:\n{decision['protocol_sensitivity']['status']}\n")
    print(f"Strong-reasoning target:\n{decision['strong_reasoning_target']}\n")
    print(f"Recommended next step:\n{decision['recommended_next_step']}")


if __name__ == "__main__":
    main()
