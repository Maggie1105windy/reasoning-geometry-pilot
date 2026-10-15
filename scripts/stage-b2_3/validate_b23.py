#!/usr/bin/env python3
"""Phase 8: independent validation from raw row-generation JSONL files."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from transformers import AutoTokenizer


ROOT = Path(os.environ.get("REASONING_GEOMETRY_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = Path(os.environ.get("STAGE_B23_OUT", ROOT / "artifacts" / "stage_b2_3_qwen3_generation")).resolve()
B1 = ROOT / "artifacts" / "stage_b1"
MODEL_PATH = ROOT / "models" / "Qwen3-8B"
RELATIONS = [
    "ABOVE", "BELOW", "LEFT", "RIGHT",
    "UPPER_LEFT", "UPPER_RIGHT", "LOWER_LEFT", "LOWER_RIGHT",
]
WORDS = {
    "above": "ABOVE", "below": "BELOW", "left": "LEFT", "right": "RIGHT",
    "upper-left": "UPPER_LEFT", "upper-right": "UPPER_RIGHT",
    "lower-left": "LOWER_LEFT", "lower-right": "LOWER_RIGHT",
}
EXPECTED_COUNTS = {1: 712, 2: 505, 3: 422, 4: 282, 5: 142}
EXPECTED_HASH = "dbc5bf84903b9d177c7f02f32e4e6226082c61be7498e466b7e3f8e0c84ee950"
PRIMARY_SEED = 20260924
PILOT_SEEDS = [20260924, 20260925, 20260926]
OUTPUT_INSTRUCTION = """End your response with exactly one final line in the following format:

Final answer: <relation>

where <relation> must be one of:
above, below, left, right, upper-left, upper-right, lower-left, lower-right."""
PARSER = re.compile(
    r"(?im)^[ \t]*Final[ \t]+answer[ \t]*:[ \t]*"
    r"(above|below|left|right|upper(?:[ _-]+)left|upper(?:[ _-]+)right|"
    r"lower(?:[ _-]+)left|lower(?:[ _-]+)right)[ \t]*$"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def row_hash(values) -> str:
    return text_digest("\n".join(str(int(value)) for value in values) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_number}") from exc
    return rows


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def parse_answer(text: str) -> str:
    matches = list(PARSER.finditer(text))
    if not matches:
        return "INVALID"
    word = re.sub(r"[ _-]+", "-", matches[-1].group(1).strip().lower())
    return WORDS.get(word, "INVALID")


def story_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Story is not a JSON list")
    return [str(item) for item in parsed]


def raw_prompt(row) -> str:
    return "Story:\n" + "\n".join(story_list(row.story)) + "\n\nQuestion:\n" + str(row.question).strip()


def independent_metrics(frame: pd.DataFrame) -> dict:
    truth = frame.gold_relation.to_numpy()
    pred = frame.parsed_relation.to_numpy()
    accuracy = float(np.mean(truth == pred))
    macro = float(f1_score(truth, pred, labels=RELATIONS, average="macro", zero_division=0))
    balanced = float(recall_score(truth, pred, labels=RELATIONS, average="macro", zero_division=0))
    recalls = recall_score(truth, pred, labels=RELATIONS, average=None, zero_division=0)
    counts = confusion_matrix(truth, pred, labels=RELATIONS)
    return {
        "n": len(frame),
        "accuracy": accuracy,
        "macro_f1": macro,
        "balanced_accuracy": balanced,
        "parse_success_rate": float(np.mean(pred != "INVALID")),
        "per_class_recall": {relation: float(value) for relation, value in zip(RELATIONS, recalls)},
        "prediction_frequency": {relation: float(np.mean(pred == relation)) for relation in RELATIONS},
        "invalid_frequency": float(np.mean(pred == "INVALID")),
        "confusion_matrix": counts.tolist(),
    }


def load_primary() -> pd.DataFrame:
    frame = pd.read_csv(B1 / "union_manifest.csv")
    frame["row_id"] = frame.row_id.astype(int)
    frame["computed_hop"] = frame.computed_hop.astype(int)
    return frame.loc[frame.primary_split.eq("test")].sort_values("row_id").reset_index(drop=True)


def validate_full(protocol_key: str, expected_thinking: bool, primary: pd.DataFrame,
                  tokenizer, resolved_max: int) -> tuple[dict, pd.DataFrame, list[str]]:
    path = OUT / "full" / protocol_key / "row_generations.jsonl"
    rows = read_jsonl(path)
    frame = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
    errors = []
    expected_ids = primary.row_id.tolist()
    checks = {
        "jsonl_rows_2063": len(frame) == 2063,
        "row_ids_unique": bool(frame.row_id.is_unique),
        "frozen_row_coverage_exact": frame.row_id.astype(int).tolist() == expected_ids,
        "primary_seed_exact": bool(frame.generation_seed.eq(PRIMARY_SEED).all()),
        "thinking_flag_exact": bool(frame.enable_thinking.eq(expected_thinking).all()),
        "generated_token_ids_present": bool(frame.generated_token_ids.map(lambda value: isinstance(value, list)).all()),
        "generated_text_present": bool(frame.generated_text.map(lambda value: isinstance(value, str)).all()),
        "max_new_tokens_exact": bool(frame.generation_config.map(lambda value: int(value["max_new_tokens"]) == resolved_max).all()),
    }
    parsed = frame.final_response_text.map(parse_answer)
    checks["independent_parser_exact"] = bool(parsed.eq(frame.parsed_relation).all())
    frame["parsed_relation_independent"] = parsed
    truth = primary.set_index("row_id").loc[frame.row_id.astype(int)]
    checks["gold_exact"] = truth.canonical_relation.tolist() == frame.gold_relation.tolist()
    checks["hop_exact"] = truth.computed_hop.astype(int).tolist() == frame.hop.astype(int).tolist()
    checks["correct_exact"] = bool((frame.gold_relation.eq(parsed)).eq(frame.correct.astype(bool)).all())
    think_boundary_success = pd.Series(False, index=frame.index, dtype=bool)
    if expected_thinking:
        # Qwen3 tokenizer/template revisions differ on whether the opening
        # <think> token is part of the prompt or the generated continuation.
        # Reconstruct the boundary independently from the recorded token flags
        # instead of trusting the runner's legacy prompt-only success flag.
        think_boundary_success = (
            (frame.prompt_has_think_open.astype(bool) | frame.generated_has_think_open.astype(bool))
            & frame.generated_has_think_close.astype(bool)
        )
        checks["normal_think_boundary_exists"] = bool(think_boundary_success.any())
        checks["thinking_content_exists"] = bool(frame.thinking_token_count.gt(0).any())
    else:
        checks["no_nonempty_thinking_content"] = bool(frame.thinking_text.fillna("").str.strip().eq("").all())
        checks["zero_thinking_tokens"] = bool(frame.thinking_token_count.eq(0).all())

    expected_core = (
        {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
        if expected_thinking
        else {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
    )
    checks["generation_config_core_exact"] = bool(frame.generation_config.map(
        lambda value: all(value.get(key) == expected for key, expected in expected_core.items())
    ).all())

    prompt_hashes = []
    for row in primary.itertuples(index=False):
        content = raw_prompt(row) + "\n\n" + OUTPUT_INSTRUCTION
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=expected_thinking,
        )
        prompt_hashes.append(text_digest(prompt))
    checks["prompt_hashes_exact"] = prompt_hashes == frame.prompt_text_hash.tolist()
    if not all(checks.values()):
        errors.extend([f"{protocol_key}:{key}" for key, value in checks.items() if not value])

    frame = frame.rename(columns={"parsed_relation_independent": "parsed_relation_validated"})
    per_hop = {}
    for hop in [1, 2, 3, 4, 5]:
        part = frame.loc[frame.hop.eq(hop)].copy()
        part["parsed_relation"] = part.parsed_relation_validated
        per_hop[str(hop)] = independent_metrics(part)
    derived = {
        metric: float(np.mean([per_hop[str(hop)][metric] for hop in [2, 3, 4, 5]]))
        for metric in ["accuracy", "macro_f1", "balanced_accuracy", "parse_success_rate"]
    }
    result = {
        "checks": checks,
        "per_hop": per_hop,
        "derived_h2_h5": derived,
        "truncation_rate": float(frame.hit_max_new_tokens.mean()),
        "think_parse_success_rate": float(think_boundary_success.mean()),
        "row_generations_sha256": digest(path),
    }
    return result, frame, errors


def compare_reported_metrics(validated: dict, protocol: str) -> bool:
    reported = pd.read_csv(OUT / "statistics" / "full_metrics_by_hop.csv")
    reported = reported.loc[reported.protocol.eq(protocol)].set_index("hop")
    for hop in [1, 2, 3, 4, 5]:
        for metric in ["accuracy", "macro_f1", "balanced_accuracy", "parse_success_rate"]:
            if not np.isclose(float(reported.loc[hop, metric]), validated["per_hop"][str(hop)][metric], atol=1e-12):
                return False
    derived = pd.read_csv(OUT / "statistics" / "derived_summary.csv").set_index("protocol")
    for metric in ["accuracy", "macro_f1", "balanced_accuracy", "parse_success_rate"]:
        if not np.isclose(float(derived.loc[protocol, metric]), validated["derived_h2_h5"][metric], atol=1e-12):
            return False
    return True


def validate_pilot() -> dict:
    checks = {}
    for protocol in ["fg_nonthinking", "fg_thinking"]:
        for seed in PILOT_SEEDS:
            rows = read_jsonl(OUT / "pilot" / protocol / f"seed_{seed}" / "row_generations.jsonl")
            ids = [int(row["row_id"]) for row in rows]
            checks[f"{protocol}_{seed}_rows_250"] = len(rows) == 250 and len(set(ids)) == 250
            checks[f"{protocol}_{seed}_seed_exact"] = all(int(row["generation_seed"]) == seed for row in rows)
    for protocol in ["fg_nonthinking", "fg_thinking"]:
        rows = read_jsonl(OUT / "pilot" / "matched_decoding" / protocol / "row_generations.jsonl")
        checks[f"matched_{protocol}_rows_250"] = len(rows) == 250 and len({row["row_id"] for row in rows}) == 250
        checks[f"matched_{protocol}_seed_exact"] = all(int(row["generation_seed"]) == PRIMARY_SEED for row in rows)
        checks[f"matched_{protocol}_config_exact"] = all(
            row["generation_config"]["temperature"] == 0.6
            and row["generation_config"]["top_p"] == 0.95
            and row["generation_config"]["top_k"] == 20
            and row["generation_config"]["min_p"] == 0.0
            for row in rows
        )
    return checks


def validate_protected() -> tuple[bool, list[str]]:
    snapshot = read_json(OUT / "audit" / "protected_snapshot.json")
    changed = []
    for relative, expected in snapshot["files"].items():
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            changed.append(relative)
    return not changed, changed


def validate_figures() -> tuple[bool, dict]:
    dimensions = {}
    expected = [
        "figure_1_behavior_by_hop.png",
        "figure_2_derived_metrics.png",
        "figure_3_thinking_gain_by_hop.png",
        "figure_4_confusion_matrices.png",
        "figure_5_prediction_distribution.png",
        "figure_6_reasoning_token_length.png",
    ]
    for name in expected:
        path = OUT / "figures" / name
        if not path.is_file() or path.stat().st_size == 0:
            dimensions[name] = None
            continue
        with Image.open(path) as image:
            dimensions[name] = list(image.size)
    return all(value and value[0] >= 800 and value[1] >= 400 for value in dimensions.values()), dimensions


def update_final_artifacts(status: str) -> None:
    decision_path = OUT / "fitness_decision.json"
    decision = read_json(decision_path)
    decision["validation_status"] = status
    write_json(decision_path, decision)
    report_path = OUT / "STAGE_B2_3_QWEN3_GENERATION_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    report = re.sub(r"<!-- VALIDATION_STATUS -->[^*\n]+", f"<!-- VALIDATION_STATUS -->{status}", report, count=1)
    write_text(report_path, report)


def final_console_summary(status: str, summary: dict) -> str:
    decision = read_json(OUT / "fitness_decision.json")
    fgn = summary["independent_metrics"]["FG-N"]
    fgt = summary["independent_metrics"]["FG-T"]
    paired = pd.read_csv(OUT / "statistics" / "paired_bootstrap.csv")
    think = paired.loc[
        paired.comparison.eq("FG-T minus FG-N") & paired.scope.eq("h2-h5-derived") & paired.metric.eq("accuracy")
    ].iloc[0]
    gen = pd.read_csv(OUT / "statistics" / "generation_stats.csv")
    think_tokens = gen.loc[gen.protocol.eq("FG-T")].set_index("hop").median_thinking_tokens
    overall = pd.read_csv(OUT / "statistics" / "overall_metrics.csv").set_index("protocol")
    lines = [
        "Stage B2.3 Qwen3-8B Free-Generation / Thinking Gate",
        "",
        f"Validation:\n{status}",
        "",
        "Frozen data:\nPASS",
        "",
        f"Pilot stochasticity:\n{decision['pilot_stochasticity']}",
        "",
        "B2.2 Candidate Native-Sum:\nh2-h5 derived accuracy = 0.1853",
        "",
        "FG-N Native Non-Thinking Free Generation:",
        *[f"h{hop} = {fgn['per_hop'][str(hop)]['accuracy']:.4f}" for hop in [1, 2, 3, 4, 5]],
        f"h2-h5 derived accuracy = {fgn['derived_h2_h5']['accuracy']:.4f}",
        f"h2-h5 Macro-F1 = {fgn['derived_h2_h5']['macro_f1']:.4f}",
        f"h2-h5 Balanced Accuracy = {fgn['derived_h2_h5']['balanced_accuracy']:.4f}",
        f"parse rate = {fgn['derived_h2_h5']['parse_success_rate']:.4f}",
        f"collapse = {'YES' if bool(overall.loc['FG-N', 'prediction_collapse']) else 'NO'}",
        "",
        "FG-T Native Thinking Free Generation:",
        *[f"h{hop} = {fgt['per_hop'][str(hop)]['accuracy']:.4f}" for hop in [1, 2, 3, 4, 5]],
        f"h2-h5 derived accuracy = {fgt['derived_h2_h5']['accuracy']:.4f}",
        f"h2-h5 Macro-F1 = {fgt['derived_h2_h5']['macro_f1']:.4f}",
        f"h2-h5 Balanced Accuracy = {fgt['derived_h2_h5']['balanced_accuracy']:.4f}",
        f"parse rate = {fgt['derived_h2_h5']['parse_success_rate']:.4f}",
        f"collapse = {'YES' if bool(overall.loc['FG-T', 'prediction_collapse']) else 'NO'}",
        "",
        "FG-T - FG-N:",
        f"derived accuracy difference = {think.point_difference:.4f}",
        f"95% CI = [{think.ci95_low:.4f}, {think.ci95_high:.4f}]",
        f"thinking benefit = {decision['thinking_improves_free_generation']}",
        "",
        "Median thinking tokens:",
        *[f"h{hop} = {think_tokens.loc[hop]:.1f}" for hop in [1, 2, 3, 4, 5]],
        "",
        f"Candidate likelihood underestimates behavior:\n{decision['candidate_likelihood_underestimates_behavior']}",
        "",
        f"Qwen3 normal-use StepGame fitness:\n{decision['normal_use_task_fitness']}",
        "",
        f"Strong-reasoning target:\n{decision['strong_reasoning_target']}",
        "",
        f"Recommended next step:\n{decision['recommended_next_step']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    primary = load_primary()
    checks = {
        "frozen_rows_2063": len(primary) == 2063,
        "frozen_hop_counts_exact": primary.groupby("computed_hop").size().to_dict() == EXPECTED_COUNTS,
        "frozen_row_hash_exact": row_hash(primary.row_id) == EXPECTED_HASH,
        "frozen_rows_unique": bool(primary.row_id.is_unique),
        "frozen_relations_exact": set(primary.canonical_relation) == set(RELATIONS),
    }
    generation_config = read_json(OUT / "audit" / "generation_config.json")
    resolved_max = int(generation_config["resolved_max_new_tokens"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)

    fgn, fgn_frame, fgn_errors = validate_full("fg_nonthinking", False, primary, tokenizer, resolved_max)
    fgt, fgt_frame, fgt_errors = validate_full("fg_thinking", True, primary, tokenizer, resolved_max)
    checks.update({f"fgn_{key}": value for key, value in fgn["checks"].items()})
    checks.update({f"fgt_{key}": value for key, value in fgt["checks"].items()})
    checks["reported_fgn_metrics_exact"] = compare_reported_metrics(fgn, "FG-N")
    checks["reported_fgt_metrics_exact"] = compare_reported_metrics(fgt, "FG-T")
    pilot_checks = validate_pilot()
    checks.update(pilot_checks)
    protected_ok, protected_changed = validate_protected()
    checks["protected_b0_to_b22_unchanged"] = protected_ok
    figures_ok, figure_dimensions = validate_figures()
    checks["six_figures_present_and_readable"] = figures_ok
    required = [
        OUT / "fitness_decision.json",
        OUT / "STAGE_B2_3_QWEN3_GENERATION_REPORT.md",
        OUT / "statistics" / "paired_bootstrap.csv",
        OUT / "comparisons" / "error_transitions.csv",
    ]
    checks["required_outputs_present"] = all(path.is_file() and path.stat().st_size > 0 for path in required)

    decision = read_json(OUT / "fitness_decision.json")
    limitations = []
    if decision.get("pilot_stochasticity") == "HIGH":
        limitations.append("Pilot seed-to-seed stochasticity is HIGH.")
    if fgt["think_parse_success_rate"] < 0.98:
        limitations.append(f"FG-T think-boundary parse success is {fgt['think_parse_success_rate']:.4f}.")
    if max(fgn["truncation_rate"], fgt["truncation_rate"]) > 0:
        limitations.append("At least one full-run generation reached max_new_tokens.")
    errors = [key for key, value in checks.items() if not value] + fgn_errors + fgt_errors
    if errors:
        status = "FAIL"
    elif limitations:
        status = "PASS_WITH_LIMITATION"
    else:
        status = "PASS"
    summary = {
        "status": status,
        "checks": checks,
        "errors": sorted(set(errors)),
        "limitations": limitations,
        "protected_changed_files": protected_changed,
        "figure_dimensions": figure_dimensions,
        "independent_metrics": {"FG-N": fgn, "FG-T": fgt},
    }
    write_json(OUT / "validation_summary.json", summary)
    write_json(OUT / "validation" / "validation_summary.json", summary)
    report = "# Stage B2.3 Independent Validation\n\n"
    report += f"Status: **{status}**\n\n"
    report += f"Checks passed: {sum(bool(value) for value in checks.values())}/{len(checks)}.\n\n"
    report += "Validation independently read the two full `row_generations.jsonl` files, reconstructed parser outputs, metrics, confusion matrices, relation recalls, prediction frequencies, derived h2–h5 means, truncation, prompt hashes, pilot coverage, generation configuration, protected artifacts, and figure readability. It did not trust `metrics.csv` as a source.\n\n"
    if limitations:
        report += "## Limitations\n\n" + "\n".join(f"- {item}" for item in limitations) + "\n\n"
    if errors:
        report += "## Errors\n\n" + "\n".join(f"- `{item}`" for item in sorted(set(errors))) + "\n"
    else:
        report += "No validation errors were found.\n"
    write_text(OUT / "validation_report.md", report)
    write_text(OUT / "validation" / "validation_report.md", report)
    update_final_artifacts(status)
    console = final_console_summary(status, summary)
    write_text(OUT / "FINAL_SUMMARY.txt", console)
    print(console)
    if status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

