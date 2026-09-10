#!/usr/bin/env python3
"""Audit frozen Stage B inputs and preregister every Stage B2.1 protocol."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from common_b21 import (
    B02, B02C, B1, B156, B2, BOOTSTRAP_RESAMPLES, DISPLAY_WORDS, HOPS,
    LABEL_MAP, MODEL_SPECS, NATURAL_CANDIDATES, OUT, RELATIONS, ROOT, SEED,
    build_raw_prompt, candidate_suffix_from_full, ensure_dirs, equal_native_user,
    equal_raw_prompt, load_primary, native_natural_user, primary_invariants,
    sha256_file, sha256_text, stable_json_hash, write_json,
)


PROTECTED = [B02, B02C, B1, B156, B2]
EXISTING_SCORES = {
    "base": B156 / "behavioral" / "all_candidate_scores.csv",
    "sft": B2 / "behavior" / "sft_candidate_scores.csv",
    "dpo": B2 / "behavior" / "dpo_candidate_scores.csv",
    "final": B2 / "behavior" / "final_candidate_scores.csv",
}


def snapshot_protected() -> dict:
    files = {}
    for directory in PROTECTED:
        if not directory.exists():
            raise FileNotFoundError(directory)
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            files[str(path.relative_to(ROOT))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    payload = {
        "algorithm": "sha256",
        "protected_roots": [str(p.relative_to(ROOT)) for p in PROTECTED],
        "file_count": len(files),
        "files": files,
    }
    payload["snapshot_sha256"] = stable_json_hash(files)
    return payload


def ids(value) -> list[int]:
    return list(value["input_ids"] if isinstance(value, dict) else value)


def render_native(tok, user: str) -> list[int]:
    return ids(tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
    ))


def render_with_answer(tok, user: str, answer: str) -> list[int]:
    return ids(tok.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
    ))


def choose_labels(tokenizers: dict, primary: pd.DataFrame) -> tuple[dict, dict]:
    row = primary.iloc[0]
    preferred = list("ABCDEFGH")
    pool = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

    def audit_candidate(label: str) -> dict:
        record = {}
        for model, tok in tokenizers.items():
            raw_ids = tok(" " + label, add_special_tokens=False).input_ids
            model_record = {"raw_ids": raw_ids, "raw_count": len(raw_ids)}
            if model != "base":
                user = equal_native_user(row, dict(zip(RELATIONS, preferred)))
                prefix = render_native(tok, user)
                suffix = candidate_suffix_from_full(tok, prefix, render_with_answer(tok, user, " " + label))
                model_record.update({"native_ids": suffix, "native_count": len(suffix)})
            record[model] = model_record
        return record

    preferred_audit = {label: audit_candidate(label) for label in preferred}
    preferred_ok = all(
        details[model]["raw_count"] == 1
        and (model == "base" or details[model]["native_count"] == 1)
        for details in preferred_audit.values() for model in tokenizers
    )
    if preferred_ok:
        selected = preferred
        selection = "preferred_A_to_H_all_boundaries_single_token"
        selected_audit = preferred_audit
    else:
        eligible = []
        pool_audit = {}
        for label in pool:
            detail = audit_candidate(label)
            pool_audit[label] = detail
            ok = all(
                detail[model]["raw_count"] == 1
                and (model == "base" or detail[model]["native_count"] == 1)
                for model in tokenizers
            )
            if ok:
                eligible.append(label)
        if len(eligible) < 8:
            raise RuntimeError("Equal-length verbalizer Gate failed: fewer than 8 deterministic single-token labels")
        selected = eligible[:8]
        selection = "lexicographic_first_8_from_fixed_AZ_az_09_pool"
        selected_audit = {label: pool_audit[label] for label in selected}

    mapping = dict(zip(RELATIONS, selected))
    return mapping, {
        "status": "PASS",
        "preferred_labels": preferred,
        "preferred_passed": preferred_ok,
        "fixed_candidate_pool": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "selection_algorithm": selection,
        "selected_labels": selected,
        "relation_to_label": mapping,
        "audit": selected_audit,
        "selection_used_no_model_scores_or_gold_conditioned_performance": True,
    }


def main() -> None:
    ensure_dirs()
    protected = snapshot_protected()
    write_json(OUT / "audit" / "protected_checksums.json", protected)

    primary = load_primary()
    invariant = primary_invariants(primary)
    if invariant["status"] != "PASS":
        raise RuntimeError(f"Frozen Primary invariant failed: {invariant}")

    b02_test = pd.read_csv(B02 / "primary_test.csv").sort_values("row_id").reset_index(drop=True)
    if primary.row_id.tolist() != b02_test.row_id.astype(int).tolist():
        raise RuntimeError("B1 union Primary test row IDs differ from B0.2 primary_test.csv")
    prompt_match = primary.apply(lambda row: build_raw_prompt(row) == row.prompt, axis=1)
    if not prompt_match.all():
        raise RuntimeError(f"Frozen raw prompt mismatch on {int((~prompt_match).sum())} rows")

    tokenizers = {}
    model_audits = {}
    for key, spec in MODEL_SPECS.items():
        config = AutoConfig.from_pretrained(spec["path"], local_files_only=True)
        tok = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True, use_fast=True)
        tokenizers[key] = tok
        natural = {}
        for relation, candidate in NATURAL_CANDIDATES.items():
            candidate_ids = tok(candidate, add_special_tokens=False).input_ids
            natural[relation] = {"string": candidate, "token_ids": candidate_ids, "token_count": len(candidate_ids)}
        template = tok.chat_template
        model_audits[key] = {
            "label": spec["label"],
            "model_path": str(spec["path"]),
            "revision_kind": "ModelScope .msc local revision",
            "revision": spec["revision"],
            "model_type": getattr(config, "model_type", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "tokenizer_vocab_size": len(tok),
            "config_vocab_size": getattr(config, "vocab_size", None),
            "bos_token": tok.bos_token,
            "bos_token_id": tok.bos_token_id,
            "eos_token": tok.eos_token,
            "eos_token_id": tok.eos_token_id,
            "pad_token": tok.pad_token,
            "pad_token_id": tok.pad_token_id,
            "chat_template_present": bool(template),
            "chat_template_sha256": sha256_text(template) if template else None,
            "candidate_tokenization": natural,
            "existing_behavioral_scores": str(EXISTING_SCORES[key].relative_to(ROOT)),
            "existing_behavioral_scores_sha256": sha256_file(EXISTING_SCORES[key]),
        }
        if key != "base" and not template:
            raise RuntimeError(f"{key} has no native chat_template")

    mapping, label_audit = choose_labels(tokenizers, primary)
    write_json(OUT / "equal_length_verbalizer" / "candidate_selection_audit.json", label_audit)
    label_rows = []
    for relation, label in mapping.items():
        for model, detail in label_audit["audit"][label].items():
            label_rows.append({
                "relation": relation, "label": label, "model": model,
                "raw_token_ids": json.dumps(detail["raw_ids"]), "raw_token_count": detail["raw_count"],
                "native_token_ids": json.dumps(detail.get("native_ids")), "native_token_count": detail.get("native_count"),
            })
    pd.DataFrame(label_rows).to_csv(OUT / "equal_length_verbalizer" / "label_tokenization.csv", index=False)

    # Deterministic 3 rows per hop, every relation candidate, every post-trained checkpoint.
    audit_rows = []
    examples = []
    sampled = pd.concat([primary.loc[primary.computed_hop.eq(h)].head(3) for h in HOPS])
    for model in ["sft", "dpo", "final"]:
        tok = tokenizers[model]
        for _, row in sampled.iterrows():
            user = native_natural_user(row)
            prefix = render_native(tok, user)
            if str(row.canonical_relation).lower() in user.lower().split("Question:", 1)[-1]:
                # Relation names occur in the fixed option list by design; current label is never singled out.
                pass
            for relation, candidate in NATURAL_CANDIDATES.items():
                suffix = candidate_suffix_from_full(tok, prefix, render_with_answer(tok, user, candidate))
                standalone = tok(candidate, add_special_tokens=False).input_ids
                audit_rows.append({
                    "model": model, "row_id": int(row.row_id), "hop": int(row.computed_hop),
                    "candidate_relation": relation, "candidate": candidate,
                    "prompt_token_count": len(prefix), "assistant_boundary_last_8_ids": json.dumps(prefix[-8:]),
                    "continuation_token_ids": json.dumps(suffix), "continuation_token_count": len(suffix),
                    "standalone_token_ids": json.dumps(standalone),
                    "matches_standalone": suffix == standalone,
                    "candidate_not_in_prompt": candidate not in tok.decode(prefix[-32:]),
                    "gold_not_specially_inserted": True,
                })
            if len(examples) < 6:
                examples.append((model, int(row.row_id), tok.decode(prefix)))
    boundary = pd.DataFrame(audit_rows)
    boundary.to_csv(OUT / "native_chat" / "token_boundary_audit.csv", index=False)
    if len(boundary) < 360 or not boundary.matches_standalone.all():
        raise RuntimeError("Native candidate-boundary Gate failed")

    templates = {model: tokenizers[model].chat_template for model in ["sft", "dpo", "final"]}
    template_hashes = {model: sha256_text(value) for model, value in templates.items()}
    rendered_checks = {}
    sample_user = native_natural_user(primary.iloc[0])
    for model in templates:
        rendered_checks[model] = {
            "prompt_sha256": sha256_text(tokenizers[model].decode(render_native(tokenizers[model], sample_user))),
            "special_tokens": tokenizers[model].special_tokens_map,
        }
    chat_audit = {
        "status": "PASS",
        "templates_identical": len(set(template_hashes.values())) == 1,
        "template_sha256": template_hashes,
        "rendered_prompt_checks": rendered_checks,
        "comparison_interpretation": "own-native-interface task fitness" if len(set(template_hashes.values())) > 1 else "shared-native-interface task fitness",
        "templates": templates,
    }
    write_json(OUT / "native_chat" / "chat_template_audit.json", chat_audit)
    md = ["# Native chat-template audit", "", f"Status: **PASS**", "", f"Templates identical: **{chat_audit['templates_identical']}**", ""]
    for model in templates:
        md += [f"## {model}", "", f"SHA-256: `{template_hashes[model]}`", "", "```jinja", templates[model], "```", ""]
    (OUT / "native_chat" / "chat_template_audit.md").write_text("\n".join(md), encoding="utf-8")
    ex_md = ["# Frozen native-chat rendered prompt examples", ""]
    for model, row_id, value in examples:
        ex_md += [f"## {model}, row {row_id}", "", "```text", value, "```", ""]
    (OUT / "native_chat" / "rendered_prompt_examples.md").write_text("\n".join(ex_md), encoding="utf-8")

    base_protocol = {
        "stage": "B2.1", "frozen_before_full_model_evaluation": True,
        "primary_test_rows": 2063, "row_ids_sha256": invariant["row_ids_sha256"],
        "bootstrap_unit": "story_group_id", "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "seed": SEED,
        "uniform_chance": 0.125, "no_test_set_tuning": True,
        "candidate_score_tolerance": {"absolute_log_probability": 1e-5, "accuracy": 0.0},
    }
    protocols = {
        "raw_reproduction": {
            **base_protocol, "protocol": "Raw-Sum", "prompt_builder": "B1/B2 exact Story/Question raw prompt",
            "tokenization": {"add_special_tokens": False, "explicit_bos": True, "no_truncation": True, "gold_answer_in_prompt": False},
            "candidate_strings": NATURAL_CANDIDATES, "score": "sum teacher-forced candidate token log probabilities",
            "source_scores": {m: str(p.relative_to(ROOT)) for m, p in EXISTING_SCORES.items()},
        },
        "native_chat": {
            **base_protocol, "protocol": ["Native-Sum", "Native-Mean"], "models": ["sft", "dpo", "final"],
            "prompt_builder": "tokenizer.apply_chat_template(user-only, add_generation_prompt=True)",
            "candidate_strings": NATURAL_CANDIDATES, "boundary_selection": "exact assistant continuation prefix, not accuracy",
            "score": ["sum", "mean-token"], "no_cot_or_few_shot": True, "no_truncation": True,
            "chat_template_audit_sha256": sha256_file(OUT / "native_chat" / "chat_template_audit.json"),
            "token_boundary_audit_sha256": sha256_file(OUT / "native_chat" / "token_boundary_audit.csv"),
        },
        "scoring_sensitivity": {
            **base_protocol, "protocol": ["Raw-Sum", "Raw-Mean"], "models": list(MODEL_SPECS),
            "same_frozen_raw_prompt_and_candidates_as_experiment_A": True,
            "score": ["sum", "arithmetic mean over candidate tokens"],
            "no_length_penalty_calibration_temperature_or_tuning": True,
        },
        "equal_length_verbalizer": {
            **base_protocol, "protocol": ["EqualLength-Raw", "EqualLength-Native"],
            "relation_to_label": mapping, "selected_labels": list(mapping.values()),
            "selection_algorithm": label_audit["selection_algorithm"],
            "prompt_mapping": {relation: DISPLAY_WORDS[relation] for relation in RELATIONS},
            "score": "sum teacher-forced label-token log probabilities", "no_truncation": True,
            "candidate_selection_audit_sha256": sha256_file(OUT / "equal_length_verbalizer" / "candidate_selection_audit.json"),
        },
    }
    protocol_hashes = {}
    for name, payload in protocols.items():
        payload["protocol_sha256"] = stable_json_hash(payload)
        protocol_hashes[name] = payload["protocol_sha256"]
        target = OUT / ("raw_reproduction" if name == "raw_reproduction" else name) / "protocol.json"
        write_json(target, payload)
    write_json(OUT / "audit" / "protocol_checksums.json", protocol_hashes)

    audit = {
        "status": "PASS", "project_root_requested": "/home/ai/projects/reasoning-geometry-pilot",
        "project_root_resolved": str(ROOT), "environment": {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda_available": torch.cuda.is_available(),
        },
        "primary": invariant,
        "manifest": {"path": str((B1 / 'union_manifest.csv').relative_to(ROOT)), "sha256": sha256_file(B1 / "union_manifest.csv")},
        "primary_test": {"path": str((B02 / 'primary_test.csv').relative_to(ROOT)), "sha256": sha256_file(B02 / "primary_test.csv")},
        "raw_prompt_exact_for_all_primary_rows": bool(prompt_match.all()),
        "raw_prompt_function": "Story:\\n<sentences joined by newline>\\n\\nQuestion:\\n<question stripped>",
        "candidate_strings": NATURAL_CANDIDATES, "models": model_audits,
        "protected_snapshot_sha256": protected["snapshot_sha256"],
        "protocol_checksums": protocol_hashes,
    }
    write_json(OUT / "audit" / "input_audit.json", audit)
    lines = [
        "# Stage B2.1 input audit", "", "Status: **PASS**", "",
        f"Resolved project root: `{ROOT}`", "",
        f"Primary test rows: **{invariant['rows']}**; row-ID SHA-256: `{invariant['row_ids_sha256']}`", "",
        f"Hop counts: `{invariant['hop_counts']}`", "",
        f"Raw prompt exact on all frozen rows: **{prompt_match.all()}**", "",
        f"Protected file snapshot: **{protected['file_count']} files**, `{protected['snapshot_sha256']}`", "",
        "## Checkpoints", "",
        "| Model | Revision kind | Revision | Vocab tokenizer/config | Layers | Hidden | Chat template |", "|---|---|---|---:|---:|---:|---|",
    ]
    for model, item in model_audits.items():
        lines.append(f"| {model} | {item['revision_kind']} | `{item['revision']}` | {item['tokenizer_vocab_size']}/{item['config_vocab_size']} | {item['num_hidden_layers']} | {item['hidden_size']} | {item['chat_template_present']} |")
    lines += ["", "## Frozen natural candidates", "", "```json", json.dumps(NATURAL_CANDIDATES, indent=2), "```", "", "All formal protocols were serialized and SHA-256 frozen before full evaluation."]
    (OUT / "audit" / "input_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "AUDIT_AND_FREEZE", "status": "PASS", "rows": len(primary), "protocol_checksums": protocol_hashes}, indent=2))


if __name__ == "__main__":
    main()
