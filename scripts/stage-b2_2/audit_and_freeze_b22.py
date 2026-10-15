#!/usr/bin/env python3
"""Audit/freeze Stage B2.2 inputs before any Qwen3 test accuracy is computed."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import sys
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from common_b22 import (
    B02, B02C, B1, B156, B2, B21, EXPECTED_B02_PRIMARY_SHA256,
    EXPECTED_B1_MANIFEST_SHA256, EXPECTED_HOP_COUNTS, EXPECTED_ROW_HASH,
    HOPS, MODEL_NAME, NATURAL_CANDIDATES, OUT, PREFERRED_LABELS, RELATIONS,
    ROOT, build_raw_prompt, deterministic_audit_rows, ensure_dirs,
    equal_native_user, equal_raw_prompt, load_primary, native_natural_user,
    primary_invariants, protocol_payloads, read_json, sha256_file, sha256_text,
    stable_json_hash, story_list, write_csv_atomic, write_json_atomic,
)


PROTECTED = [B02, B02C, B1, B156, B2, B21]
TOKENIZER_FILE_NAMES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "chat_template.json", "generation_config.json",
}
LABEL_POOL = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def resolve_model_path(explicit: str | None) -> tuple[Path, list[str]]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("QWEN3_8B_PATH"):
        candidates.append(Path(os.environ["QWEN3_8B_PATH"]).expanduser())
    candidates.extend([
        ROOT / "models" / "Qwen3-8B",
        ROOT / "models" / "Qwen" / "Qwen3-8B",
        ROOT.parent / "models" / "Qwen3-8B",
        Path.home() / "models" / "Qwen3-8B",
        Path.home() / "chenchen" / "models" / "Qwen3-8B",
        Path.home() / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen3-8B",
        Path.home() / ".cache" / "modelscope" / "hub" / "Qwen" / "Qwen3-8B",
        Path("/data/models/Qwen3-8B"),
        Path("/mnt/data/models/Qwen3-8B"),
    ])
    hf_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-8B" / "snapshots"
    if hf_root.is_dir():
        candidates.extend(sorted((path for path in hf_root.iterdir() if path.is_dir()), reverse=True))
    seen = set()
    searched = []
    for candidate in candidates:
        path = candidate.resolve() if candidate.exists() else candidate.absolute()
        if str(path) in seen:
            continue
        seen.add(str(path))
        searched.append(str(path))
        if (path / "config.json").is_file() and (
            (path / "model.safetensors.index.json").is_file()
            or any(path.glob("*.safetensors"))
            or (path / "pytorch_model.bin.index.json").is_file()
            or any(path.glob("pytorch_model*.bin"))
        ):
            return path, searched
    raise FileNotFoundError("No complete local Qwen3-8B candidate found. Searched:\n" + "\n".join(searched))


def weight_manifest(model_path: Path) -> dict:
    index_paths = [model_path / "model.safetensors.index.json", model_path / "pytorch_model.bin.index.json"]
    index_path = next((path for path in index_paths if path.is_file()), None)
    if index_path:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(payload.get("weight_map", {}).values()))
        missing = [name for name in names if not (model_path / name).is_file()]
        shards = [
            {"file": name, "bytes": (model_path / name).stat().st_size, "sha256": sha256_file(model_path / name)}
            for name in names if (model_path / name).is_file()
        ]
        return {
            "format": "indexed",
            "index_file": index_path.name,
            "index_sha256": sha256_file(index_path),
            "declared_shards": len(names),
            "missing_shards": missing,
            "complete": not missing and bool(names),
            "shards": shards,
        }
    files = sorted([*model_path.glob("*.safetensors"), *model_path.glob("pytorch_model*.bin")])
    return {
        "format": "unindexed",
        "declared_shards": len(files),
        "missing_shards": [],
        "complete": bool(files),
        "shards": [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
    }


def snapshot_protected() -> dict:
    files = {}
    roots = []
    for directory in PROTECTED:
        if not directory.is_dir():
            raise FileNotFoundError(f"Protected directory missing: {directory}")
        roots.append(str(directory.relative_to(ROOT)))
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = str(path.relative_to(ROOT)).replace("\\", "/")
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "algorithm": "sha256",
        "protected_roots": roots,
        "file_count": len(files),
        "files": files,
        "snapshot_sha256": stable_json_hash(files),
    }


def normalize_story(value) -> str:
    return json.dumps(story_list(value), ensure_ascii=False, separators=(",", ":"))


def audit_frozen_data(primary: pd.DataFrame) -> dict:
    invariant = primary_invariants(primary)
    b02_path = B02 / "primary_test.csv"
    manifest_path = B1 / "union_manifest.csv"
    if not b02_path.is_file():
        raise FileNotFoundError(b02_path)
    source = pd.read_csv(b02_path)
    source["row_id"] = source.row_id.astype("int64")
    source = source.sort_values("row_id").reset_index(drop=True)
    comparisons = {
        "row_ids_exact_b0_vs_b1": primary.row_id.tolist() == source.row_id.tolist(),
        "b1_manifest_sha256_exact_b21_source": sha256_file(manifest_path) == EXPECTED_B1_MANIFEST_SHA256,
        "b02_primary_sha256_exact_b21_source": sha256_file(b02_path) == EXPECTED_B02_PRIMARY_SHA256,
    }
    source_by_id = source.set_index("row_id")
    primary_by_id = primary.set_index("row_id")
    if comparisons["row_ids_exact_b0_vs_b1"]:
        if "canonical_relation" in source.columns:
            comparisons["gold_relations_exact_b0_vs_b1"] = bool(
                primary_by_id.canonical_relation.astype(str).eq(source_by_id.canonical_relation.astype(str)).all()
            )
        if "story" in source.columns:
            comparisons["story_exact_b0_vs_b1"] = all(
                normalize_story(primary_by_id.loc[row_id, "story"]) == normalize_story(source_by_id.loc[row_id, "story"])
                for row_id in primary_by_id.index
            )
        if "question" in source.columns:
            comparisons["question_exact_b0_vs_b1"] = bool(
                primary_by_id.question.astype(str).str.strip().eq(source_by_id.question.astype(str).str.strip()).all()
            )
        if "prompt" in source.columns:
            comparisons["raw_prompt_exact_b0_vs_b1"] = bool(
                primary_by_id.prompt.astype(str).eq(source_by_id.prompt.astype(str)).all()
            )
    checks = {**invariant["checks"], **comparisons}
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "primary": invariant,
        "frozen_manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256_file(manifest_path)},
        "primary_test": {"path": str(b02_path.relative_to(ROOT)), "sha256": sha256_file(b02_path)},
    }


def native_render(tok, user: str) -> tuple[str, list[int]]:
    kwargs = {
        "conversation": [{"role": "user", "content": user}],
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        text = tok.apply_chat_template(tokenize=False, **kwargs)
        ids = list(tok.apply_chat_template(tokenize=True, **kwargs))
    except TypeError as exc:
        raise RuntimeError("Tokenizer cannot apply enable_thinking=False; full run is prohibited") from exc
    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if any(content.strip() for content in think_blocks):
        raise RuntimeError("enable_thinking=False rendered non-empty thinking content; full run is prohibited")
    reencoded = list(tok(text, add_special_tokens=False).input_ids)
    if reencoded != ids:
        raise RuntimeError("Native rendered text does not round-trip to the tokenized generation prompt")
    return text, ids


def contextual_suffix(tok, prompt_text: str, prompt_ids: list[int], candidate: str, explicit_bos: bool, raw_bos_id: int) -> tuple[list[int], list[int]]:
    text_prompt_ids = list(tok(prompt_text, add_special_tokens=False).input_ids)
    expected = ([raw_bos_id] if explicit_bos else []) + text_prompt_ids
    if prompt_ids != expected:
        raise ValueError("Prompt IDs do not match the rendered text under the frozen BOS policy")
    joint = ([raw_bos_id] if explicit_bos else []) + list(tok(prompt_text + candidate, add_special_tokens=False).input_ids)
    if joint[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Candidate joint tokenization changes tokens before the scoring boundary")
    suffix = joint[len(prompt_ids):]
    if not suffix:
        raise ValueError("Empty contextual candidate suffix")
    return suffix, joint


def token_pieces(tok, token_ids: list[int]) -> list[dict]:
    raw = tok.convert_ids_to_tokens(token_ids)
    return [
        {"id": int(token_id), "token": token, "decoded": tok.decode([token_id], clean_up_tokenization_spaces=False)}
        for token_id, token in zip(token_ids, raw)
    ]


def audit_candidate_boundaries(tok, primary: pd.DataFrame, raw_bos_id: int) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    rendered_examples = []
    for row in deterministic_audit_rows(primary).itertuples(index=False):
        raw_text = build_raw_prompt(row)
        raw_prefix = [raw_bos_id] + list(tok(raw_text, add_special_tokens=False).input_ids)
        native_user = native_natural_user(row)
        native_text, native_prefix = native_render(tok, native_user)
        if len(rendered_examples) < 5:
            rendered_examples.append({
                "row_id": int(row.row_id), "hop": int(row.computed_hop),
                "rendered_text": native_text, "rendered_token_ids": native_prefix,
                "prompt_length": len(native_prefix), "final_16_tokens": token_pieces(tok, native_prefix[-16:]),
                "empty_think_scaffold_present": bool(re.search(r"<think>\s*</think>", native_text, flags=re.DOTALL)),
            })
        for relation in RELATIONS:
            candidate = NATURAL_CANDIDATES[relation]
            standalone = list(tok(candidate, add_special_tokens=False).input_ids)
            for boundary, prompt_text, prefix, explicit_bos in [
                ("raw", raw_text, raw_prefix, True),
                ("native", native_text, native_prefix, False),
            ]:
                suffix, joint = contextual_suffix(tok, prompt_text, prefix, candidate, explicit_bos, raw_bos_id)
                rows.append({
                    "boundary": boundary,
                    "row_id": int(row.row_id),
                    "hop": int(row.computed_hop),
                    "candidate_relation": relation,
                    "candidate_raw_string": candidate,
                    "standalone_token_ids": json.dumps(standalone, separators=(",", ":")),
                    "standalone_token_count": len(standalone),
                    "standalone_token_pieces": json.dumps(token_pieces(tok, standalone), ensure_ascii=False, separators=(",", ":")),
                    "prompt_token_length": len(prefix),
                    "boundary_final_16_ids": json.dumps(prefix[-16:], separators=(",", ":")),
                    "joint_token_ids": json.dumps(joint, separators=(",", ":")),
                    "continuation_token_ids": json.dumps(suffix, separators=(",", ":")),
                    "continuation_token_count": len(suffix),
                    "continuation_token_pieces": json.dumps(token_pieces(tok, suffix), ensure_ascii=False, separators=(",", ":")),
                    "matches_standalone": suffix == standalone,
                    "scoring_uses_contextual_continuation_exactly": True,
                })
    frame = pd.DataFrame(rows)
    expected = 3 * len(HOPS) * len(RELATIONS)
    counts = frame.groupby("boundary").size().to_dict()
    if counts != {"native": expected, "raw": expected}:
        raise RuntimeError(f"Boundary audit count mismatch: {counts}")
    return frame, rendered_examples


def label_is_single_token(tok, label: str, sample_rows: pd.DataFrame, mapping: dict[str, str], raw_bos_id: int) -> tuple[bool, list[dict]]:
    details = []
    candidate = " " + label
    standalone = list(tok(candidate, add_special_tokens=False).input_ids)
    for row in sample_rows.itertuples(index=False):
        raw_text = equal_raw_prompt(row, mapping)
        raw_prefix = [raw_bos_id] + list(tok(raw_text, add_special_tokens=False).input_ids)
        native_text, native_prefix = native_render(tok, equal_native_user(row, mapping))
        raw_suffix, _ = contextual_suffix(tok, raw_text, raw_prefix, candidate, True, raw_bos_id)
        native_suffix, _ = contextual_suffix(tok, native_text, native_prefix, candidate, False, raw_bos_id)
        details.append({
            "row_id": int(row.row_id), "hop": int(row.computed_hop), "label": label,
            "standalone_token_ids": standalone,
            "raw_continuation_token_ids": raw_suffix,
            "native_continuation_token_ids": native_suffix,
            "raw_single_token": len(raw_suffix) == 1,
            "native_single_token": len(native_suffix) == 1,
        })
    passed = len(standalone) == 1 and all(item["raw_single_token"] and item["native_single_token"] for item in details)
    return passed, details


def select_equal_labels(tok, primary: pd.DataFrame, raw_bos_id: int) -> tuple[dict[str, str], dict]:
    sample = deterministic_audit_rows(primary)
    provisional = PREFERRED_LABELS.copy()
    preferred_details = {}
    preferred_ok = True
    for label in list("ABCDEFGH"):
        passed, details = label_is_single_token(tok, label, sample, provisional, raw_bos_id)
        preferred_details[label] = {"passed": passed, "checks": details}
        preferred_ok &= passed
    if preferred_ok:
        mapping = provisional
        algorithm = "preferred_A_to_H_all_audited_raw_and_native_boundaries_single_token"
        audit = preferred_details
    else:
        eligible = []
        pool_audit = {}
        for label in LABEL_POOL:
            passed, details = label_is_single_token(tok, label, sample, provisional, raw_bos_id)
            pool_audit[label] = {"passed": passed, "checks": details}
            if passed:
                eligible.append(label)
            if len(eligible) >= 8:
                break
        if len(eligible) < 8:
            raise RuntimeError("Equal-length Gate failed: fewer than 8 score-independent single-token labels")
        mapping = dict(zip(RELATIONS, eligible[:8]))
        algorithm = "first_8_eligible_from_fixed_AZ_az_09_pool"
        audit = {label: pool_audit[label] for label in eligible[:8]}
        # Re-audit under the final mapping; no model score or accuracy is read.
        for label in eligible[:8]:
            passed, details = label_is_single_token(tok, label, sample, mapping, raw_bos_id)
            audit[label] = {"passed": passed, "checks": details}
            if not passed:
                raise RuntimeError("Selected equal-length label failed final contextual re-audit")
    return mapping, {
        "status": "PASS",
        "preferred_mapping": PREFERRED_LABELS,
        "preferred_passed": preferred_ok,
        "fixed_candidate_pool": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "selection_algorithm": algorithm,
        "relation_to_label": mapping,
        "selection_used_only_tokenizer": True,
        "selection_used_no_model_probabilities": True,
        "selection_used_no_gold_conditioned_accuracy": True,
        "audit": audit,
    }


def model_manifest(model_path: Path, config, tok) -> dict:
    tokenizer_files = {}
    for path in sorted(item for item in model_path.iterdir() if item.is_file()):
        if path.name in TOKENIZER_FILE_NAMES or path.name.startswith("tokenizer"):
            tokenizer_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    template = tok.chat_template
    revision = model_path.name if model_path.parent.name == "snapshots" else getattr(config, "_commit_hash", None)
    msc = model_path / ".msc"
    if revision is None and msc.exists():
        revision = sha256_file(msc) if msc.is_file() else msc.name
    if revision is None:
        revision = "local_directory_no_revision_metadata"
    weights = weight_manifest(model_path)
    if not weights["complete"]:
        raise RuntimeError(f"Incomplete Qwen3-8B weights: {weights['missing_shards']}")
    return {
        "status": "PASS",
        "canonical_model_name": MODEL_NAME,
        "local_path": str(model_path),
        "revision_or_snapshot_id": revision,
        "config_sha256": sha256_file(model_path / "config.json"),
        "tokenizer_files": tokenizer_files,
        "chat_template_present": bool(template),
        "chat_template_sha256": sha256_text(template) if template else None,
        "weight_shard_manifest": weights,
        "requested_dtype": "torch.bfloat16",
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "config_vocab_size": getattr(config, "vocab_size", None),
        "tokenizer_vocab_size": len(tok),
        "bos_token": tok.bos_token,
        "bos_token_id": tok.bos_token_id,
        "config_bos_token_id": getattr(config, "bos_token_id", None),
        "raw_explicit_bos_token_id": tok.bos_token_id if tok.bos_token_id is not None else getattr(config, "bos_token_id", None),
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
    }


def environment_manifest() -> dict:
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append({
                "index": index, "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            })
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpus,
        "transformers_expected": "4.57.6",
        "transformers_matches_expected": transformers.__version__ == "4.57.6",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--phase", choices=["preflight", "boundary"], required=True)
    args = parser.parse_args()
    ensure_dirs()

    primary = load_primary()
    if args.phase == "preflight":
        frozen = audit_frozen_data(primary)
        write_json_atomic(OUT / "audit" / "frozen_data_audit.json", frozen)
        if frozen["status"] != "PASS":
            raise RuntimeError("Frozen data audit FAIL; model loading and full inference are prohibited")

        protected = snapshot_protected()
        write_json_atomic(OUT / "audit" / "protected_files_audit.json", protected)

        environment = environment_manifest()
        write_json_atomic(OUT / "audit" / "environment.json", environment)
        if not environment["cuda_available"]:
            raise RuntimeError("CUDA unavailable; Qwen3-8B main experiment may not run on CPU")

        model_path, searched = resolve_model_path(args.model_path)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        if not tok.chat_template:
            raise RuntimeError("Qwen3 tokenizer has no built-in chat_template")
        manifest = model_manifest(model_path, config, tok)
        manifest["model_identity_sha256"] = stable_json_hash(manifest)
        manifest["searched_paths_until_match"] = searched
        write_json_atomic(OUT / "audit" / "model_manifest.json", manifest)
        repo_inspection = {
            "status": "PASS",
            "stage_b21_scripts": sorted(str(path.relative_to(ROOT)) for path in (ROOT / "scripts" / "stage-b2_1").glob("*.py")),
            "stage_b21_report": str((ROOT / "docs" / "stage-b" / "STAGE_B2_1_MODEL_TASK_FITNESS_REPORT.md").relative_to(ROOT)),
            "frozen_split_source": str((B1 / "union_manifest.csv").relative_to(ROOT)),
            "frozen_primary_crosscheck": str((B02 / "primary_test.csv").relative_to(ROOT)),
            "reused_implementations": {
                "prompt_and_relation_order": "scripts/stage-b2_1/common_b21.py",
                "audit_and_protocol_freeze": "scripts/stage-b2_1/audit_and_freeze_b21.py",
                "teacher_forced_scoring": "scripts/stage-b2_1/run_protocols_b21.py",
                "bootstrap_and_reporting": "scripts/stage-b2_1/finalize_b21.py",
                "independent_validation": "scripts/stage-b2_1/validate_b21.py",
            },
            "stage_b21_artifact_root": str(B21.relative_to(ROOT)),
        }
        write_json_atomic(OUT / "audit" / "repo_inspection.json", repo_inspection)
        preflight = {
            "stage": "B2.2", "phase": "PREFLIGHT", "status": "PASS",
            "frozen_rows": len(primary), "hop_counts": EXPECTED_HOP_COUNTS,
            "row_ids_sha256": EXPECTED_ROW_HASH, "model_path": str(model_path),
            "protected_snapshot_sha256": protected["snapshot_sha256"],
        }
        write_json_atomic(OUT / "audit" / "preflight_summary.json", preflight)
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    else:
        preflight = read_json(OUT / "audit" / "preflight_summary.json")
        smoke = read_json(OUT / "audit" / "model_load_smoke.json")
        if preflight.get("status") != "PASS" or smoke.get("status") != "PASS":
            raise RuntimeError("Preflight and Qwen load smoke must PASS before boundary audit")
        manifest = read_json(OUT / "audit" / "model_manifest.json")
        model_path = Path(manifest["local_path"])
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        protected = read_json(OUT / "audit" / "protected_files_audit.json")

    raw_bos_id = manifest["raw_explicit_bos_token_id"]
    if raw_bos_id is None:
        raise RuntimeError("Neither tokenizer nor model config supplies the explicit BOS required by frozen Raw-Sum")
    boundary, examples = audit_candidate_boundaries(tok, primary, int(raw_bos_id))
    write_csv_atomic(OUT / "audit" / "candidate_boundary_audit.csv", boundary)
    write_json_atomic(OUT / "audit" / "raw_boundary_audit.json", {
        "status": "PASS", "checks": int((boundary.boundary == "raw").sum()),
        "all_scoring_continuations_exact": bool(boundary.loc[boundary.boundary.eq("raw"), "scoring_uses_contextual_continuation_exactly"].all()),
        "records": boundary.loc[boundary.boundary.eq("raw")].to_dict("records"),
        "records_sha256": sha256_file(OUT / "audit" / "candidate_boundary_audit.csv"),
    })
    write_json_atomic(OUT / "audit" / "native_boundary_audit.json", {
        "status": "PASS", "enable_thinking": False, "add_generation_prompt": True,
        "checks": int((boundary.boundary == "native").sum()),
        "all_scoring_continuations_exact": bool(boundary.loc[boundary.boundary.eq("native"), "scoring_uses_contextual_continuation_exactly"].all()),
        "rendered_examples": examples,
        "records": boundary.loc[boundary.boundary.eq("native")].to_dict("records"),
        "records_sha256": sha256_file(OUT / "audit" / "candidate_boundary_audit.csv"),
    })
    write_json_atomic(OUT / "audit" / "tokenizer_audit.json", {
        "status": "PASS", "relation_order": RELATIONS, "natural_candidates": NATURAL_CANDIDATES,
        "raw_checks": int((boundary.boundary == "raw").sum()),
        "native_checks": int((boundary.boundary == "native").sum()),
        "raw_matches_standalone": int(boundary.loc[boundary.boundary.eq("raw"), "matches_standalone"].sum()),
        "native_matches_standalone": int(boundary.loc[boundary.boundary.eq("native"), "matches_standalone"].sum()),
        "important": "contextual continuation IDs, not assumed standalone IDs, are used for scoring",
    })

    labels, label_audit = select_equal_labels(tok, primary, int(raw_bos_id))
    write_json_atomic(OUT / "audit" / "equal_length_gate.json", label_audit)

    protocols = protocol_payloads(labels)
    protocol_hashes = {}
    for name, payload in protocols.items():
        payload["protocol_sha256"] = stable_json_hash(payload)
        protocol_hashes[name] = payload["protocol_sha256"]
        write_json_atomic(OUT / "audit" / f"protocol_{name}.json", payload)
    write_json_atomic(OUT / "audit" / "protocol_checksums.json", protocol_hashes)

    (OUT / "audit" / "future_work.md").write_text(
        "# Future work\n\nQwen3 thinking-mode free generation plus final-answer parsing is deliberately outside Stage B2.2. "
        "It must not be substituted for the preregistered non-thinking teacher-forced Native-Sum result.\n",
        encoding="utf-8",
    )
    summary = {
        "stage": "B2.2", "phase": "AUDIT_AND_FREEZE", "status": "PASS",
        "frozen_rows": len(primary), "hop_counts": EXPECTED_HOP_COUNTS,
        "row_ids_sha256": EXPECTED_ROW_HASH, "model_path": str(model_path),
        "boundary_checks": len(boundary), "equal_length_mapping": labels,
        "protected_snapshot_sha256": protected["snapshot_sha256"],
        "protocol_checksums": protocol_hashes,
    }
    write_json_atomic(OUT / "audit" / "audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
