#!/usr/bin/env python3
"""Paired story-group bootstrap for frozen Stage B2 within-hop probes.

The unit of resampling is story_group_id.  One bootstrap draw is shared by all
four checkpoints, layers, and pre-registered summaries for a given hop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ai/projects/reasoning-geometry-pilot")
OUT = ROOT / "artifacts/stage_b2/within_hop"
MODELS = ("base", "sft", "dpo", "final")
PAIRS = (("base", "sft"), ("sft", "dpo"), ("dpo", "final"), ("base", "final"))
LAYERS = tuple(range(32))
LATE = tuple(range(16, 32))
REPRESENTATIVE = (0, 7, 15, 23, 31)
BASE_PEAK = {1: 17, 2: 19, 3: 31, 4: 30, 5: 1}
LABELS = (
    "ABOVE", "BELOW", "LEFT", "RIGHT", "UPPER_LEFT", "UPPER_RIGHT",
    "LOWER_LEFT", "LOWER_RIGHT",
)
SEED = 20260901
N_BOOT = 1000


def macro_f1_from_conf(conf: np.ndarray) -> np.ndarray:
    """Compute 8-class macro-F1 from [..., true, predicted] confusion arrays."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    actual = conf.sum(axis=-1)
    predicted = conf.sum(axis=-2)
    denom = 2.0 * tp + (predicted - tp) + (actual - tp)
    per_class = np.divide(2.0 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)
    return per_class.mean(axis=-1)


def load_predictions() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        path = OUT / f"{model}_primary_predictions.csv"
        frame = pd.read_csv(path)
        frame["model"] = model
        frames.append(frame[["model", "row_id", "story_group_id", "hop", "layer", "true_relation", "predicted_relation"]])
    all_predictions = pd.concat(frames, ignore_index=True)
    # The frozen probe test split has 2,063 rows (behavior uses a different
    # 2,435-row Primary test population), so derive this invariant from Base.
    n_test_rows = int(frames[0].loc[frames[0]["layer"].eq(0), "row_id"].nunique())
    expected = len(MODELS) * len(LAYERS) * n_test_rows
    if len(all_predictions) != expected:
        raise RuntimeError(f"prediction count {len(all_predictions)} != {expected}")
    key = ["row_id", "story_group_id", "hop", "layer"]
    reference = frames[0][key + ["true_relation"]].sort_values(key).reset_index(drop=True)
    for model, frame in zip(MODELS[1:], frames[1:]):
        other = frame[key + ["true_relation"]].sort_values(key).reset_index(drop=True)
        if not reference.equals(other):
            raise RuntimeError(f"prediction alignment differs for {model}")
    return all_predictions


def build_group_confusions(frame: pd.DataFrame, groups: list[str]) -> np.ndarray:
    """Return [model, layer, group, true, pred] integer confusion counts."""
    model_idx = {name: i for i, name in enumerate(MODELS)}
    group_idx = {name: i for i, name in enumerate(groups)}
    label_idx = {name: i for i, name in enumerate(LABELS)}
    shape = (len(MODELS), len(LAYERS), len(groups), len(LABELS), len(LABELS))
    result = np.zeros(shape, dtype=np.int16)
    mi = frame["model"].map(model_idx).to_numpy()
    li = frame["layer"].to_numpy(dtype=np.int64)
    gi = frame["story_group_id"].map(group_idx).to_numpy()
    ti = frame["true_relation"].map(label_idx).to_numpy()
    pi = frame["predicted_relation"].map(label_idx).to_numpy()
    if any(np.isnan(x).any() for x in (mi, gi, ti, pi)):
        raise RuntimeError("unmapped model, group, or relation label")
    np.add.at(result, (mi.astype(int), li, gi.astype(int), ti.astype(int), pi.astype(int)), 1)
    return result


def summarize_delta(values: np.ndarray, point_values: np.ndarray, later: int, earlier: int, layers: tuple[int, ...]) -> tuple[float, np.ndarray]:
    boot_delta = values[:, later, layers].mean(axis=-1) - values[:, earlier, layers].mean(axis=-1)
    point_delta = float(point_values[later, layers].mean() - point_values[earlier, layers].mean())
    return point_delta, boot_delta


def main() -> None:
    pred = load_predictions()
    rng = np.random.default_rng(SEED)
    rows = []
    audit = {
        "status": "PASS",
        "seed": SEED,
        "resamples": N_BOOT,
        "unit": "story_group_id",
        "same_group_resample_across_all_models": True,
        "models": list(MODELS),
        "stage_pairs": [f"{a}->{b}" for a, b in PAIRS],
        "late_layers": list(LATE),
        "representative_layers": list(REPRESENTATIVE),
        "base_peak_layers": BASE_PEAK,
        "hops": {},
    }
    for hop in range(1, 6):
        h = pred[pred["hop"] == hop].copy()
        groups = sorted(h["story_group_id"].unique().tolist())
        conf_by_group = build_group_confusions(h, groups)
        point_conf = conf_by_group.sum(axis=2)
        point_f1 = macro_f1_from_conf(point_conf)
        # Multinomial bootstrap counts; one draw is reused across all model/layer cells.
        weights = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)), size=N_BOOT)
        boot_conf_flat = np.einsum(
            "bg,mlgk->bmlk",
            weights,
            conf_by_group.reshape(len(MODELS), len(LAYERS), len(groups), -1),
            optimize=True,
        )
        boot_conf = boot_conf_flat.reshape(N_BOOT, len(MODELS), len(LAYERS), len(LABELS), len(LABELS))
        boot_f1 = macro_f1_from_conf(boot_conf)
        audit["hops"][str(hop)] = {
            "n_rows": int(h[h["model"] == "base"]["row_id"].nunique()),
            "n_story_groups": len(groups),
        }

        summary_layers = {
            "all_layer_mean": LAYERS,
            "late_layer_mean": LATE,
            "base_peak_layer": (BASE_PEAK[hop],),
        }
        summary_layers.update({f"representative_layer_{layer}": (layer,) for layer in REPRESENTATIVE})
        for earlier_name, later_name in PAIRS:
            earlier = MODELS.index(earlier_name)
            later = MODELS.index(later_name)
            for summary, layers in summary_layers.items():
                point, delta = summarize_delta(boot_f1, point_f1, later, earlier, layers)
                lo, hi = np.quantile(delta, [0.025, 0.975])
                rows.append({
                    "earlier_model": earlier_name,
                    "later_model": later_name,
                    "stage_pair": f"{later_name}-{earlier_name}",
                    "hop": hop,
                    "summary": summary,
                    "layers": ",".join(map(str, layers)),
                    "point_delta_macro_f1": point,
                    "bootstrap_mean_delta_macro_f1": float(delta.mean()),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                    "n_story_groups": len(groups),
                    "n_bootstrap": N_BOOT,
                    "seed": SEED,
                })

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "paired_probe_bootstrap.csv", index=False)
    with open(OUT / "paired_probe_bootstrap_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(json.dumps({"status": "PASS", "rows": len(result), "output": str(OUT / "paired_probe_bootstrap.csv")}, indent=2))


if __name__ == "__main__":
    main()
