import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import torch

project = Path("/home/ai/projects/linear-probe-repro")
results = project / "results"
activation_dir = project / "activations/sanity/pythia-2.8b/world/coords"
required = [
    results / "a2_2_sample_manifest.csv",
    results / "a2_2_activation_sanity_report.md",
    results / "a2_2_activation_stats.csv",
    results / "a2_2_hf_tl_comparison.csv",
    results / "a2_2_token_alignment_control.csv",
    results / "a2_2_quantization_roundtrip.csv",
    results / "a2_compatibility_notes.md",
    activation_dir / "activations.pt",
    activation_dir / "metadata.json",
    activation_dir / "tokenization_examples.json",
    activation_dir / "numeric_examples.json",
]
for path in required:
    assert path.is_file() and path.stat().st_size > 0, path

manifest = pd.read_csv(required[0])
stats = pd.read_csv(results / "a2_2_activation_stats.csv")
comparison = pd.read_csv(results / "a2_2_hf_tl_comparison.csv")
quant = pd.read_csv(results / "a2_2_quantization_roundtrip.csv")
metadata = json.loads((activation_dir / "metadata.json").read_text())
numeric = json.loads((activation_dir / "numeric_examples.json").read_text())
activations = torch.load(activation_dir / "activations.pt", map_location="cpu", weights_only=True)

assert len(manifest) == 16
assert manifest.is_test.astype(bool).value_counts().to_dict() == {False: 8, True: 8}
assert sorted(activations) == list(range(32))
assert all(tuple(activations[i].shape) == (16, 2560) for i in range(32))
assert all(activations[i].dtype == torch.float16 for i in range(32))
assert int(stats.nan_count.sum()) == 0 and int(stats.inf_count.sum()) == 0
assert metadata["gate2_status"] == "PASS"
assert subprocess.check_output(
    ["git", "-C", str(project / "external/world-models"), "status", "--porcelain"],
    text=True,
).strip() == ""

processed = comparison[comparison.comparison == "official_processed_TL_vs_HF_block_hook"]
raw = comparison[comparison.comparison == "unprocessed_TL_vs_HF_diagnostic"]

print("manifest_split_counts", manifest.is_test.astype(bool).value_counts().to_dict())
print("manifest_type_counts")
print(manifest.groupby([manifest.is_test.astype(bool), "entity_type"]).size().to_string())
print("activation_layers", len(activations))
print("activation_shapes", {i: list(activations[i].shape) for i in [0, 15, 31]})
print("activation_dtype", activations[0].dtype)
print("nan_total", int(stats.nan_count.sum()), "inf_total", int(stats.inf_count.sum()))
print("representative_stats")
print(stats[stats.layer.isin([0, 15, 31])].to_string(index=False))
print("processed_tl_hf_max_abs", processed.max_absolute_difference.max())
print("processed_tl_hf_min_cos", processed.cosine_similarity.min())
print("raw_tl_hf_max_abs", raw.max_absolute_difference.max())
print("raw_tl_hf_min_cos", raw.cosine_similarity.min())
print("quant_mean_mae", quant.mae.mean())
print("quant_max_error", quant.max_error.max())
print("quant_min_mean_cos", quant.mean_cosine_similarity.min())
print("numeric_examples")
print(json.dumps(numeric, ensure_ascii=False, indent=2))
print("runtime", metadata["extraction_runtime_seconds"], metadata["total_runtime_seconds"])
print("gpu_peaks", json.dumps(metadata["gpu_peak_phases_bytes"]))
print("sha256")
for path in required:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    print(h, path)
print("gate2_output_verification=PASS")
