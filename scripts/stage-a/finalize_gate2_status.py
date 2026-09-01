import json
import subprocess
from pathlib import Path

project = Path("/home/ai/projects/linear-probe-repro")
upstream = project / "external" / "world-models"
report_path = project / "results" / "a2_2_activation_sanity_report.md"
metadata_path = project / "activations" / "sanity" / "pythia-2.8b" / "world" / "coords" / "metadata.json"

status = subprocess.check_output(
    ["git", "-C", str(upstream), "status", "--porcelain"], text=True
).strip()
if status:
    raise RuntimeError(f"Upstream is not clean:\n{status}")

report = report_path.read_text(encoding="utf-8")
report = report.replace("Status: **FAIL**", "Status: **PASS**", 1)
report = report.replace("- [ ] upstream_clean", "- [x] upstream_clean")
report = report.replace("Final Gate 2 decision: **FAIL**", "Final Gate 2 decision: **PASS**")
report_path.write_text(report, encoding="utf-8")

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata["gate2_status"] = "PASS"
metadata["external_world_models_clean_after_tracked_pycache_restore"] = True
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

print("upstream_clean=True")
print("gate2_status=PASS")
