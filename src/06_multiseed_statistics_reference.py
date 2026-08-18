
"""Statistical reporting protocol used in the final v3 manuscript."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/"config/protocol_v3.json").read_text())
print("Seeds:",cfg["training"]["five_seeds"])
print("""
Report initialization uncertainty as mean ± sample SD across seeds.
For paired model comparisons, resample both:
  (1) random initialization seeds, and
  (2) winter calendar days.
Use paired daily MSE differences for Wilcoxon signed-rank tests.
Apply Holm correction across model/horizon/target comparisons.
Five seeds are NOT five independent test datasets.
""")
