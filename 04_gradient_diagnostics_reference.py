
"""Reference implementation notes for the fixed-physics gradient diagnostic."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/"config/protocol_v3.json").read_text())
print("Selected physics horizon:",cfg["physics"]["selected_physics_h"])
print("Fixed lambda:",cfg["physics"]["lambda_fixed_physics"])
print("6 h coefficients:",cfg["physics"]["effective_6h_coefficients"])
print("""
Diagnostic definition:
gd = grad(L_data, theta)
gp = grad(L_phys, theta)
cosine = dot(gd,gp)/(||gd|| ||gp|| + 1e-12)
ratio = ||gp||/(||gd|| + 1e-12)

Use 4096 fixed representative training origins, batch=256.
Report median cosine, fraction cosine<0, fraction cosine<-0.25,
median raw physics/data gradient norm ratio, p10 and p90 cosine.
Model selection is based on validation DATA MSE.
""")
