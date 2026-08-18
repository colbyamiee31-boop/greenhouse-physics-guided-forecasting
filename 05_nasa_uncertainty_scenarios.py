
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/"config/protocol_v3.json").read_text())
rows=[["Reference",0,1,1]]
for b in cfg["nasa_stress_tests"]["temperature_bias_C"]: rows.append([f"T2M {b:+g}C",b,1,1])
for s in cfg["nasa_stress_tests"]["solar_scale"]: rows.append([f"Solar {100*(s-1):+g}%",0,1,s])
for w in cfg["nasa_stress_tests"]["wind_scale"]: rows.append([f"Wind {100*(w-1):+g}%",0,w,1])
rows += [["Combined low",-1,0.8,0.9],["Combined high",1,1.2,1.1]]
with (ROOT/"results/nasa_stress_test_scenarios.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f); w.writerow(["scenario","T2M_bias_C","wind_scale","solar_scale"]); w.writerows(rows)
print("Controlled stress tests only; not measured NASA-vs-onsite bias.")
