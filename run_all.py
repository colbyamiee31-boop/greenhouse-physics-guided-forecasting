
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SRC=ROOT/"src"

def run(args):
    print("\n>>>"," ".join(map(str,args)),flush=True)
    subprocess.run([sys.executable,*map(str,args)],check=True,cwd=SRC)

def main():
    ap=argparse.ArgumentParser(description="Run the final v3 reproducibility workflow.")
    ap.add_argument("--mode",choices=["quick","full","analysis-only"],default="quick")
    args=ap.parse_args()

    run([SRC/"01_validate_v3_dataset.py"])
    run([SRC/"03_fit_greybox_multiscale.py"])

    if args.mode=="quick":
        # Pipeline smoke test only. Not for manuscript numerical reporting.
        run([
            SRC/"04_train_fixed_physics_and_gradient_diagnostics.py",
            "--max-epochs","3",
            "--bootstrap","100",
            "--diagnostic-epochs","0,1,3",
            "--diagnostic-samples","256"
        ])
        run([SRC/"05_nasa_boundary_uncertainty.py","--bootstrap","100"])
        run([SRC/"06_multiseed_statistics.py","--bootstrap","300"])
    elif args.mode=="full":
        # Manuscript-grade settings. Runtime can be several minutes/hours depending on hardware.
        run([SRC/"02_train_modern_baselines.py"])
        run([
            SRC/"04_train_fixed_physics_and_gradient_diagnostics.py",
            "--max-epochs","14",
            "--bootstrap","700",
            "--diagnostic-epochs","0,1,3,6,9,12,14",
            "--diagnostic-samples","4096"
        ])
        run([SRC/"05_nasa_boundary_uncertainty.py","--bootstrap","300"])
        run([SRC/"06_multiseed_statistics.py","--bootstrap","1000"])
    else:
        # Fastest route for already-trained / archived outputs.
        run([SRC/"06_multiseed_statistics.py","--bootstrap","1000"])

    print("\nWorkflow completed.")

if __name__=="__main__":
    main()
