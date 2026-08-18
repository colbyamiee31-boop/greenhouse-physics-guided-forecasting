
from __future__ import annotations
import argparse, csv, math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import HORIZONS, load_sequence_rows, make_examples

ROOT=Path(__file__).resolve().parents[1]

def holm_adjust(pvals):
    pvals=np.asarray(pvals,float)
    order=np.argsort(pvals)
    adj=np.empty(len(pvals),float)
    running=0.0
    m=len(pvals)
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*pvals[idx])
        adj[idx]=min(1.0,running)
    return adj

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bootstrap",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=20260817)
    args=ap.parse_args()

    archive=np.load(ROOT/"results/v3_multiseed_predictions_correct_nasa.npz")
    seeds=archive["seeds"]
    obs=archive["obs"]
    model_keys=["LSTM","GRU","TCN","Transformer","FixedPhysicsLSTM"]
    display={"FixedPhysicsLSTM":"Fixed physics LSTM"}

    # Reconstruct the exact 4750 winter test origins to recover calendar-day blocks.
    rows=load_sequence_rows(ROOT/"data/greenhouse_model_normalized_v3_correct_nasa.csv")
    _,_,meta=make_examples(rows,"test",1)
    if len(meta)!=obs.shape[0]:
        raise RuntimeError(f"Test-origin mismatch: meta={len(meta)}, archive={obs.shape[0]}")
    origin_days=np.asarray([m["origin"].date().isoformat() for m in meta])
    unique_days=np.unique(origin_days)
    day_idx={d:np.where(origin_days==d)[0] for d in unique_days}

    seed_rows=[]
    summary_rows=[]
    stacks={}
    for key in model_keys:
        arr=archive[key]
        stacks[key]=arr
        for si,seed in enumerate(seeds):
            p=arr[si]
            for hi,h in enumerate(HORIZONS):
                for var,off,unit in [("Temperature",0,"degC"),("Humidity",1,"%RH")]:
                    y=obs[:,2*hi+off]; yp=p[:,2*hi+off]
                    seed_rows.append([
                        display.get(key,key),int(seed),h,var,unit,len(y),
                        math.sqrt(mean_squared_error(y,yp)),
                        mean_absolute_error(y,yp),r2_score(y,yp)
                    ])
        for h in HORIZONS:
            for var in ["Temperature","Humidity"]:
                rr=[r for r in seed_rows if r[0]==display.get(key,key) and r[2]==h and r[3]==var]
                rm=np.asarray([r[6] for r in rr]); ma=np.asarray([r[7] for r in rr]); r2=np.asarray([r[8] for r in rr])
                summary_rows.append([
                    display.get(key,key),h,var,rr[0][4],len(rr),
                    rm.mean(),rm.std(ddof=1),rm.min(),rm.max(),
                    rr[int(np.argmin(rm))][1],rr[int(np.argmax(rm))][1],
                    ma.mean(),ma.std(ddof=1),r2.mean(),r2.std(ddof=1)
                ])

    rng=np.random.default_rng(args.seed)
    pair_rows=[]
    for alt in ["GRU","TCN","Transformer","FixedPhysicsLSTM"]:
        A=stacks[alt]; B=stacks["LSTM"]
        for hi,h in enumerate(HORIZONS):
            for var,off in [("Temperature",0),("Humidity",1)]:
                y=obs[:,2*hi+off]
                a=A[:,:,2*hi+off]; b=B[:,:,2*hi+off]
                rm_b=math.sqrt(np.mean((b-y[None,:])**2))
                rm_a=math.sqrt(np.mean((a-y[None,:])**2))
                point=rm_a-rm_b
                boot=[]
                for _ in range(args.bootstrap):
                    seed_sel=rng.integers(0,len(seeds),size=len(seeds))
                    sampled_days=rng.choice(unique_days,size=len(unique_days),replace=True)
                    idx=np.concatenate([day_idx[d] for d in sampled_days])
                    boot.append(
                        math.sqrt(np.mean((a[seed_sel][:,idx]-y[idx][None,:])**2))-
                        math.sqrt(np.mean((b[seed_sel][:,idx]-y[idx][None,:])**2))
                    )
                lo,med,hi_ci=np.percentile(boot,[2.5,50,97.5])
                daily_diff=[]
                for d in unique_days:
                    idx=day_idx[d]
                    daily_diff.append(
                        np.mean((a[:,idx]-y[idx][None,:])**2)-
                        np.mean((b[:,idx]-y[idx][None,:])**2)
                    )
                stat,p_raw=wilcoxon(daily_diff,zero_method="wilcox",alternative="two-sided")
                pair_rows.append([
                    display.get(alt,alt),h,var,rm_b,rm_a,point,lo,med,hi_ci,
                    float(np.median(daily_diff)),float(stat),float(p_raw),len(unique_days)
                ])

    adj=holm_adjust([r[11] for r in pair_rows])
    strict=[]
    for i,r in enumerate(pair_rows):
        robust_ci=(r[6]>0) or (r[8]<0)
        robust=robust_ci and adj[i]<0.05
        strict.append(r+[float(adj[i]),"better" if r[5]<0 else "worse","yes" if robust else "no"])

    with (ROOT/"results/multiseed_seed_metrics_recomputed.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["model","seed","horizon_h","variable","unit","N","RMSE","MAE","R2"])
        w.writerows(seed_rows)

    with (ROOT/"results/multiseed_summary_recomputed.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["model","horizon_h","variable","unit","n_seeds","RMSE_mean","RMSE_SD","RMSE_min","RMSE_max",
                    "best_seed","worst_seed","MAE_mean","MAE_SD","R2_mean","R2_SD"])
        w.writerows(summary_rows)

    with (ROOT/"results/multiseed_significance_recomputed.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["alternative_vs_LSTM","horizon_h","variable","LSTM_pooled_RMSE","alternative_pooled_RMSE",
                    "delta_RMSE","CI95_low","bootstrap_median","CI95_high","median_daily_MSE_diff",
                    "wilcoxon_stat","p_raw","n_days","holm_p","direction","robust_both"])
        w.writerows(strict)

    print("Seeds:",seeds.tolist())
    print("Test origins:",len(meta),"calendar days:",len(unique_days))
    print("Robust differences vs LSTM:")
    for r in strict:
        if r[-1]=="yes":
            print(r[0],r[1],r[2],r[14],f"delta={r[5]:+.3f}",f"CI[{r[6]:+.3f},{r[8]:+.3f}]",f"p_holm={r[13]:.3g}")

if __name__=="__main__":
    main()
