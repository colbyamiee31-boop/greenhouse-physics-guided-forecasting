
from __future__ import annotations

import argparse
import bisect
import copy
import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import (
    HORIZONS, LSTMModel, load_sequence_rows, make_examples,
    predict_raw, read_normalization_params, set_seed
)

ROOT = Path(__file__).resolve().parents[1]

def build_physics_targets(master_path: Path, config: dict):
    """
    Build the retrospective 6 h coarse-grained physics target used by the final v3 analysis.

    IMPORTANT:
    - The recurrent model input remains strictly historical/current at forecast origin.
    - The 6 h interval-mean forcing below is used ONLY to construct the retrospective
      physics-loss / physics-consistency target.
    - It is NOT supplied as an inference input to the neural forecaster.
    """
    df = pd.read_csv(
        master_path,
        usecols=[
            "timestamp_sensor_UTC6","split","segment_id","core_row_valid",
            "avg_temp","土温","NASA_T2M_C",
            "NASA_ALLSKY_SFC_SW_DWN_W_m2_equiv"
        ]
    )
    df = df[df["core_row_valid"] == 1].copy()
    df["t"] = pd.to_datetime(df["timestamp_sensor_UTC6"])
    df["G"] = df["NASA_ALLSKY_SFC_SW_DWN_W_m2_equiv"] / 1000.0
    df = df.sort_values(["split","segment_id","t"])

    recs = df.to_dict("records")
    times = [r["t"].to_pydatetime() for r in recs]
    Ta = np.asarray([r["avg_temp"] for r in recs], float)
    Ts = np.asarray([r["土温"] for r in recs], float)
    To = np.asarray([r["NASA_T2M_C"] for r in recs], float)
    G  = np.asarray([r["G"] for r in recs], float)

    bykey = defaultdict(list)
    for i,r in enumerate(recs):
        bykey[(r["split"], r["segment_id"])].append(i)

    mass_cfg = config["physics"]["mass_state"]
    tau_air = float(mass_cfg["tau_air_h"])
    tau_out = float(mass_cfg["tau_out_h"])

    Tm = np.empty(len(recs), float)
    for _, inds in bykey.items():
        mass = Ts[inds[0]]
        Tm[inds[0]] = mass
        for k in range(1, len(inds)):
            i0, i1 = inds[k-1], inds[k]
            dh = (times[i1] - times[i0]).total_seconds() / 3600.0
            mass += dh * ((Ta[i0] - mass)/tau_air + (To[i0] - mass)/tau_out)
            Tm[i1] = mass

    coeff = config["physics"]["effective_6h_coefficients"]
    aG, aM, aS = float(coeff["aG"]), float(coeff["aM"]), float(coeff["aS"])

    maps = {"train":{}, "validation":{}, "test":{}}
    endpoint_maps = {"train":{}, "validation":{}, "test":{}}

    for (sp, sid), inds in bykey.items():
        ts = [times[i] for i in inds]
        for li, i in enumerate(inds):
            desired = times[i] + dt.timedelta(hours=6)
            j = bisect.bisect_left(ts, desired, lo=li+1)
            cand = []
            for q in (j-1,j):
                if li < q < len(inds):
                    err = abs((ts[q]-desired).total_seconds())/60.0
                    if err <= 7:
                        cand.append((err,q))
            if not cand:
                continue
            _, q = min(cand)
            jg = inds[q]
            block = inds[li:q]
            gbar = float(np.mean(G[block]))
            tsbar = float(np.mean(Ts[block]))
            rate = aG*gbar + aM*(Tm[i]-Ta[i]) + aS*(tsbar-Ta[i])
            maps[sp][times[i]] = float(rate)
            endpoint_maps[sp][times[i]] = {
                "origin_temp": float(Ta[i]),
                "target_temp": float(Ta[jg]),
                "physics_rate": float(rate)
            }

    return maps, endpoint_maps

def make_physics_arrays(meta, qmap):
    q = np.asarray([qmap[m["origin"]] for m in meta], np.float32)
    t0 = np.asarray([m["current_temp"] for m in meta], np.float32)
    return q, t0

def flatten_grads(grads, params):
    return torch.cat([
        (torch.zeros_like(p) if g is None else g).reshape(-1)
        for g,p in zip(grads,params)
    ])

def loss_parts(model, xb, yb, qb, t0b, tmin, tmax, q_std):
    pred = model(xb)
    data_loss = nn.functional.mse_loss(pred, yb)
    T6 = pred[:,2]*(tmax-tmin) + tmin
    qpred = (T6 - t0b)/6.0
    physics_loss = torch.mean(((qpred-qb)/q_std)**2)
    return pred, data_loss, physics_loss

def gradient_stats(model, diag_loader, tmin, tmax, q_std):
    params = [p for p in model.parameters() if p.requires_grad]
    cosines=[]; ratios=[]
    model.eval()
    for xb,yb,qb,tb in diag_loader:
        _, ld, lp = loss_parts(model,xb,yb,qb,tb,tmin,tmax,q_std)
        gd = torch.autograd.grad(ld,params,retain_graph=True,allow_unused=True)
        gp = torch.autograd.grad(lp,params,allow_unused=True)
        a = flatten_grads(gd,params)
        b = flatten_grads(gp,params)
        na = torch.linalg.vector_norm(a)
        nb = torch.linalg.vector_norm(b)
        cos = torch.dot(a,b)/(na*nb + 1e-12)
        cosines.append(float(cos))
        ratios.append(float(nb/(na+1e-12)))
    c=np.asarray(cosines)
    return {
        "mean_cos":float(c.mean()),
        "median_cos":float(np.median(c)),
        "negative_frac":float(np.mean(c<0)),
        "strong_negative_frac":float(np.mean(c<-0.25)),
        "median_phys_to_data_norm":float(np.median(ratios)),
        "p10_cos":float(np.percentile(c,10)),
        "p90_cos":float(np.percentile(c,90)),
    }

def train_pure_lstm(Xtr,ytr,Xv,yv,seed,batch,max_epochs,patience,lr,grad_clip):
    set_seed(seed)
    model=LSTMModel()
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    tr=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),
                  batch_size=batch,shuffle=True)
    va=DataLoader(TensorDataset(torch.from_numpy(Xv),torch.from_numpy(yv)),
                  batch_size=batch,shuffle=False)
    best=None; best_val=float("inf"); bad=0
    hist=[]
    for ep in range(1,max_epochs+1):
        model.train(); ss=n=0
        for xb,yb in tr:
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(model(xb),yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            opt.step()
            ss += loss.item()*len(xb); n += len(xb)
        model.eval(); vs=vn=0
        with torch.no_grad():
            for xb,yb in va:
                lv=nn.functional.mse_loss(model(xb),yb)
                vs += lv.item()*len(xb); vn += len(xb)
        val=vs/vn
        hist.append([ep,ss/n,val])
        if val<best_val-1e-6:
            best_val=val; best=copy.deepcopy(model.state_dict()); bad=0
        else:
            bad += 1
            if bad>=patience:
                break
    model.load_state_dict(best)
    return model,best_val,hist

def train_fixed_physics(
    Xtr,ytr,qtr,t0tr,Xv,yv,tmin,tmax,q_std,
    seed,batch,max_epochs,patience,lr,grad_clip,lam,diag_loader,diagnostic_epochs
):
    set_seed(seed)
    model=LSTMModel()
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    tr=DataLoader(
        TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr),
                      torch.from_numpy(qtr),torch.from_numpy(t0tr)),
        batch_size=batch,shuffle=True
    )
    va=DataLoader(TensorDataset(torch.from_numpy(Xv),torch.from_numpy(yv)),
                  batch_size=batch,shuffle=False)

    best=None; best_val=float("inf"); bad=0
    diagnostic_epochs=set(diagnostic_epochs)
    grad_history=[]
    if 0 in diagnostic_epochs:
        grad_history.append({"epoch":0, **gradient_stats(model,diag_loader,tmin,tmax,q_std)})
    train_history=[]

    for ep in range(1,max_epochs+1):
        model.train(); ss=sn=0
        for xb,yb,qb,tb in tr:
            opt.zero_grad(set_to_none=True)
            _,ld,lp=loss_parts(model,xb,yb,qb,tb,tmin,tmax,q_std)
            total=ld+lam*lp
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            opt.step()
            ss += total.detach().item()*len(xb); sn += len(xb)

        model.eval(); vs=vn=0
        with torch.no_grad():
            for xb,yb in va:
                lv=nn.functional.mse_loss(model(xb),yb)
                vs += lv.item()*len(xb); vn += len(xb)
        val=vs/vn
        train_history.append([ep,ss/sn,val])
        if ep in diagnostic_epochs:
            grad_history.append({"epoch":ep, **gradient_stats(model,diag_loader,tmin,tmax,q_std)})

        if val<best_val-1e-6:
            best_val=val; best=copy.deepcopy(model.state_dict()); bad=0
        else:
            bad += 1
            if bad>=patience:
                break

    model.load_state_dict(best)
    return model,best_val,train_history,grad_history

def observation_matrix(meta):
    obs=np.zeros((len(meta),8),float)
    for i,m in enumerate(meta):
        for hi,tr in enumerate(m["target_rows"]):
            obs[i,2*hi]=tr["temp_raw"]
            obs[i,2*hi+1]=tr["hum_raw"]
    return obs

def model_metrics(name,pred,obs):
    rows=[]
    for hi,h in enumerate(HORIZONS):
        for var,off,unit in [("Temperature",0,"degC"),("Humidity",1,"%RH")]:
            y=obs[:,2*hi+off]; p=pred[:,2*hi+off]
            rows.append([
                name,h,var,unit,len(y),
                math.sqrt(mean_squared_error(y,p)),
                mean_absolute_error(y,p),r2_score(y,p)
            ])
    return rows

def day_block_delta(pred_a,pred_b,obs,meta,n_boot=700,seed=20260817):
    days=np.asarray([m["origin"].date().isoformat() for m in meta])
    unique=np.unique(days); idxmap={d:np.where(days==d)[0] for d in unique}
    rng=np.random.default_rng(seed)
    out=[]
    for hi,h in enumerate(HORIZONS):
        for var,off in [("Temperature",0),("Humidity",1)]:
            y=obs[:,2*hi+off]; a=pred_a[:,2*hi+off]; b=pred_b[:,2*hi+off]
            point=math.sqrt(mean_squared_error(y,a))-math.sqrt(mean_squared_error(y,b))
            vals=[]
            for _ in range(n_boot):
                ds=rng.choice(unique,size=len(unique),replace=True)
                idx=np.concatenate([idxmap[d] for d in ds])
                vals.append(
                    math.sqrt(mean_squared_error(y[idx],a[idx]))-
                    math.sqrt(mean_squared_error(y[idx],b[idx]))
                )
            lo,med,hi_ci=np.percentile(vals,[2.5,50,97.5])
            out.append([h,var,point,lo,med,hi_ci])
    return out

def physics_consistency(pred,meta,qmap):
    mask=np.asarray([m["origin"] in qmap for m in meta])
    qtrue=np.asarray([qmap[m["origin"]] for m in meta if m["origin"] in qmap],float)
    t0=np.asarray([m["current_temp"] for m in meta if m["origin"] in qmap],float)
    qpred=(pred[mask,2]-t0)/6.0
    return mask,qtrue,qpred,t0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=20260817)
    ap.add_argument("--max-epochs",type=int,default=14)
    ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--patience",type=int,default=4)
    ap.add_argument("--bootstrap",type=int,default=700)
    ap.add_argument("--diagnostic-samples",type=int,default=4096)
    ap.add_argument(
        "--diagnostic-epochs", type=str, default="0,1,3,6,9,12,14",
        help="Comma-separated epochs for gradient diagnostics; epoch 0 means pre-training."
    )
    args=ap.parse_args()
    diagnostic_epochs=[int(x) for x in args.diagnostic_epochs.split(",") if x.strip()]
    if args.max_epochs not in diagnostic_epochs:
        diagnostic_epochs.append(args.max_epochs)

    cfg=json.loads((ROOT/"config/protocol_v3.json").read_text())
    data=ROOT/"data/greenhouse_model_normalized_v3_correct_nasa.csv"
    master=ROOT/"data/greenhouse_model_master_v3_correct_nasa.csv"
    params=ROOT/"data/normalization_parameters_v3_correct_nasa.csv"
    outdir=ROOT/"results"; outdir.mkdir(exist_ok=True)

    rows=load_sequence_rows(data)
    Xtr,ytr,mtr=make_examples(rows,"train",2)
    Xv,yv,mv=make_examples(rows,"validation",1)
    Xt,yt,mt=make_examples(rows,"test",1)

    norm=read_normalization_params(params)
    tmin,tmax=norm["avg_temp"]; hmin,hmax=norm["avg_hum"]

    qmaps,_=build_physics_targets(master,cfg)
    qtr,t0tr=make_physics_arrays(mtr,qmaps["train"])
    q_std=float(np.std(qtr))

    n_diag=min(args.diagnostic_samples,len(Xtr))
    diag_idx=np.linspace(0,len(Xtr)-1,n_diag,dtype=int)
    diag_loader=DataLoader(
        TensorDataset(
            torch.from_numpy(Xtr[diag_idx]),torch.from_numpy(ytr[diag_idx]),
            torch.from_numpy(qtr[diag_idx]),torch.from_numpy(t0tr[diag_idx])
        ),
        batch_size=256,shuffle=False
    )

    tc=cfg["training"]
    pure,pure_val,pure_hist=train_pure_lstm(
        Xtr,ytr,Xv,yv,args.seed,args.batch_size,args.max_epochs,args.patience,
        tc["lr"],tc["grad_clip"]
    )
    fixed,fixed_val,fixed_hist,grad_hist=train_fixed_physics(
        Xtr,ytr,qtr,t0tr,Xv,yv,tmin,tmax,q_std,
        args.seed,args.batch_size,args.max_epochs,args.patience,
        tc["lr"],tc["grad_clip"],cfg["physics"]["lambda_fixed_physics"],diag_loader,
        diagnostic_epochs
    )

    pred_pure=predict_raw(pure,Xt,tmin,tmax,hmin,hmax)
    pred_fixed=predict_raw(fixed,Xt,tmin,tmax,hmin,hmax)
    obs=observation_matrix(mt)

    metric_rows=model_metrics("LSTM",pred_pure,obs)+model_metrics("Fixed physics LSTM",pred_fixed,obs)
    deltas=day_block_delta(pred_fixed,pred_pure,obs,mt,args.bootstrap,args.seed)

    mask,qtrue,q_pure,t0=physics_consistency(pred_pure,mt,qmaps["test"])
    _,_,q_fixed,_=physics_consistency(pred_fixed,mt,qmaps["test"])
    phys_pure=math.sqrt(mean_squared_error(qtrue,q_pure))
    phys_fixed=math.sqrt(mean_squared_error(qtrue,q_fixed))

    # paired day-block bootstrap for physics-consistency delta
    p_days=np.asarray([mt[i]["origin"].date().isoformat() for i in np.where(mask)[0]])
    unique=np.unique(p_days); idxmap={d:np.where(p_days==d)[0] for d in unique}
    rng=np.random.default_rng(args.seed+1)
    vals=[]
    for _ in range(args.bootstrap):
        ds=rng.choice(unique,size=len(unique),replace=True)
        idx=np.concatenate([idxmap[d] for d in ds])
        vals.append(
            math.sqrt(mean_squared_error(qtrue[idx],q_fixed[idx]))-
            math.sqrt(mean_squared_error(qtrue[idx],q_pure[idx]))
        )
    phys_lo,phys_med,phys_hi=np.percentile(vals,[2.5,50,97.5])

    with (outdir/"fixed_physics_prediction_metrics_reproduced.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["model","horizon_h","variable","unit","N","RMSE","MAE","R2"])
        w.writerows(metric_rows)

    with (outdir/"fixed_physics_bootstrap_reproduced.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["horizon_h","variable","delta_RMSE_fixed_minus_lstm","CI95_low","bootstrap_median","CI95_high"])
        w.writerows(deltas)

    with (outdir/"gradient_conflict_diagnostics_reproduced.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=["epoch","mean_cos","median_cos","negative_frac","strong_negative_frac",
                "median_phys_to_data_norm","p10_cos","p90_cos"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(grad_hist)

    with (outdir/"physics_consistency_reproduced.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["model","RMSE_C_per_h","R2"])
        w.writerow(["LSTM",phys_pure,r2_score(qtrue,q_pure)])
        w.writerow(["Fixed physics LSTM",phys_fixed,r2_score(qtrue,q_fixed)])
        w.writerow(["Fixed-LSTM delta",phys_fixed-phys_pure,""])
        w.writerow(["CI95",phys_lo,phys_hi])

    torch.save({"state_dict":pure.state_dict(),"seed":args.seed,"best_val_MSE":pure_val},
               outdir/"lstm_reproduced.pt")
    torch.save({"state_dict":fixed.state_dict(),"seed":args.seed,"best_val_MSE":fixed_val,
                "lambda_phys":cfg["physics"]["lambda_fixed_physics"]},
               outdir/"fixed_physics_lstm_reproduced.pt")

    t6_pure=next(r for r in metric_rows if r[0]=="LSTM" and r[1]==6 and r[2]=="Temperature")
    t6_fixed=next(r for r in metric_rows if r[0]=="Fixed physics LSTM" and r[1]==6 and r[2]=="Temperature")
    final_grad=grad_hist[-1] if grad_hist else {
        "median_cos":float("nan"),"negative_frac":float("nan"),
        "median_phys_to_data_norm":float("nan")
    }

    print("Dataset:",Xtr.shape,Xv.shape,Xt.shape)
    print(f"6 h T RMSE: LSTM={t6_pure[5]:.6f}, Fixed={t6_fixed[5]:.6f}, delta={t6_fixed[5]-t6_pure[5]:+.6f}")
    print(f"Physics RMSE: LSTM={phys_pure:.6f}, Fixed={phys_fixed:.6f}, delta={phys_fixed-phys_pure:+.6f}")
    print(f"Physics delta 95% day-block CI: [{phys_lo:+.6f}, {phys_hi:+.6f}]")
    print(f"Final gradient median cosine={final_grad['median_cos']:+.6f}, "
          f"negative fraction={final_grad['negative_frac']:.4f}, "
          f"phys/data norm ratio={final_grad['median_phys_to_data_norm']:.3f}x")

if __name__=="__main__":
    main()
