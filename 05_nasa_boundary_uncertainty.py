
from __future__ import annotations
import argparse, bisect, csv, datetime as dt, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score

from common import (
    HORIZONS, INPUT_ENV_COLS, LSTMModel, load_sequence_rows, make_examples,
    predict_raw, read_normalization_params
)

ROOT=Path(__file__).resolve().parents[1]

SCENARIOS=[
    ("Reference",0,1,1),
    ("T2M -2C",-2,1,1),("T2M -1C",-1,1,1),("T2M +1C",1,1,1),("T2M +2C",2,1,1),
    ("Solar -20%",0,1,0.8),("Solar -10%",0,1,0.9),("Solar +10%",0,1,1.1),("Solar +20%",0,1,1.2),
    ("Wind -50%",0,0.5,1),("Wind -20%",0,0.8,1),("Wind +20%",0,1.2,1),("Wind +50%",0,1.5,1),
    ("Combined low",-1,0.8,0.9),("Combined high",1,1.2,1.1)
]

def obs_matrix(meta):
    obs=np.zeros((len(meta),8),float)
    for i,m in enumerate(meta):
        for hi,tr in enumerate(m["target_rows"]):
            obs[i,2*hi]=tr["temp_raw"]; obs[i,2*hi+1]=tr["hum_raw"]
    return obs

def load_model(path):
    m=LSTMModel()
    ck=torch.load(path,map_location="cpu")
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m

def perturb_inputs(X,norm,tbias=0,wscale=1,gscale=1):
    Y=X.copy()
    index={
        "T":INPUT_ENV_COLS.index("NASA_T2M_C_norm"),
        "W":INPUT_ENV_COLS.index("NASA_WS2M_m_s_norm"),
        "G":INPUT_ENV_COLS.index("NASA_ALLSKY_SFC_SW_DWN_W_m2_equiv_norm")
    }
    ranges={
        "T":norm["NASA_T2M_C"],
        "W":norm["NASA_WS2M_m_s"],
        "G":norm["NASA_ALLSKY_SFC_SW_DWN_W_m2_equiv"]
    }
    for key,add,mul in [("T",tbias,1),("W",0,wscale),("G",0,gscale)]:
        idx=index[key]; lo,hi=ranges[key]
        raw=Y[:,:,idx]*(hi-lo)+lo
        raw=(raw+add)*mul
        if key in ("W","G"): raw=np.maximum(raw,0)
        Y[:,:,idx]=(raw-lo)/(hi-lo)
    return Y.astype(np.float32)

def predictor_stress(models,Xtest,obs,meta,norm,tmin,tmax,hmin,hmax,n_boot,seed):
    days=np.asarray([m["origin"].date().isoformat() for m in meta])
    unique=np.unique(days); idxmap={d:np.where(days==d)[0] for d in unique}
    rng=np.random.default_rng(seed)

    ref={}
    for name,m in models.items():
        ref[name]=predict_raw(m,Xtest,tmin,tmax,hmin,hmax)

    rows=[]
    for name,m in models.items():
        reference=ref[name]
        for sc,tb,ws,gs in SCENARIOS:
            p=reference if sc=="Reference" else predict_raw(m,perturb_inputs(Xtest,norm,tb,ws,gs),tmin,tmax,hmin,hmax)
            for hi,h in enumerate(HORIZONS):
                for var,off in [("Temperature",0),("Humidity",1)]:
                    y=obs[:,2*hi+off]; yp=p[:,2*hi+off]; rp=reference[:,2*hi+off]
                    rm=math.sqrt(mean_squared_error(y,yp))
                    r0=math.sqrt(mean_squared_error(y,rp))
                    shift=float(np.mean(np.abs(yp-rp)))
                    if sc=="Reference":
                        lo=med=high=0.0
                    else:
                        vals=[]
                        for _ in range(n_boot):
                            ds=rng.choice(unique,size=len(unique),replace=True)
                            idx=np.concatenate([idxmap[d] for d in ds])
                            vals.append(
                                math.sqrt(mean_squared_error(y[idx],yp[idx]))-
                                math.sqrt(mean_squared_error(y[idx],rp[idx]))
                            )
                        lo,med,high=np.percentile(vals,[2.5,50,97.5])
                    rows.append([name,sc,tb,ws,gs,h,var,rm,rm-r0,shift,lo,med,high])
    return rows

def prepare_greybox(master_path,cfg):
    df=pd.read_csv(
        master_path,
        usecols=["timestamp_sensor_UTC6","split","segment_id","core_row_valid",
                 "avg_temp","土温","NASA_T2M_C","NASA_WS2M_m_s",
                 "NASA_ALLSKY_SFC_SW_DWN_W_m2_equiv"]
    )
    df=df[df["core_row_valid"]==1].copy()
    df["t"]=pd.to_datetime(df["timestamp_sensor_UTC6"])
    df["G"]=df["NASA_ALLSKY_SFC_SW_DWN_W_m2_equiv"]/1000.0
    df=df.sort_values(["split","segment_id","t"])
    recs=df.to_dict("records")
    times=[r["t"].to_pydatetime() for r in recs]
    Ta=np.asarray([r["avg_temp"] for r in recs],float)
    Ts=np.asarray([r["土温"] for r in recs],float)
    To=np.asarray([r["NASA_T2M_C"] for r in recs],float)
    W=np.asarray([r["NASA_WS2M_m_s"] for r in recs],float)
    G=np.asarray([r["G"] for r in recs],float)
    bykey=defaultdict(list)
    for i,r in enumerate(recs): bykey[(r["split"],r["segment_id"])].append(i)

    T6=[]
    for (sp,sid),inds in bykey.items():
        if sp!="test": continue
        ts=[times[i] for i in inds]
        for li,i in enumerate(inds):
            desired=times[i]+dt.timedelta(hours=6)
            j=bisect.bisect_left(ts,desired,lo=li+1)
            cand=[]
            for q in (j-1,j):
                if li<q<len(inds):
                    err=abs((ts[q]-desired).total_seconds())/60
                    if err<=7: cand.append((err,q))
            if not cand: continue
            _,q=min(cand); jg=inds[q]
            block=inds[li:q]
            T6.append((i,jg,(times[jg]-times[i]).total_seconds()/3600,
                       float(np.mean(G[block])),float(np.mean(Ts[block])),
                       float(np.mean(To[block])),float(np.mean(W[block]))))
    return recs,times,Ta,Ts,To,W,G,bykey,T6

def tm_for_test(times,Ta,Ts,To,bykey,tbias,tau_air,tau_out):
    out={}
    for (sp,sid),inds in bykey.items():
        if sp!="test": continue
        mass=Ts[inds[0]]
        out[times[inds[0]]]=mass
        for k in range(1,len(inds)):
            i0,i1=inds[k-1],inds[k]
            dh=(times[i1]-times[i0]).total_seconds()/3600
            mass += dh*((Ta[i0]-mass)/tau_air + ((To[i0]+tbias)-mass)/tau_out)
            out[times[i1]]=mass
    return out

def greybox_stress(master_path,cfg,n_boot,seed):
    recs,times,Ta,Ts,To,W,G,bykey,T6=prepare_greybox(master_path,cfg)
    coeff=cfg["physics"]["effective_6h_coefficients"]
    aG,aM,aS=coeff["aG"],coeff["aM"],coeff["aS"]
    tau_air=cfg["physics"]["mass_state"]["tau_air_h"]
    tau_out=cfg["physics"]["mass_state"]["tau_out_h"]

    def scenario(tb=0,ws=1,gs=1):
        # ws is retained in the scenario definition, but the final reduced grey-box has
        # no direct wind term because outdoor/wind exchange was practically non-identifiable.
        tm=tm_for_test(times,Ta,Ts,To,bykey,tb,tau_air,tau_out)
        origins=[]; y=[]; r=[]; Tobs=[]; Tpred=[]
        for i,j,dh,gbar,tsbar,tobar,wbar in T6:
            rate=aG*(gbar*gs)+aM*(tm[times[i]]-Ta[i])+aS*(tsbar-Ta[i])
            origins.append(times[i]); y.append((Ta[j]-Ta[i])/dh); r.append(rate)
            Tobs.append(Ta[j]); Tpred.append(Ta[i]+rate*dh)
        return np.asarray(origins,dtype=object),np.asarray(y),np.asarray(r),np.asarray(Tobs),np.asarray(Tpred)

    ref_o,ref_y,ref_r,ref_to,ref_tp=scenario()
    ref_rr=math.sqrt(mean_squared_error(ref_y,ref_r))
    ref_tr=math.sqrt(mean_squared_error(ref_to,ref_tp))
    days=np.asarray([x.date().isoformat() for x in ref_o])
    unique=np.unique(days); idxmap={d:np.where(days==d)[0] for d in unique}
    rng=np.random.default_rng(seed)
    rows=[]
    for sc,tb,ws,gs in SCENARIOS:
        o,y,r,to,tp=scenario(tb,ws,gs)
        rr=math.sqrt(mean_squared_error(y,r)); tr=math.sqrt(mean_squared_error(to,tp))
        rshift=float(np.mean(np.abs(r-ref_r))); tshift=float(np.mean(np.abs(tp-ref_tp)))
        if sc=="Reference":
            lo=med=high=0.0
        else:
            vals=[]
            for _ in range(n_boot):
                ds=rng.choice(unique,size=len(unique),replace=True)
                idx=np.concatenate([idxmap[d] for d in ds])
                vals.append(
                    math.sqrt(mean_squared_error(y[idx],r[idx]))-
                    math.sqrt(mean_squared_error(ref_y[idx],ref_r[idx]))
                )
            lo,med,high=np.percentile(vals,[2.5,50,97.5])
        rows.append([sc,tb,ws,gs,len(y),rr,rr-ref_rr,r2_score(y,r),rshift,
                     tr,tr-ref_tr,tshift,lo,med,high])
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lstm-checkpoint",default="results/lstm_reproduced.pt")
    ap.add_argument("--fixed-checkpoint",default="results/fixed_physics_lstm_reproduced.pt")
    ap.add_argument("--bootstrap",type=int,default=300)
    ap.add_argument("--seed",type=int,default=20260817)
    args=ap.parse_args()

    cfg=json.loads((ROOT/"config/protocol_v3.json").read_text())
    norm=read_normalization_params(ROOT/"data/normalization_parameters_v3_correct_nasa.csv")
    tmin,tmax=norm["avg_temp"]; hmin,hmax=norm["avg_hum"]

    seqrows=load_sequence_rows(ROOT/"data/greenhouse_model_normalized_v3_correct_nasa.csv")
    Xtest,ytest,meta=make_examples(seqrows,"test",1)
    obs=obs_matrix(meta)

    models={
        "LSTM":load_model(ROOT/args.lstm_checkpoint),
        "Fixed physics LSTM":load_model(ROOT/args.fixed_checkpoint)
    }
    pred_rows=predictor_stress(models,Xtest,obs,meta,norm,tmin,tmax,hmin,hmax,args.bootstrap,args.seed)
    phys_rows=greybox_stress(ROOT/"data/greenhouse_model_master_v3_correct_nasa.csv",cfg,args.bootstrap,args.seed+11)

    with (ROOT/"results/nasa_boundary_predictor_recomputed.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["model","scenario","T2M_bias_C","wind_scale","solar_scale","horizon_h","variable",
                    "RMSE","delta_RMSE","mean_abs_prediction_shift","CI95_low","median","CI95_high"])
        w.writerows(pred_rows)

    with (ROOT/"results/nasa_boundary_greybox_recomputed.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f)
        w.writerow(["scenario","T2M_bias_C","wind_scale","solar_scale","N","dTdt_RMSE","delta_dTdt_RMSE",
                    "dTdt_R2","mean_abs_rate_shift","endpoint_T_RMSE","delta_endpoint_T_RMSE",
                    "mean_abs_endpoint_T_shift","CI95_low","median","CI95_high"])
        w.writerows(phys_rows)

    print("Predictor stress rows:",len(pred_rows))
    print("Grey-box stress rows:",len(phys_rows))
    for sc in ["T2M -2C","T2M +2C","Solar -20%","Solar +20%","Wind -50%","Wind +50%"]:
        r=next(x for x in phys_rows if x[0]==sc)
        print(sc, f"grey-box mean |endpoint shift|={r[11]:.3f} degC",
              f"delta residual RMSE={r[6]:+.3f} degC/h")
    print("Interpret wind-zero sensitivity as non-identifiability of the direct wind term, not physical irrelevance.")
    print("These are controlled stress tests, not measured NASA bias.")

if __name__=="__main__":
    main()
