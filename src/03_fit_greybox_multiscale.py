
from pathlib import Path
import bisect, csv, datetime as dt, json, math
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/"config/protocol_v3.json").read_text())
df=pd.read_csv(ROOT/"data/greenhouse_model_master_v3_correct_nasa.csv",
    usecols=["timestamp_sensor_UTC6","split","segment_id","core_row_valid",
             "avg_temp","土温","NASA_T2M_C","NASA_ALLSKY_SFC_SW_DWN_W_m2_equiv"])
df=df[df.core_row_valid==1].copy()
df["t"]=pd.to_datetime(df.timestamp_sensor_UTC6)
df["G"]=df.NASA_ALLSKY_SFC_SW_DWN_W_m2_equiv/1000.0
df=df.sort_values(["split","segment_id","t"])
recs=df.to_dict("records")
times=[r["t"].to_pydatetime() for r in recs]
Ta=np.array([r["avg_temp"] for r in recs]); Ts=np.array([r["土温"] for r in recs])
To=np.array([r["NASA_T2M_C"] for r in recs]); G=np.array([r["G"] for r in recs])

bykey=defaultdict(list)
for i,r in enumerate(recs): bykey[(r["split"],r["segment_id"])].append(i)

tauA=cfg["physics"]["mass_state"]["tau_air_h"]; tauO=cfg["physics"]["mass_state"]["tau_out_h"]
Tm=np.empty(len(recs))
for _,inds in bykey.items():
    mass=Ts[inds[0]]; Tm[inds[0]]=mass
    for k in range(1,len(inds)):
        i0,i1=inds[k-1],inds[k]; dh=(times[i1]-times[i0]).total_seconds()/3600
        mass += dh*((Ta[i0]-mass)/tauA+(To[i0]-mass)/tauO); Tm[i1]=mass

def transitions(hours,split):
    out=[]
    for (sp,sid),inds in bykey.items():
        if sp!=split: continue
        ts=[times[i] for i in inds]
        for li,i in enumerate(inds):
            desired=times[i]+dt.timedelta(hours=hours)
            j=bisect.bisect_left(ts,desired,lo=li+1); cand=[]
            for q in (j-1,j):
                if li<q<len(inds):
                    err=abs((ts[q]-desired).total_seconds())/60
                    if err<=7: cand.append((err,q))
            if not cand: continue
            _,q=min(cand); jg=inds[q]; block=inds[li:q]
            out.append((i,jg,(times[jg]-times[i]).total_seconds()/3600,
                        float(np.mean(G[block])),float(np.mean(Ts[block]))))
    return out

rows=[]
for h in cfg["physics"]["timescales_h"]:
    tr=transitions(h,"train")
    X=np.asarray([[g,Tm[i]-Ta[i],ts-Ta[i]] for i,j,dh,g,ts in tr])
    y=np.asarray([(Ta[j]-Ta[i])/dh for i,j,dh,g,ts in tr])
    b,_=nnls(X,y)
    for sp in ["train","validation","test"]:
        rr=transitions(h,sp)
        Xs=np.asarray([[g,Tm[i]-Ta[i],ts-Ta[i]] for i,j,dh,g,ts in rr])
        ys=np.asarray([(Ta[j]-Ta[i])/dh for i,j,dh,g,ts in rr])
        p=Xs@b
        rows.append([h,sp,len(ys),*b,math.sqrt(mean_squared_error(ys,p)),r2_score(ys,p)])

with (ROOT/"results/greybox_multiscale_reproduced.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f); w.writerow(["physics_scale_h","split","N","aG","aM","aS","dTdt_RMSE","dTdt_R2"]); w.writerows(rows)
