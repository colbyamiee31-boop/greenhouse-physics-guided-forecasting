
from __future__ import annotations
import bisect, csv, datetime as dt, math, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

INPUT_ENV_COLS = [
    "avg_temp_norm","avg_hum_norm","土温_norm","土湿_norm",
    "greenhouse_light_mean_norm","NASA_T2M_C_norm","NASA_WS2M_m_s_norm",
    "NASA_ALLSKY_SFC_SW_DWN_W_m2_equiv_norm","NASA_RH2M_pct_norm",
    "NASA_T2MDEW_C_norm"
]
HORIZONS=[1,6,24,72]
SEQ_LEN=72

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def read_normalization_params(path):
    out={}
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            out[r["variable"]]=(float(r["train_min"]),float(r["train_max"]))
    return out

def load_sequence_rows(path):
    rows=[]
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            if r["core_row_valid"]!="1": continue
            t=dt.datetime.strptime(r["timestamp_sensor_UTC6"],"%Y-%m-%d %H:%M:%S")
            vals=[float(r[c]) for c in INPUT_ENV_COLS]
            dmin=float(r["dt_prev_min"]) if r["dt_prev_min"] else 10.0
            dmin=min(15.0,max(5.0,dmin))
            hour=t.hour+t.minute/60.0
            doy=t.timetuple().tm_yday
            feats=vals+[
                (dmin-5.0)/10.0,
                math.sin(2*math.pi*hour/24),math.cos(2*math.pi*hour/24),
                math.sin(2*math.pi*doy/365.25),math.cos(2*math.pi*doy/365.25)
            ]
            rows.append({
                "t":t,"split":r["split"],"seg":int(r["segment_id"]),
                "x":np.asarray(feats,np.float32),
                "temp_raw":float(r["avg_temp"]),"hum_raw":float(r["avg_hum"]),
                "temp_norm":float(r["avg_temp_norm"]),"hum_norm":float(r["avg_hum_norm"])
            })
    return rows

def make_examples(rows, split, stride):
    segs=defaultdict(list)
    for r in rows:
        segs[(r["split"],r["seg"])].append(r)
    for k in segs:
        segs[k].sort(key=lambda z:z["t"])
    X=[];Y=[];M=[]
    for (sp,sid),rs in segs.items():
        if sp!=split or len(rs)<SEQ_LEN+2: continue
        ts=[r["t"] for r in rs]
        for i in range(SEQ_LEN-1,len(rs),stride):
            origin=rs[i]; y=[]; target_rows=[]; ok=True
            for h in HORIZONS:
                desired=origin["t"]+dt.timedelta(hours=h)
                j=bisect.bisect_left(ts,desired,lo=i+1)
                cand=[]
                for q in (j-1,j):
                    if i<q<len(rs):
                        err=abs((ts[q]-desired).total_seconds())/60
                        if err<=7: cand.append((err,q))
                if not cand:
                    ok=False; break
                _,q=min(cand)
                tr=rs[q]
                y += [tr["temp_norm"],tr["hum_norm"]]
                target_rows.append(tr)
            if not ok: continue
            X.append(np.stack([r["x"] for r in rs[i-SEQ_LEN+1:i+1]]))
            Y.append(np.asarray(y,np.float32))
            M.append({"origin":origin["t"],"current_temp":origin["temp_raw"],
                      "current_hum":origin["hum_raw"],"target_rows":target_rows})
    return np.stack(X),np.stack(Y),M

class LSTMModel(nn.Module):
    def __init__(self,input_size=15,hidden=24):
        super().__init__()
        self.rnn=nn.LSTM(input_size,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden,24),nn.ReLU(),nn.Linear(24,8))
    def forward(self,x):
        o,_=self.rnn(x); return self.head(o[:,-1,:])

class GRUModel(nn.Module):
    def __init__(self,input_size=15,hidden=24):
        super().__init__()
        self.rnn=nn.GRU(input_size,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden,24),nn.ReLU(),nn.Linear(24,8))
    def forward(self,x):
        o,_=self.rnn(x); return self.head(o[:,-1,:])

class CausalBlock(nn.Module):
    def __init__(self,inc,outc,k=3,d=1):
        super().__init__()
        self.pad=(k-1)*d
        self.conv=nn.Conv1d(inc,outc,k,dilation=d,padding=self.pad)
        self.act=nn.ReLU()
    def forward(self,x):
        y=self.conv(x)
        if self.pad: y=y[:,:,:-self.pad]
        return self.act(y)

class TCNModel(nn.Module):
    def __init__(self,input_size=15,ch=24):
        super().__init__()
        self.net=nn.Sequential(
            CausalBlock(input_size,ch,3,1),
            CausalBlock(ch,ch,3,2),
            CausalBlock(ch,ch,3,4)
        )
        self.head=nn.Sequential(nn.Linear(ch,24),nn.ReLU(),nn.Linear(24,8))
    def forward(self,x):
        y=self.net(x.transpose(1,2)); return self.head(y[:,:,-1])

class TransformerModel(nn.Module):
    def __init__(self,input_size=15,d_model=16,nhead=4,ff=32,max_len=72):
        super().__init__()
        self.input_proj=nn.Linear(input_size,d_model)
        self.pos=nn.Parameter(torch.zeros(1,max_len,d_model))
        layer=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=nhead,dim_feedforward=ff,dropout=0.0,
            batch_first=True,activation="relu",norm_first=True
        )
        self.encoder=nn.TransformerEncoder(layer,num_layers=1)
        self.head=nn.Sequential(nn.Linear(d_model,24),nn.ReLU(),nn.Linear(24,8))
        nn.init.normal_(self.pos,std=0.02)
    def forward(self,x):
        z=self.input_proj(x)+self.pos[:,:x.size(1),:]
        z=self.encoder(z); return self.head(z[:,-1,:])

def train_supervised(model,Xtr,ytr,Xv,yv,seed=20260817,batch_size=256,
                     max_epochs=14,patience=4,lr=1e-3,grad_clip=1.0):
    import copy
    set_seed(seed)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    gen=torch.Generator().manual_seed(seed)
    tr=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),
                  batch_size=batch_size,shuffle=True,generator=gen)
    va=DataLoader(TensorDataset(torch.from_numpy(Xv),torch.from_numpy(yv)),
                  batch_size=batch_size,shuffle=False)
    mse=nn.MSELoss()
    best=None; best_val=float("inf"); bad=0
    for _ in range(max_epochs):
        model.train()
        for xb,yb in tr:
            opt.zero_grad(set_to_none=True)
            loss=mse(model(xb),yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            opt.step()
        model.eval(); ss=n=0
        with torch.no_grad():
            for xb,yb in va:
                v=mse(model(xb),yb); ss+=v.item()*len(xb); n+=len(xb)
        val=ss/n
        if val<best_val-1e-6:
            best_val=val; best=copy.deepcopy(model.state_dict()); bad=0
        else:
            bad+=1
            if bad>=patience: break
    model.load_state_dict(best)
    return model,best_val

def predict_raw(model,X,tmin,tmax,hmin,hmax,batch=512):
    parts=[]; model.eval()
    with torch.no_grad():
        for i in range(0,len(X),batch):
            parts.append(model(torch.from_numpy(X[i:i+batch])).numpy())
    pn=np.concatenate(parts); pr=np.zeros_like(pn,dtype=float)
    for hi in range(4):
        pr[:,2*hi]=pn[:,2*hi]*(tmax-tmin)+tmin
        pr[:,2*hi+1]=pn[:,2*hi+1]*(hmax-hmin)+hmin
    return pr
