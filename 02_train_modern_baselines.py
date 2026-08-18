
from pathlib import Path
import csv, math
import numpy as np, torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from common import *

ROOT=Path(__file__).resolve().parents[1]
rows=load_sequence_rows(ROOT/"data/greenhouse_model_normalized_v3_correct_nasa.csv")
Xtr,ytr,mtr=make_examples(rows,"train",2)
Xv,yv,mv=make_examples(rows,"validation",1)
Xt,yt,mt=make_examples(rows,"test",1)

norm=read_normalization_params(ROOT/"data/normalization_parameters_v3_correct_nasa.csv")
tmin,tmax=norm["avg_temp"]; hmin,hmax=norm["avg_hum"]

obs=np.zeros((len(mt),8),float)
for i,m in enumerate(mt):
    for hi,tr in enumerate(m["target_rows"]):
        obs[i,2*hi]=tr["temp_raw"]; obs[i,2*hi+1]=tr["hum_raw"]

models={"LSTM":LSTMModel(),"GRU":GRUModel(),"TCN":TCNModel(),"Transformer":TransformerModel()}
metrics=[]
for name,model in models.items():
    model,bv=train_supervised(model,Xtr,ytr,Xv,yv)
    pred=predict_raw(model,Xt,tmin,tmax,hmin,hmax)
    torch.save({"state_dict":model.state_dict(),"model":name,"best_val_MSE":bv},
               ROOT/"results"/f"{name.lower()}_single_seed.pt")
    for hi,h in enumerate(HORIZONS):
        for var,off,unit in [("Temperature",0,"degC"),("Humidity",1,"%RH")]:
            y=obs[:,2*hi+off]; p=pred[:,2*hi+off]
            metrics.append([name,h,var,unit,len(y),
                            math.sqrt(mean_squared_error(y,p)),
                            mean_absolute_error(y,p),r2_score(y,p)])

with (ROOT/"results/modern_baseline_metrics_reproduced.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f); w.writerow(["model","horizon_h","variable","unit","N","RMSE","MAE","R2"]); w.writerows(metrics)
print("Train/val/test:",Xtr.shape,Xv.shape,Xt.shape)
