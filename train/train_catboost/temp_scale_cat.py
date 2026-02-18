# train/temp_scale_cat.py
import numpy as np, pandas as pd
from sklearn.metrics import log_loss

DATA_DIR= "../../data/processed"
TARGET_COL="bird_group"
ID_COL="track_id"

REQUIRED=["Clutter","Cormorants","Pigeons","Ducks","Geese","Gulls","Birds of Prey","Waders","Songbirds"]
EPS=1e-15

OOF="../result/ensemble_many/oof_ens.npy"
TEST="../result/ensemble_many/test_ens.npy"
MAP="../out/result14/catboost/label_mapping_cat.csv"

def align(p, mapping):
    cls=pd.read_csv(mapping)["label"].astype(str).tolist()
    df=pd.DataFrame(p, columns=cls)[REQUIRED]
    p=df.to_numpy(np.float64)
    p=np.clip(p, EPS, 1); p/=p.sum(1, keepdims=True)
    return p

def y_idx():
    y_raw=pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")[TARGET_COL].astype(str).values
    m={c:i for i,c in enumerate(REQUIRED)}
    return np.array([m[c] for c in y_raw], np.int64)

def temp(p,t):
    p=np.clip(p, EPS, 1); p/=p.sum(1, keepdims=True)
    p=p**t
    p=np.clip(p, EPS, 1); p/=p.sum(1, keepdims=True)
    return p

y=y_idx()
oof=align(np.load(OOF), MAP)
test=align(np.load(TEST), MAP)

best=(1e9,None)
for t in np.linspace(0.6,1.4,401):   # шаг 0.002
    ll=log_loss(y, temp(oof,t), labels=np.arange(len(REQUIRED)))
    if ll<best[0]:
        best=(ll,float(t))

print("BEST t:", best[1], "OOF logloss:", best[0])

test_t=temp(test, best[1])
np.save("test_proba_cat_temp.npy", test_t)

test_ids=pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")[[ID_COL]]
sub=pd.concat([test_ids, pd.DataFrame(test_t, columns=REQUIRED)], axis=1)
sub.to_csv("submission_cat_temp.csv", index=False)
print("Saved submission_cat_temp.csv")
