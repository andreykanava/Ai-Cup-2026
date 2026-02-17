import numpy as np
import pandas as pd

DATA_DIR = "../data/processed"
ID_COL = "track_id"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese", "Gulls",
    "Birds of Prey", "Waders", "Songbirds"
]

W_CAT = 0.709
W_LGB = 1.0 - W_CAT

test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")

p_cat = np.load("out/result12(5266) - overfit/test_proba_cat_temp.npy")
p_lgb = np.load("out/result12(5266) - overfit/test_proba_lgbm_weighted.npy")

cls_cat = pd.read_csv("out/result12(5266) - overfit/label_mapping_cat_temp.csv")["label"].astype(str).tolist()
cls_lgb = pd.read_csv("out/result12(5266) - overfit/label_mapping_lgbm.csv")["label"].astype(str).tolist()

df_cat = pd.DataFrame(p_cat, columns=cls_cat)[REQUIRED]
df_lgb = pd.DataFrame(p_lgb, columns=cls_lgb)[REQUIRED]

p = W_CAT * df_cat.values + W_LGB * df_lgb.values

# safety
p = np.clip(p, 1e-15, 1.0)
p = p / p.sum(axis=1, keepdims=True)

sub = pd.DataFrame(p, columns=REQUIRED)
sub.insert(0, ID_COL, test_ids[ID_COL].values)

out = "submission_ens_best.csv"
sub.to_csv(out, index=False)

print("saved", out, "shape", sub.shape, "weights:", W_CAT, W_LGB)
print(sub.head())
