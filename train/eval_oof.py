import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "../data/processed"
TARGET_COL = "bird_group"

# load true labels
y_df = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")
y_raw = y_df[TARGET_COL].astype(str).values

le = LabelEncoder()
y = le.fit_transform(y_raw)

# load OOF predictions
oof_cat = np.load("result9(537)/oof_proba_cat.npy")
oof_lgb = np.load("oof_proba_lgbm.npy")

# individual scores
print("CatBoost OOF logloss:", log_loss(y, oof_cat))
print("LightGBM OOF logloss:", log_loss(y, oof_lgb))

# ensemble test
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    oof = w * oof_cat + (1 - w) * oof_lgb
    score = log_loss(y, oof)
    print(f"ensemble w_cat={w:.2f} logloss={score:.5f}")
