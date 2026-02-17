import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

DATA_DIR = "../data/processed"
TARGET_COL = "bird_group"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

def load_y_as_required_indices():
    y_raw = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")[TARGET_COL].astype(str).values
    missing = sorted(set(y_raw) - set(REQUIRED))
    if missing:
        raise ValueError(f"Train labels contain unknown classes not in REQUIRED: {missing}")
    label_to_idx = {c: i for i, c in enumerate(REQUIRED)}
    y = np.array([label_to_idx[c] for c in y_raw], dtype=np.int64)
    return y

def align_proba_to_required(proba: np.ndarray, mapping_csv: str) -> np.ndarray:
    classes = pd.read_csv(mapping_csv)["label"].astype(str).tolist()
    df = pd.DataFrame(proba, columns=classes)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{mapping_csv}: missing columns {missing}")

    df = df[REQUIRED].copy()  # reorder exactly
    p = df.values.astype(np.float64)

    # safety: normalize just in case
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p

def main():
    y = load_y_as_required_indices()

    oof_cat_raw = np.load("out/result12(5266) - overfit/oof_proba_cat_temp.npy")
    oof_lgb_raw = np.load("out/result12(5266) - overfit/oof_proba_lgbm_weighted.npy")

    oof_cat = align_proba_to_required(oof_cat_raw, "out/result12(5266) - overfit/label_mapping_cat_temp.csv")
    oof_lgb = align_proba_to_required(oof_lgb_raw, "out/result12(5266) - overfit/label_mapping_lgbm.csv")

    ll_cat = log_loss(y, oof_cat, labels=np.arange(len(REQUIRED)))
    ll_lgb = log_loss(y, oof_lgb, labels=np.arange(len(REQUIRED)))
    print("CatBoost OOF logloss:", ll_cat)
    print("LightGBM OOF logloss:", ll_lgb)
    print()

    best_w, best_ll = None, float("inf")

    for w in np.arange(0.0, 1.0001, 0.001):
        p = w * oof_cat + (1.0 - w) * oof_lgb
        p = np.clip(p, 1e-15, 1.0)
        p = p / p.sum(axis=1, keepdims=True)

        ll = log_loss(y, p, labels=np.arange(len(REQUIRED)))
        if ll < best_ll:
            best_ll = ll
            best_w = w

    print("BEST RESULT:")
    print("weight_cat =", round(float(best_w), 3))
    print("weight_lgb =", round(float(1.0 - best_w), 3))
    print("logloss    =", float(best_ll))

if __name__ == "__main__":
    main()
