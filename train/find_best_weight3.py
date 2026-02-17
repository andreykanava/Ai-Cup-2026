import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

DATA_DIR = "../data/processed"
TARGET_COL = "bird_group"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

EPS = 1e-15


def load_y_as_required_indices():
    y_raw = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")[TARGET_COL].astype(str).values
    missing = sorted(set(y_raw) - set(REQUIRED))
    if missing:
        raise ValueError(f"Train labels contain unknown classes not in REQUIRED: {missing}")
    label_to_idx = {c: i for i, c in enumerate(REQUIRED)}
    return np.array([label_to_idx[c] for c in y_raw], dtype=np.int64)


def align_proba_to_required(proba: np.ndarray, mapping_csv: str) -> np.ndarray:
    classes = pd.read_csv(mapping_csv)["label"].astype(str).tolist()
    df = pd.DataFrame(proba, columns=classes)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{mapping_csv}: missing columns {missing}")

    p = df[REQUIRED].to_numpy(dtype=np.float64)

    # safety: normalize
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def normalize_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def main():
    y = load_y_as_required_indices()
    labels = np.arange(len(REQUIRED))

    # ----- LOAD -----
    oof_cat_raw = np.load("out/result11(536)/result_cat_temp/oof_proba_cat_temp.npy")
    oof_lgb_raw = np.load("out/result11(536)/oof_proba_lgbm_weighted.npy")
    oof_third_raw = np.load("out/result12(5266) - overfit/oof_proba_cat_temp.npy")  # <-- поменяй имя

    oof_cat = align_proba_to_required(oof_cat_raw, "out/result11(536)/result_cat_temp/label_mapping_cat_temp.csv")
    oof_lgb = align_proba_to_required(oof_lgb_raw, "out/result10(537)/label_mapping_lgbm.csv")
    oof_third = align_proba_to_required(oof_third_raw, "out/result12(5266) - overfit/label_mapping_cat_temp.csv")  # <-- поменяй имя

    # ----- SINGLE MODEL SCORES -----
    ll_cat = log_loss(y, oof_cat, labels=labels)
    ll_lgb = log_loss(y, oof_lgb, labels=labels)
    ll_third = log_loss(y, oof_third, labels=labels)

    print("OOF logloss:")
    print("  CatBoost :", ll_cat)
    print("  LightGBM :", ll_lgb)
    print("  RF    :", ll_third)
    print()

    # ----- GRID SEARCH WEIGHTS (w1+w2+w3=1) -----
    # step=0.01 => ~5151 комбинаций; step=0.001 => ~500k (дольше)
    step = 0.01

    best = {
        "w_cat": None,
        "w_lgb": None,
        "w_third": None,
        "logloss": float("inf"),
    }

    ws = np.arange(0.0, 1.0 + 1e-12, step)

    for w_cat in ws:
        for w_lgb in ws:
            w_third = 1.0 - w_cat - w_lgb
            if w_third < -1e-12:
                continue
            if w_third < 0:
                w_third = 0.0

            p = w_cat * oof_cat + w_lgb * oof_lgb + w_third * oof_third
            p = normalize_rows(p)

            ll = log_loss(y, p, labels=labels)
            if ll < best["logloss"]:
                best.update(w_cat=float(w_cat), w_lgb=float(w_lgb), w_third=float(w_third), logloss=float(ll))

    print("BEST RESULT:")
    print("  weight_cat  =", round(best["w_cat"], 3))
    print("  weight_lgb  =", round(best["w_lgb"], 3))
    print("  weight_rf=", round(best["w_third"], 3))
    print("  logloss     =", best["logloss"])


if __name__ == "__main__":
    main()
