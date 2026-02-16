# ensemble_3models.py
import numpy as np
import pandas as pd

DATA_DIR = "../data/processed"
ID_COL = "track_id"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese", "Gulls",
    "Birds of Prey", "Waders", "Songbirds"
]

# ---- weights (set yours) ----
W_CAT = 0.64
W_LGB = 0.29
W_HGB = 0.07  # e.g. 0.05..0.15 if it helps

# optional: auto-normalize weights to sum=1
W_SUM = W_CAT + W_LGB + W_HGB
W_CAT, W_LGB, W_HGB = W_CAT / W_SUM, W_LGB / W_SUM, W_HGB / W_SUM


def load_proba(path_npy: str, path_mapping_csv: str) -> pd.DataFrame:
    p = np.load(path_npy)
    cls = pd.read_csv(path_mapping_csv)["label"].astype(str).tolist()
    df = pd.DataFrame(p, columns=cls)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path_npy}: missing required columns {missing}")

    return df[REQUIRED]


def main():
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")

    # --- load 3 models ---
    df_cat = load_proba(
        "result8(5515)/test_proba_cat.npy",
        "result8(5515)/label_mapping_cat.csv",
    )
    df_lgb = load_proba(
        "result8(5515)/test_proba_lgbm_ens.npy",
        "result8(5515)/label_mapping_lgbm.csv",
    )
    # HGB temperature-scaled output from your script
    # (if you want raw instead, point to test_proba_hgb.npy + its mapping)
    df_hgb = load_proba(
        "result8(5515)/test_proba_hgb_ms_ts.npy",
        "result8(5515)/label_mapping_hgb_ms_ts.csv",
    )

    # --- weighted blend ---
    p = (
        W_CAT * df_cat.values
        + W_LGB * df_lgb.values
        + W_HGB * df_hgb.values
    )

    # safety: clip + renorm
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)

    sub = pd.DataFrame(p, columns=REQUIRED)
    sub.insert(0, ID_COL, test_ids[ID_COL].values)

    out = "submission_ens_3modelsBEST.csv"
    sub.to_csv(out, index=False)

    print("saved", out, "shape", sub.shape)
    print("weights:", {"cat": W_CAT, "lgb": W_LGB, "hgb": W_HGB})
    print(sub.head())


if __name__ == "__main__":
    main()
