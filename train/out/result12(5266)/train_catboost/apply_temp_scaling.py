# train/apply_temp_scaling.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATA_DIR = "../../data/processed"
ID_COL = "track_id"
TARGET_COL = "bird_group"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

EPS = 1e-15

# ---- INPUTS ----
T = 1.124
OOF_IN  = "oof_proba_cat_weighted.npy"
TEST_IN = "test_proba_cat_weighted.npy"
MAP_IN  = "label_mapping_cat.csv"

OUT_DIR = "../out/result12(5266)/cat_files/result_cat_temp"
os.makedirs(OUT_DIR, exist_ok=True)


def align_to_required(proba: np.ndarray, mapping_csv: str) -> np.ndarray:
    classes = pd.read_csv(mapping_csv)["label"].astype(str).tolist()
    df = pd.DataFrame(proba, columns=classes)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{mapping_csv}: missing columns {missing}")
    p = df[REQUIRED].to_numpy(dtype=np.float64)
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def temp_scale(p: np.ndarray, t: float) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    p = p ** t
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def main():
    oof_raw = np.load(OOF_IN)
    test_raw = np.load(TEST_IN)

    oof = align_to_required(oof_raw, MAP_IN)
    test = align_to_required(test_raw, MAP_IN)

    oof_t = temp_scale(oof, T)
    test_t = temp_scale(test, T)

    # save npy
    np.save(f"{OUT_DIR}/oof_proba_cat_temp.npy", oof_t)
    np.save(f"{OUT_DIR}/test_proba_cat_temp.npy", test_t)

    # mapping
    pd.DataFrame({"label": REQUIRED}).to_csv(f"{OUT_DIR}/label_mapping_cat_temp.csv", index=False)

    # submissions
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")[[ID_COL]]

    sub_proba = pd.concat([test_ids.reset_index(drop=True),
                           pd.DataFrame(test_t, columns=REQUIRED)], axis=1)
    sub_proba.to_csv(f"{OUT_DIR}/submission_cat_temp_proba.csv", index=False)

    pred_idx = test_t.argmax(axis=1)
    sub_label = test_ids.copy()
    sub_label[TARGET_COL] = [REQUIRED[i] for i in pred_idx]
    sub_label.to_csv(f"{OUT_DIR}/submission_cat_temp_label.csv", index=False)

    print("Saved:")
    print(f" - {OUT_DIR}/oof_proba_cat_temp.npy")
    print(f" - {OUT_DIR}/test_proba_cat_temp.npy")
    print(f" - {OUT_DIR}/label_mapping_cat_temp.csv")
    print(f" - {OUT_DIR}/submission_cat_temp_proba.csv")
    print(f" - {OUT_DIR}/submission_cat_temp_label.csv")


if __name__ == "__main__":
    main()
