# train/train_catboost_3cfg_x2seed.py
# Train 3 CatBoost configs × 2 seeds with StratifiedKFold OOF + Test proba saving.
# Outputs (per model):
#  - out/<RUN_NAME>/<model_name>/oof.npy
#  - out/<RUN_NAME>/<model_name>/test.npy
#  - out/<RUN_NAME>/<model_name>/meta.json
# Also outputs:
#  - out/<RUN_NAME>/oof_mean.npy
#  - out/<RUN_NAME>/test_mean.npy
#  - out/<RUN_NAME>/label_mapping.csv
#  - out/<RUN_NAME>/submission_mean.csv

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from catboost import CatBoostClassifier


# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "../../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

N_SPLITS = 5
USE_GPU = False

# feature drop list (optional). set to None to disable
FEATURES_TO_DROP_CSV = "../features_to_drop.csv"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

RUN_NAME = "cat_3cfg_x2seed"
OUT_DIR = f"result"
os.makedirs(OUT_DIR, exist_ok=True)

# 3 configs × 2 seeds = 6 models total
SEEDS = [228, 1488]

CONFIGS: List[Dict] = [
    # cfg0: baseline-ish
    dict(
        name="cfg0_depth6_rsm075",
        iterations=7000,
        learning_rate=0.03,
        depth=6,
        rsm=0.75,
        l2_leaf_reg=15.0,
        min_data_in_leaf=20,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        random_strength=1.5,
        od_wait=250,
    ),
    # cfg1: slightly shallower + stronger regularization
    dict(
        name="cfg1_depth5_rsm09_reg30",
        iterations=9000,
        learning_rate=0.025,
        depth=5,
        rsm=0.90,
        l2_leaf_reg=30.0,
        min_data_in_leaf=30,
        bootstrap_type="Bayesian",
        bagging_temperature=0.6,
        random_strength=2.0,
        od_wait=300,
    ),
    # cfg2: deeper + more stochasticity
    dict(
        name="cfg2_depth7_rsm06_hotbag",
        iterations=9000,
        learning_rate=0.022,
        depth=7,
        rsm=0.60,
        l2_leaf_reg=12.0,
        min_data_in_leaf=15,
        bootstrap_type="Bayesian",
        bagging_temperature=2.0,
        random_strength=1.0,
        od_wait=300,
    ),
]


# -----------------------
# IO / PREPROCESS
# -----------------------
def load_data() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    y_train_df = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")
    X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")

    y_raw = y_train_df[TARGET_COL].astype(str).values
    return X_train, y_raw, X_test, test_ids


def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    # drop features
    drop_list: List[str] = []
    if FEATURES_TO_DROP_CSV and os.path.exists(FEATURES_TO_DROP_CSV):
        drop_list = pd.read_csv(FEATURES_TO_DROP_CSV)["feature"].astype(str).tolist()
        drop_list = [c for c in drop_list if c in X_train.columns]

    if drop_list:
        print(f"Dropping {len(drop_list)} features")
        X_train = X_train.drop(columns=drop_list)
        X_test = X_test.drop(columns=drop_list)
    else:
        print("Dropping 0 features")

    # ensure cat columns are strings with missing token
    for df in (X_train, X_test):
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype("string").fillna("__MISSING__")

    cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    if cat_cols:
        print(f"Categorical columns: {len(cat_cols)}")
    else:
        print("Categorical columns: 0")

    return X_train, X_test, cat_idx


def ensure_required_labels(classes: np.ndarray):
    extra = sorted(set(classes) - set(REQUIRED))
    missing = sorted(set(REQUIRED) - set(classes))
    if extra or missing:
        raise ValueError(f"Label mismatch. extra={extra}, missing={missing}")


# -----------------------
# TRAINING
# -----------------------
def train_one_model(
    cfg: Dict,
    seed: int,
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    cat_idx: List[int],
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray, float, List[float], List[int]]:

    model_name = f"{cfg['name']}_seed{seed}"
    print(f"\n========== {model_name} ==========")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    oof = np.zeros((len(X_train), n_classes), dtype=np.float32)
    test = np.zeros((len(X_test), n_classes), dtype=np.float32)

    fold_ll: List[float] = []
    best_iters: List[int] = []

    for fold, (tr, va) in enumerate(skf.split(X_train, y), 1):
        X_tr, y_tr = X_train.iloc[tr], y[tr]
        X_va, y_va = X_train.iloc[va], y[va]

        params = dict(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=int(cfg["iterations"]),
            learning_rate=float(cfg["learning_rate"]),
            depth=int(cfg["depth"]),
            rsm=float(cfg["rsm"]),
            l2_leaf_reg=float(cfg["l2_leaf_reg"]),
            min_data_in_leaf=int(cfg["min_data_in_leaf"]),
            bootstrap_type=str(cfg["bootstrap_type"]),
            bagging_temperature=float(cfg["bagging_temperature"]),
            random_strength=float(cfg["random_strength"]),
            random_seed=int(seed),
            od_type="Iter",
            od_wait=int(cfg["od_wait"]),
            task_type="GPU" if USE_GPU else "CPU",
            thread_count=-1,
            verbose=200,
        )

        model = CatBoostClassifier(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            cat_features=cat_idx if cat_idx else None,
            use_best_model=True,
        )

        va_proba = model.predict_proba(X_va)
        oof[va] = va_proba

        ll = log_loss(y_va, va_proba, labels=np.arange(n_classes))
        fold_ll.append(float(ll))

        best_iter = int(getattr(model, "best_iteration_", -1))
        best_iters.append(best_iter)

        print(f"[{model_name} fold {fold}] logloss={ll:.6f} best_iter={best_iter}")

        test += model.predict_proba(X_test) / N_SPLITS

    full_ll = float(log_loss(y, oof, labels=np.arange(n_classes)))
    print(f"{model_name} CV logloss = {full_ll:.6f}")

    return oof, test, full_ll, fold_ll, best_iters


def save_model_outputs(run_dir: str, model_name: str, oof: np.ndarray, test: np.ndarray, meta: Dict):
    mdir = os.path.join(run_dir, model_name)
    os.makedirs(mdir, exist_ok=True)

    np.save(os.path.join(mdir, "oof.npy"), oof)
    np.save(os.path.join(mdir, "test.npy"), test)

    with open(os.path.join(mdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    X_train, y_raw, X_test, test_ids = load_data()
    X_train, X_test, cat_idx = preprocess(X_train, X_test)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    ensure_required_labels(classes)

    n_classes = len(classes)
    print("classes:", list(classes))
    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    # Save mapping once
    pd.DataFrame({"label": classes}).to_csv(f"{OUT_DIR}/label_mapping.csv", index=False)

    all_oof = []
    all_test = []
    all_scores = []

    for cfg in CONFIGS:
        for seed in SEEDS:
            model_name = f"{cfg['name']}_seed{seed}"

            oof, test, score, fold_ll, best_iters = train_one_model(
                cfg, seed, X_train, y, X_test, cat_idx, n_classes
            )

            meta = {
                "model_name": model_name,
                "cfg": cfg,
                "seed": seed,
                "n_splits": N_SPLITS,
                "cv_logloss": score,
                "fold_logloss": fold_ll,
                "best_iters": best_iters,
                "best_iter_mean": float(np.mean([x for x in best_iters if x >= 0])) if any(x >= 0 for x in best_iters) else None,
                "best_iter_median": float(np.median([x for x in best_iters if x >= 0])) if any(x >= 0 for x in best_iters) else None,
                "use_gpu": USE_GPU,
                "n_features": int(X_train.shape[1]),
                "n_classes": int(n_classes),
            }

            save_model_outputs(OUT_DIR, model_name, oof, test, meta)

            all_oof.append(oof)
            all_test.append(test)
            all_scores.append((model_name, score))

    # mean ensemble of 6 models (simple average)
    oof_mean = np.mean(np.stack(all_oof, axis=0), axis=0)
    test_mean = np.mean(np.stack(all_test, axis=0), axis=0)

    final_ll = float(log_loss(y, oof_mean, labels=np.arange(n_classes)))
    print("\n========== FINAL MEAN ENSEMBLE (6 models) ==========")
    print("model scores:")
    for name, sc in sorted(all_scores, key=lambda x: x[1]):
        print(f" - {name}: {sc:.6f}")
    print("mean-ensemble OOF logloss:", final_ll)

    np.save(f"{OUT_DIR}/oof_mean.npy", oof_mean)
    np.save(f"{OUT_DIR}/test_mean.npy", test_mean)

    # submission_mean.csv (probabilities)
    proba_df = pd.DataFrame(test_mean, columns=classes)[REQUIRED]
    sub = pd.concat(
        [test_ids[[ID_COL]].reset_index(drop=True), proba_df.reset_index(drop=True)],
        axis=1
    )
    sub.to_csv(f"{OUT_DIR}/submission_mean.csv", index=False)
    print("Saved:", f"{OUT_DIR}/submission_mean.csv")


if __name__ == "__main__":
    main()
