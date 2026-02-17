# train/train_xgb_seed_ens.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

import xgboost as xgb


DATA_DIR = "../../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

N_SPLITS = 5
RANDOM_STATE = 42

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

# сиды ансамбля (можешь добавить больше)
SEEDS = [1, 2, 3, 4, 5]   # старт
# SEEDS = [1,2,3,4,5,7,9,11]  # если хочешь пожирнее

OUT_DIR = "../out/result11(536)/result_xgb_seedens"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-7


def clip_norm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, 1.0 - EPS)
    p /= p.sum(axis=1, keepdims=True)
    return p


def load_data():
    X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    y_train_df = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")
    X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")[[ID_COL]]
    y_raw = y_train_df[TARGET_COL].astype(str).values
    return X_train, y_raw, X_test, test_ids


def ensure_required_labels(y_raw: np.ndarray) -> np.ndarray:
    missing = sorted(set(y_raw) - set(REQUIRED))
    if missing:
        raise ValueError(f"Train labels contain unknown classes not in REQUIRED: {missing}")
    label_to_idx = {c: i for i, c in enumerate(REQUIRED)}
    return np.array([label_to_idx[c] for c in y_raw], dtype=np.int64)


def encode_non_numeric_together(X_train: pd.DataFrame, X_test: pd.DataFrame):
    Xtr = X_train.copy()
    Xte = X_test.copy()

    non_num = [c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])]
    if non_num:
        print(f"Non-numeric cols -> factorize: {len(non_num)} {non_num[:10]}{'...' if len(non_num) > 10 else ''}")

    for c in non_num:
        combined = pd.concat([Xtr[c], Xte[c]], axis=0).astype(str).fillna("__NA__")
        codes, _ = pd.factorize(combined, sort=True)
        Xtr[c] = codes[: len(Xtr)].astype(np.int32)
        Xte[c] = codes[len(Xtr) :].astype(np.int32)

    for c in Xtr.columns:
        if pd.api.types.is_numeric_dtype(Xtr[c]):
            med = np.nanmedian(Xtr[c].to_numpy(dtype=np.float64))
            if np.isnan(med):
                med = 0.0
            Xtr[c] = Xtr[c].fillna(med)
            Xte[c] = Xte[c].fillna(med)

    return Xtr, Xte


def main():
    print("xgboost version:", getattr(xgb, "__version__", "unknown"))
    print("SEEDS:", SEEDS)

    X_train, y_raw, X_test, test_ids = load_data()
    y = ensure_required_labels(y_raw)

    X_train, X_test = encode_non_numeric_together(X_train, X_test)

    Xtr = X_train.to_numpy(dtype=np.float32)
    Xte = X_test.to_numpy(dtype=np.float32)

    print(f"X_train={Xtr.shape} X_test={Xte.shape} classes={len(REQUIRED)}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros((Xtr.shape[0], len(REQUIRED)), dtype=np.float64)
    test_sum = np.zeros((Xte.shape[0], len(REQUIRED)), dtype=np.float64)

    rows = []
    best_iters = []

    base_params = {
        "objective": "multi:softprob",
        "num_class": len(REQUIRED),
        "eval_metric": "mlogloss",

        "eta": 0.02,
        "max_depth": 6,
        "min_child_weight": 3.0,

        "subsample": 0.8,
        "colsample_bytree": 0.8,

        "lambda": 2.0,
        "alpha": 0.0,
        "gamma": 0.0,

        "tree_method": "hist",
    }

    num_boost_round = 6000
    early_stopping_rounds = 400

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y), 1):
        dtrain = xgb.DMatrix(Xtr[tr_idx], label=y[tr_idx])
        dvalid = xgb.DMatrix(Xtr[va_idx], label=y[va_idx])
        dtest  = xgb.DMatrix(Xte)

        fold_va = np.zeros((len(va_idx), len(REQUIRED)), dtype=np.float64)
        fold_te = np.zeros((Xte.shape[0], len(REQUIRED)), dtype=np.float64)

        fold_best = []

        for s in SEEDS:
            params = dict(base_params)
            params["seed"] = 100000 * fold + s + RANDOM_STATE  # чтобы сид реально менял всё

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,  # чтобы не спамило; хочешь — поставь 200
            )

            best_it = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
            fold_best.append(best_it)

            p_va = booster.predict(dvalid, iteration_range=(0, best_it + 1))
            p_te = booster.predict(dtest,  iteration_range=(0, best_it + 1))

            fold_va += clip_norm(p_va) / len(SEEDS)
            fold_te += clip_norm(p_te) / len(SEEDS)

        oof[va_idx] = fold_va
        test_sum += fold_te / N_SPLITS

        ll = log_loss(y[va_idx], fold_va, labels=np.arange(len(REQUIRED)))
        acc = accuracy_score(y[va_idx], np.argmax(fold_va, axis=1))

        rows.append({
            "fold": fold,
            "logloss": ll,
            "acc": acc,
            "best_iteration_mean": float(np.mean(fold_best)),
            "best_iteration_min": int(np.min(fold_best)),
            "best_iteration_max": int(np.max(fold_best)),
        })
        best_iters.append(fold_best)

        print(f"[fold {fold}] acc={acc:.4f} logloss={ll:.4f} best_it(mean)={np.mean(fold_best):.1f}")

    cv = pd.DataFrame(rows)
    oof_ll = log_loss(y, oof, labels=np.arange(len(REQUIRED)))
    oof_acc = accuracy_score(y, np.argmax(oof, axis=1))

    print("\n=== CV SUMMARY (XGB seed ensemble) ===")
    print(f"acc:     {oof_acc:.4f}")
    print(f"logloss: {oof_ll:.6f}")
    print("best_iteration list per fold (per seed):")
    for i, lst in enumerate(best_iters, 1):
        print(f"  fold {i}: {lst}")

    test_sum = clip_norm(test_sum)

    np.save(f"{OUT_DIR}/oof_proba_xgb_seedens.npy", oof)
    np.save(f"{OUT_DIR}/test_proba_xgb_seedens.npy", test_sum)

    pd.DataFrame({"label": REQUIRED}).to_csv(f"{OUT_DIR}/label_mapping_xgb_seedens.csv", index=False)
    cv.to_csv(f"{OUT_DIR}/cv_metrics_xgb_seedens.csv", index=False)

    sub_proba = pd.concat([test_ids.reset_index(drop=True), pd.DataFrame(test_sum, columns=REQUIRED)], axis=1)
    sub_proba.to_csv(f"{OUT_DIR}/submission_xgb_seedens_proba.csv", index=False)

    pred_idx = test_sum.argmax(axis=1)
    sub_label = test_ids.copy()
    sub_label[TARGET_COL] = [REQUIRED[i] for i in pred_idx]
    sub_label.to_csv(f"{OUT_DIR}/submission_xgb_seedens_label.csv", index=False)

    print("\nSaved files:")
    print(f" - {OUT_DIR}/oof_proba_xgb_seedens.npy")
    print(f" - {OUT_DIR}/test_proba_xgb_seedens.npy")
    print(f" - {OUT_DIR}/label_mapping_xgb_seedens.csv")
    print(f" - {OUT_DIR}/cv_metrics_xgb_seedens.csv")
    print(f" - {OUT_DIR}/submission_xgb_seedens_proba.csv")
    print(f" - {OUT_DIR}/submission_xgb_seedens_label.csv")


if __name__ == "__main__":
    main()
