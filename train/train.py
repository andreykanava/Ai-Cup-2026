# train_lgbm_strong.py
# Strong LGBM for multiclass with OHE alignment + OOF/Test proba saving
# Produces:
#  - oof_proba_lgbm.npy
#  - test_proba_lgbm.npy
#  - label_mapping_lgbm.csv
#  - submission_lgbm_proba.csv (track_id + class probability columns)

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score

import lightgbm as lgb


DATA_DIR = "../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

N_SPLITS = 5
RANDOM_STATE = 42


def load_data():
    X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    y_train_df = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")
    X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")

    y_raw = y_train_df[TARGET_COL].astype(str).values
    return X_train, y_raw, X_test, test_ids


def ohe_align(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # make all non-numeric -> string with __MISSING__
    non_num = [c for c in train_df.columns if not pd.api.types.is_numeric_dtype(train_df[c])]
    for c in non_num:
        train_df[c] = train_df[c].astype("string").fillna("__MISSING__")
        test_df[c] = test_df[c].astype("string").fillna("__MISSING__")

    # OHE
    train_ohe = pd.get_dummies(train_df, dummy_na=False)
    test_ohe = pd.get_dummies(test_df, dummy_na=False)

    # Align to TRAIN columns (safe & standard)
    train_ohe, test_ohe = train_ohe.align(test_ohe, join="left", axis=1, fill_value=0)

    # Drop constant columns (on train)
    nun = train_ohe.nunique(dropna=False)
    const_cols = nun[nun <= 1].index.tolist()
    if const_cols:
        train_ohe = train_ohe.drop(columns=const_cols)
        test_ohe = test_ohe.drop(columns=const_cols)

    # sanity
    assert list(train_ohe.columns) == list(test_ohe.columns)

    # float32 for speed/memory
    return train_ohe.astype(np.float32), test_ohe.astype(np.float32)


def build_model(n_classes: int):
    # "Ideal" strong baseline for OHE multi-class:
    # - moderate LR + many estimators
    # - regularization to fight overfit
    # - subsample/colsample to generalize
    # - min_child_samples not too small (OHE can overfit like crazy)
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,

        learning_rate=0.03,
        n_estimators=15000,

        # tree complexity
        num_leaves=64,
        max_depth=8,
        min_data_in_leaf=100,

        # stability / generalization
        min_child_samples=30,
        min_child_weight=1e-3,
        min_split_gain=0.0,

        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,

        # regularization
        reg_alpha=0.0,
        reg_lambda=6.0,

        # extra
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def make_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    # Balanced sample weights: total/(K*count[class])
    c = np.bincount(y, minlength=n_classes).astype(np.float32)
    cw = (c.sum() / (n_classes * np.maximum(c, 1.0))).astype(np.float32)
    return cw


def main():
    X_train_df, y_raw, X_test_df, test_ids = load_data()

    drop_list = pd.read_csv("features_to_drop.csv")["feature"].astype(str).tolist()

    # drop only columns that exist (safe)
    drop_list = [c for c in drop_list if c in X_train_df.columns]

    print(f"Dropping {len(drop_list)} features")
    X_train_df = X_train_df.drop(columns=drop_list)
    X_test_df = X_test_df.drop(columns=drop_list)

    X_train, X_test = ohe_align(X_train_df, X_test_df)
    print(f"After OHE: X_train={X_train.shape} X_test={X_test.shape}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    classes = le.classes_.tolist()
    n_classes = len(classes)

    # Rare class handling: put all rare (<2) always into train, never in val
    counts = np.bincount(y, minlength=n_classes)
    rare_classes = np.where(counts < 2)[0]
    rare_mask = np.isin(y, rare_classes)
    rare_idx = np.where(rare_mask)[0]
    ok_idx = np.where(~rare_mask)[0]

    ok_counts = np.bincount(y[ok_idx], minlength=n_classes)
    ok_min = int(ok_counts[ok_counts > 0].min()) if (ok_counts > 0).any() else 2
    n_splits = min(N_SPLITS, ok_min)
    if n_splits < 2:
        n_splits = 2

    print(f"classes={n_classes} rare_classes={len(rare_classes)} n_splits={n_splits}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_proba = np.zeros((len(X_train), n_classes), dtype=np.float32)
    test_proba = np.zeros((len(X_test), n_classes), dtype=np.float32)

    fold_acc, fold_ll, fold_best = [], [], []

    X_ok = X_train.iloc[ok_idx]
    y_ok = y[ok_idx]

    for fold, (tr_local, va_local) in enumerate(skf.split(X_ok, y_ok), 1):
        tr_idx = ok_idx[tr_local]
        va_idx = ok_idx[va_local]

        # force rare into train
        if len(rare_idx) > 0:
            tr_idx = np.concatenate([tr_idx, rare_idx])

        X_tr = X_train.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y[va_idx]

        # class-balanced sample weights on train fold
        cw = make_class_weights(y_tr, n_classes)
        sw = cw[y_tr]

        model = build_model(n_classes)

        model.fit(
            X_tr,
            y_tr,
            sample_weight=sw,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(stopping_rounds=500, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )

        va_proba = model.predict_proba(X_va)
        oof_proba[va_idx] = va_proba

        acc = accuracy_score(y_va, va_proba.argmax(axis=1))
        ll = log_loss(y_va, va_proba, labels=np.arange(n_classes))

        fold_acc.append(acc)
        fold_ll.append(ll)
        fold_best.append(int(model.best_iteration_ or 0))

        print(f"[fold {fold}] acc={acc:.4f} logloss={ll:.5f} best_iter={model.best_iteration_}")

        test_proba += model.predict_proba(X_test)

    # average test by actual folds
    test_proba /= len(fold_acc)

    # CV summary (non-rare only)
    overall_acc = accuracy_score(y[ok_idx], oof_proba[ok_idx].argmax(axis=1))
    overall_ll = log_loss(y[ok_idx], oof_proba[ok_idx], labels=np.arange(n_classes))

    print("\n=== CV SUMMARY (non-rare only) ===")
    print(f"acc:     {overall_acc:.4f}")
    print(f"logloss: {overall_ll:.5f}")
    print(f"best_iter mean: {np.mean(fold_best):.1f}")

    # Save probs
    np.save("oof_proba_lgbm.npy", oof_proba)
    np.save("test_proba_lgbm.npy", test_proba)
    pd.DataFrame({"label": classes}).to_csv("label_mapping_lgbm.csv", index=False)

    # Save submission with probabilities (track_id + class columns)
    sub_proba = pd.DataFrame(test_proba, columns=classes)
    sub_proba.insert(0, ID_COL, test_ids[ID_COL].values)
    sub_proba.to_csv("submission_lgbm_proba.csv", index=False)

    # Also save hard-label submission if you want
    test_pred = le.inverse_transform(test_proba.argmax(axis=1))
    sub_hard = pd.DataFrame({ID_COL: test_ids[ID_COL].values, TARGET_COL: test_pred})
    sub_hard.to_csv("submission_lgbm.csv", index=False)

    print("\nSaved:")
    print(" - submission_lgbm_proba.csv")
    print(" - submission_lgbm.csv")
    print(" - oof_proba_lgbm.npy / test_proba_lgbm.npy")
    print(" - label_mapping_lgbm.csv")


if __name__ == "__main__":
    main()
