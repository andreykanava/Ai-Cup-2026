# train_lgbm_ensemble_strong.py
# LGBM multiclass with:
#  - OHE + alignment
#  - multi-seed ensemble
#  - 2 model "profiles" per seed (diversity)
#  - saves per-seed and final averaged probs
#
# Produces (final):
#  - oof_proba_lgbm_ens.npy
#  - test_proba_lgbm_ens.npy
#  - label_mapping_lgbm.csv
#  - submission_lgbm_ens_proba.csv
#  - submission_lgbm_ens.csv
# Also saves per-seed artifacts in ./lgbm_seeds/

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score

import lightgbm as lgb


DATA_DIR = "../../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

N_SPLITS = 5

# ensemble seeds (add/remove)
SEEDS = [1, 42, 1337, 2026, 777]

# If you already have this file (from your CatBoost feature drop):
DROP_CSV = "features_to_drop.csv"

# Training speed/quality knobs
EARLY_STOPPING = 300       # 500 -> safer, 200-300 -> faster
LOG_EVERY = 200
N_ESTIMATORS = 20000       # big ceiling; early stopping will cut it
LEARNING_RATE = 0.05       # faster than 0.03; usually similar quality with reg
FORCE_ROW_WISE = True      # often faster for wide OHE matrices


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

    # Align to TRAIN columns
    train_ohe, test_ohe = train_ohe.align(test_ohe, join="left", axis=1, fill_value=0)

    # Drop constant columns (on train)
    nun = train_ohe.nunique(dropna=False)
    const_cols = nun[nun <= 1].index.tolist()
    if const_cols:
        train_ohe = train_ohe.drop(columns=const_cols)
        test_ohe = test_ohe.drop(columns=const_cols)

    assert list(train_ohe.columns) == list(test_ohe.columns)

    return train_ohe.astype(np.float32), test_ohe.astype(np.float32)


def make_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    # Balanced sample weights: total/(K*count[class])
    c = np.bincount(y, minlength=n_classes).astype(np.float32)
    cw = (c.sum() / (n_classes * np.maximum(c, 1.0))).astype(np.float32)
    return cw


def build_model_profile(profile: str, n_classes: int, seed: int) -> lgb.LGBMClassifier:
    """
    Two profiles to diversify the ensemble:
      - "regularized": more conservative, less overfit
      - "leafy": a bit stronger capacity
    """
    common = dict(
        objective="multiclass",
        num_class=n_classes,
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,

        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,

        # better generalization on OHE
        extra_trees=True,

        # speed for wide matrices
        force_row_wise=FORCE_ROW_WISE,

        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )

    if profile == "regularized":
        return lgb.LGBMClassifier(
            **common,
            num_leaves=48,
            max_depth=-1,
            min_data_in_leaf=120,
            min_child_samples=40,
            min_child_weight=1e-3,
            min_split_gain=0.0,
            reg_alpha=0.0,
            reg_lambda=8.0,
            max_bin=255,
        )

    if profile == "leafy":
        return lgb.LGBMClassifier(
            **common,
            num_leaves=96,
            max_depth=-1,
            min_data_in_leaf=80,
            min_child_samples=25,
            min_child_weight=1e-3,
            min_split_gain=0.0,
            reg_alpha=0.0,
            reg_lambda=5.0,
            max_bin=255,
        )

    raise ValueError(f"Unknown profile: {profile}")


def maybe_drop_features(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(DROP_CSV):
        print(f"[warn] {DROP_CSV} not found, skipping feature drop")
        return X_train_df, X_test_df

    drop_list = pd.read_csv(DROP_CSV)["feature"].astype(str).tolist()
    drop_list = [c for c in drop_list if c in X_train_df.columns]
    print(f"Dropping {len(drop_list)} features from {DROP_CSV}")
    return X_train_df.drop(columns=drop_list), X_test_df.drop(columns=drop_list)


def run_one_seed(
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    classes: list[str],
    seed: int,
    out_dir: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_classes = len(classes)

    # Rare (<2) handling: keep them always in train to avoid fold crash
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

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_proba = np.zeros((len(X_train), n_classes), dtype=np.float32)
    test_proba = np.zeros((len(X_test), n_classes), dtype=np.float32)

    fold_acc, fold_ll, fold_best = [], [], []

    X_ok = X_train.iloc[ok_idx]
    y_ok = y[ok_idx]

    profiles = ["regularized", "leafy"]  # average both profiles inside each fold

    print(f"\n===== SEED {seed} | classes={n_classes} rare_classes={len(rare_classes)} n_splits={n_splits} =====")

    for fold, (tr_local, va_local) in enumerate(skf.split(X_ok, y_ok), 1):
        tr_idx = ok_idx[tr_local]
        va_idx = ok_idx[va_local]
        if len(rare_idx) > 0:
            tr_idx = np.concatenate([tr_idx, rare_idx])

        X_tr = X_train.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y[va_idx]

        cw = make_class_weights(y_tr, n_classes)
        sw = cw[y_tr]

        # average of two profiles for diversity
        va_proba_avg = np.zeros((len(X_va), n_classes), dtype=np.float32)
        te_proba_avg = np.zeros((len(X_test), n_classes), dtype=np.float32)
        best_iters = []

        for profile in profiles:
            model = build_model_profile(profile, n_classes, seed + 10_000 * fold)  # fold-dependent seed for extra variety

            model.fit(
                X_tr,
                y_tr,
                sample_weight=sw,
                eval_set=[(X_va, y_va)],
                eval_metric="multi_logloss",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
                    lgb.log_evaluation(period=LOG_EVERY),
                ],
            )

            va_proba_avg += model.predict_proba(X_va).astype(np.float32)
            te_proba_avg += model.predict_proba(X_test).astype(np.float32)
            best_iters.append(int(model.best_iteration_ or 0))

        va_proba_avg /= len(profiles)
        te_proba_avg /= len(profiles)

        oof_proba[va_idx] = va_proba_avg

        acc = accuracy_score(y_va, va_proba_avg.argmax(axis=1))
        ll = log_loss(y_va, va_proba_avg, labels=np.arange(n_classes))

        fold_acc.append(acc)
        fold_ll.append(ll)
        fold_best.append(int(np.mean(best_iters)))

        print(f"[seed {seed} | fold {fold}] acc={acc:.4f} logloss={ll:.5f} best_iter~{fold_best[-1]}")

        test_proba += te_proba_avg

    test_proba /= len(fold_acc)

    # summary (non-rare only)
    overall_acc = accuracy_score(y[ok_idx], oof_proba[ok_idx].argmax(axis=1))
    overall_ll = log_loss(y[ok_idx], oof_proba[ok_idx], labels=np.arange(n_classes))

    metrics = {
        "seed": seed,
        "n_splits": int(n_splits),
        "rare_classes": int(len(rare_classes)),
        "cv_acc_nonrare": float(overall_acc),
        "cv_logloss_nonrare": float(overall_ll),
        "best_iter_mean": float(np.mean(fold_best) if fold_best else 0.0),
    }

    # save per-seed
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"oof_proba_lgbm_seed{seed}.npy"), oof_proba)
    np.save(os.path.join(out_dir, f"test_proba_lgbm_seed{seed}.npy"), test_proba)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, f"metrics_seed{seed}.csv"), index=False)

    return oof_proba, test_proba, metrics


def main():
    X_train_df, y_raw, X_test_df, test_ids = load_data()

    X_train_df, X_test_df = maybe_drop_features(X_train_df, X_test_df)

    X_train, X_test = ohe_align(X_train_df, X_test_df)
    print(f"After OHE: X_train={X_train.shape} X_test={X_test.shape}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    classes = le.classes_.tolist()
    n_classes = len(classes)
    print("classes:", classes)

    # where to store per-seed artifacts
    seed_dir = "../lgbm_seeds"
    os.makedirs(seed_dir, exist_ok=True)

    # ensemble accumulators
    oof_sum = np.zeros((len(X_train), n_classes), dtype=np.float32)
    test_sum = np.zeros((len(X_test), n_classes), dtype=np.float32)
    all_metrics = []

    for seed in SEEDS:
        oof_proba, test_proba, metrics = run_one_seed(
            X_train=X_train,
            y=y,
            X_test=X_test,
            classes=classes,
            seed=seed,
            out_dir=seed_dir,
        )
        oof_sum += oof_proba
        test_sum += test_proba
        all_metrics.append(metrics)

    oof_ens = oof_sum / len(SEEDS)
    test_ens = test_sum / len(SEEDS)

    # overall ensemble CV (non-rare handling not applied here; still useful signal)
    overall_acc = accuracy_score(y, oof_ens.argmax(axis=1))
    overall_ll = log_loss(y, oof_ens, labels=np.arange(n_classes))

    print("\n===== ENSEMBLE SUMMARY (all rows) =====")
    print(f"acc:     {overall_acc:.4f}")
    print(f"logloss: {overall_ll:.5f}")

    # save final artifacts
    np.save("oof_proba_lgbm_ens.npy", oof_ens)
    np.save("test_proba_lgbm_ens.npy", test_ens)
    pd.DataFrame({"label": classes}).to_csv("label_mapping_lgbm.csv", index=False)

    pd.DataFrame(all_metrics).to_csv(os.path.join(seed_dir, "metrics_all_seeds.csv"), index=False)

    # submission with probabilities
    sub_proba = pd.DataFrame(test_ens, columns=classes)
    sub_proba.insert(0, ID_COL, test_ids[ID_COL].values)
    sub_proba.to_csv("submission_lgbm_ens_proba.csv", index=False)

    # hard-label submission
    test_pred = le.inverse_transform(test_ens.argmax(axis=1))
    sub_hard = pd.DataFrame({ID_COL: test_ids[ID_COL].values, TARGET_COL: test_pred})
    sub_hard.to_csv("submission_lgbm_ens.csv", index=False)

    print("\nSaved FINAL:")
    print(" - submission_lgbm_ens_proba.csv")
    print(" - submission_lgbm_ens.csv")
    print(" - oof_proba_lgbm_ens.npy / test_proba_lgbm_ens.npy")
    print(" - label_mapping_lgbm.csv")
    print(f" - per-seed stuff in ./{seed_dir}/")


if __name__ == "__main__":
    main()
