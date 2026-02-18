# train/train_lgbm_multicfg_xseed.py
# LGBM multiclass with:
#  - OHE + alignment
#  - multi-seed
#  - multi-config (6 configs)
#  - saves per-(cfg×seed) oof/test
#  - saves per-seed ensemble (mean over configs for that seed)
#  - saves per-cfg ensemble (mean over seeds for that cfg)
#  - NO mandatory global ensemble (so you can search weights later)
#
# FIX:
#  - DART does NOT support early stopping -> we skip early_stopping callback for dart
#  - DART uses its own n_estimators ceiling (cfg-level), default 6000, to avoid endless training
#
# Outputs:
#  out/<RUN_NAME>/
#    label_mapping_lgbm.csv
#    oof_seed<seed>.npy
#    test_seed<seed>.npy
#    submission_seed<seed>.csv
#    scores_seed<seed>.json
#    oof_cfg_<cfgname>.npy
#    test_cfg_<cfgname>.npy
#    submission_cfg_<cfgname>.csv
#    scores_cfg_<cfgname>.json
#    <cfgname>_seed<seed>/
#       oof.npy
#       test.npy
#       meta.json

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score

import lightgbm as lgb


# -----------------------
# CONFIG (edit here)
# -----------------------
DATA_DIR = "../../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

RUN_NAME = "lgbm_6cfg_xseed"
OUT_DIR = os.path.join("out", RUN_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

N_SPLITS = 5
SEEDS = [743, 756, 214, 463, 235]  # edit

DROP_CSV = "../features_to_drop.csv"  # optional; keep if you use same list
FORCE_ROW_WISE = True

EARLY_STOPPING = 450
LOG_EVERY = 200
N_ESTIMATORS = 30000
LEARNING_RATE = 0.05

# DART-specific ceilings (prevent "forever" training)
DART_N_ESTIMATORS_DEFAULT = 6000
DART_LEARNING_RATE_DEFAULT = 0.03

# If True: also compute/save global mean over ALL cfg×seed (not recommended if you plan weight search)
SAVE_GLOBAL_MEAN = False

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]


# -----------------------
# 6 CONFIG VARIATIONS
# -----------------------
# Все общие параметры задаются в build_model(); тут только различия.
LGBM_CONFIGS: List[Dict] = [
    dict(
        name="cfg0_safe_reg",
        boosting_type="gbdt",
        num_leaves=48,
        min_data_in_leaf=180,
        reg_lambda=12.0,
        reg_alpha=0.0,
        colsample_bytree=0.70,
        subsample=0.80,
        extra_trees=False,
        max_bin=255,
    ),
    dict(
        name="cfg1_capacity",
        boosting_type="gbdt",
        num_leaves=80,
        min_data_in_leaf=120,
        reg_lambda=6.0,
        reg_alpha=0.0,
        colsample_bytree=0.75,
        subsample=0.85,
        extra_trees=False,
        max_bin=255,
    ),
    dict(
        name="cfg2_extratrees",
        boosting_type="gbdt",
        num_leaves=64,
        min_data_in_leaf=140,
        reg_lambda=8.0,
        reg_alpha=0.0,
        colsample_bytree=0.65,
        subsample=0.75,
        extra_trees=True,
        max_bin=255,
    ),
    dict(
        name="cfg3_dart",
        boosting_type="dart",
        num_leaves=64,
        min_data_in_leaf=140,
        reg_lambda=8.0,
        reg_alpha=0.0,
        colsample_bytree=0.70,
        subsample=0.80,
        extra_trees=False,
        max_bin=255,
        # dart knobs
        drop_rate=0.10,
        skip_drop=0.50,
        max_drop=50,
        # ceilings for dart (override defaults if you want)
        n_estimators_dart=6000,
        learning_rate_dart=0.03,
    ),
    dict(
        name="cfg4_strong_reg",
        boosting_type="gbdt",
        num_leaves=40,
        min_data_in_leaf=240,
        reg_lambda=18.0,
        reg_alpha=0.0,
        colsample_bytree=0.60,
        subsample=0.80,
        extra_trees=False,
        max_bin=255,
    ),
    dict(
        name="cfg5_leafy_stoch",
        boosting_type="gbdt",
        num_leaves=96,
        min_data_in_leaf=110,
        reg_lambda=5.0,
        reg_alpha=0.0,
        colsample_bytree=0.62,
        subsample=0.70,
        extra_trees=True,
        max_bin=255,
    ),
]


# -----------------------
# DATA
# -----------------------
def load_data():
    X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    y_train_df = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")
    X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")

    y_raw = y_train_df[TARGET_COL].astype(str).values
    return X_train, y_raw, X_test, test_ids


def maybe_drop_features(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DROP_CSV or not os.path.exists(DROP_CSV):
        print(f"[warn] drop list not found ({DROP_CSV}), skipping")
        return X_train_df, X_test_df
    drop_list = pd.read_csv(DROP_CSV)["feature"].astype(str).tolist()
    drop_list = [c for c in drop_list if c in X_train_df.columns]
    print(f"Dropping {len(drop_list)} features from {DROP_CSV}")
    return X_train_df.drop(columns=drop_list), X_test_df.drop(columns=drop_list)


def ohe_align(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    non_num = [c for c in train_df.columns if not pd.api.types.is_numeric_dtype(train_df[c])]
    for c in non_num:
        train_df[c] = train_df[c].astype("string").fillna("__MISSING__")
        test_df[c] = test_df[c].astype("string").fillna("__MISSING__")

    train_ohe = pd.get_dummies(train_df, dummy_na=False)
    test_ohe = pd.get_dummies(test_df, dummy_na=False)

    train_ohe, test_ohe = train_ohe.align(test_ohe, join="left", axis=1, fill_value=0)

    nun = train_ohe.nunique(dropna=False)
    const_cols = nun[nun <= 1].index.tolist()
    if const_cols:
        train_ohe = train_ohe.drop(columns=const_cols)
        test_ohe = test_ohe.drop(columns=const_cols)

    assert list(train_ohe.columns) == list(test_ohe.columns)
    return train_ohe.astype(np.float32), test_ohe.astype(np.float32)


def make_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    c = np.bincount(y, minlength=n_classes).astype(np.float32)
    cw = (c.sum() / (n_classes * np.maximum(c, 1.0))).astype(np.float32)
    return cw


def save_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_submission(test_proba: np.ndarray, classes: List[str], test_ids: pd.DataFrame, out_csv: str):
    df = pd.DataFrame(test_proba, columns=classes)[REQUIRED]
    sub = pd.concat([test_ids[[ID_COL]].reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    sub.to_csv(out_csv, index=False)


# -----------------------
# MODEL
# -----------------------
def build_model(cfg: Dict, n_classes: int, seed: int) -> lgb.LGBMClassifier:
    boosting_type = str(cfg.get("boosting_type", "gbdt")).lower()

    # global defaults
    lr = float(cfg.get("learning_rate", LEARNING_RATE))
    n_estimators = int(cfg.get("n_estimators", N_ESTIMATORS))

    # dart-specific overrides (prevents endless training)
    if boosting_type == "dart":
        lr = float(cfg.get("learning_rate_dart", DART_LEARNING_RATE_DEFAULT))
        n_estimators = int(cfg.get("n_estimators_dart", DART_N_ESTIMATORS_DEFAULT))

    common = dict(
        objective="multiclass",
        num_class=n_classes,
        learning_rate=lr,
        n_estimators=n_estimators,

        subsample=float(cfg["subsample"]),
        subsample_freq=1,
        colsample_bytree=float(cfg["colsample_bytree"]),

        extra_trees=bool(cfg.get("extra_trees", False)),
        boosting_type=boosting_type,

        num_leaves=int(cfg["num_leaves"]),
        min_data_in_leaf=int(cfg["min_data_in_leaf"]),
        min_child_samples=int(cfg.get("min_child_samples", 25)),
        min_child_weight=float(cfg.get("min_child_weight", 1e-3)),
        min_split_gain=float(cfg.get("min_split_gain", 0.0)),

        reg_alpha=float(cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(cfg.get("reg_lambda", 0.0)),
        max_bin=int(cfg.get("max_bin", 255)),

        force_row_wise=FORCE_ROW_WISE,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
    )

    # DART knobs (only used if boosting_type='dart')
    if boosting_type == "dart":
        common["drop_rate"] = float(cfg.get("drop_rate", 0.10))
        common["skip_drop"] = float(cfg.get("skip_drop", 0.50))
        common["max_drop"] = int(cfg.get("max_drop", 50))

    return lgb.LGBMClassifier(**common)


def make_callbacks(cfg: Dict) -> list:
    boosting_type = str(cfg.get("boosting_type", "gbdt")).lower()
    cbs = [lgb.log_evaluation(period=LOG_EVERY)]
    # FIX: early stopping not available for dart -> skip it
    if boosting_type != "dart":
        cbs.insert(0, lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False))
    return cbs


# -----------------------
# TRAIN ONE (cfg×seed)
# -----------------------
def train_one_cfg_seed(
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    classes: List[str],
    cfg: Dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict]:
    n_classes = len(classes)

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

    oof = np.zeros((len(X_train), n_classes), dtype=np.float32)
    test = np.zeros((len(X_test), n_classes), dtype=np.float32)

    fold_ll, fold_acc, fold_best = [], [], []

    X_ok = X_train.iloc[ok_idx]
    y_ok = y[ok_idx]

    cfg_name = cfg["name"]
    model_tag = f"{cfg_name}_seed{seed}"
    boosting_type = str(cfg.get("boosting_type", "gbdt")).lower()
    print(f"\n===== {model_tag} | n_splits={n_splits} rare={len(rare_classes)} boosting={boosting_type} =====")

    callbacks = make_callbacks(cfg)

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

        model = build_model(cfg, n_classes, seed + 1000 * fold)  # fold-variety

        model.fit(
            X_tr,
            y_tr,
            sample_weight=sw,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=callbacks,
        )

        va_proba = model.predict_proba(X_va).astype(np.float32)
        te_proba = model.predict_proba(X_test).astype(np.float32)

        oof[va_idx] = va_proba

        acc = accuracy_score(y_va, va_proba.argmax(axis=1))
        ll = log_loss(y_va, va_proba, labels=np.arange(n_classes))

        fold_acc.append(float(acc))
        fold_ll.append(float(ll))

        # best_iteration_ exists only with early stopping; for dart it'll be None
        bi = getattr(model, "best_iteration_", None)
        fold_best.append(int(bi) if (bi is not None and bi != 0) else (-1 if boosting_type == "dart" else 0))

        test += te_proba

        print(f"[{model_tag} fold {fold}] acc={acc:.4f} logloss={ll:.5f} best_iter={fold_best[-1]}")

    test /= len(fold_acc)

    overall_acc = accuracy_score(y[ok_idx], oof[ok_idx].argmax(axis=1))
    overall_ll = log_loss(y[ok_idx], oof[ok_idx], labels=np.arange(n_classes))

    metrics = {
        "cfg": cfg_name,
        "seed": int(seed),
        "boosting_type": boosting_type,
        "n_splits": int(n_splits),
        "rare_classes": int(len(rare_classes)),
        "cv_acc_nonrare": float(overall_acc),
        "cv_logloss_nonrare": float(overall_ll),
        "best_iter_mean": float(np.mean([x for x in fold_best if x >= 0]) if any(x >= 0 for x in fold_best) else -1.0),
        "fold_acc": fold_acc,
        "fold_logloss": fold_ll,
        "fold_best_iter": fold_best,
        "cfg_params": {k: v for k, v in cfg.items() if k != "name"},
        "globals": {
            "EARLY_STOPPING": EARLY_STOPPING,
            "LOG_EVERY": LOG_EVERY,
            "N_ESTIMATORS": N_ESTIMATORS,
            "LEARNING_RATE": LEARNING_RATE,
            "DART_N_ESTIMATORS_DEFAULT": DART_N_ESTIMATORS_DEFAULT,
            "DART_LEARNING_RATE_DEFAULT": DART_LEARNING_RATE_DEFAULT,
        },
    }

    return oof, test, metrics


def save_cfg_seed_artifacts(cfg_name: str, seed: int, oof: np.ndarray, test: np.ndarray, meta: Dict):
    mdir = os.path.join(OUT_DIR, f"{cfg_name}_seed{seed}")
    os.makedirs(mdir, exist_ok=True)
    np.save(os.path.join(mdir, "oof.npy"), oof)
    np.save(os.path.join(mdir, "test.npy"), test)
    save_json(os.path.join(mdir, "meta.json"), meta)


# -----------------------
# MAIN
# -----------------------
def main():
    X_train_df, y_raw, X_test_df, test_ids = load_data()
    X_train_df, X_test_df = maybe_drop_features(X_train_df, X_test_df)

    X_train, X_test = ohe_align(X_train_df, X_test_df)
    print(f"After OHE: X_train={X_train.shape} X_test={X_test.shape}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    classes = le.classes_.tolist()
    print("classes:", classes)

    pd.DataFrame({"label": classes}).to_csv(os.path.join(OUT_DIR, "label_mapping_lgbm.csv"), index=False)
    n_classes = len(classes)

    all_models = []

    per_seed_oofs: Dict[int, List[np.ndarray]] = {s: [] for s in SEEDS}
    per_seed_tests: Dict[int, List[np.ndarray]] = {s: [] for s in SEEDS}
    per_seed_scores: Dict[int, List[Tuple[str, float]]] = {s: [] for s in SEEDS}

    per_cfg_oofs: Dict[str, List[np.ndarray]] = {c["name"]: [] for c in LGBM_CONFIGS}
    per_cfg_tests: Dict[str, List[np.ndarray]] = {c["name"]: [] for c in LGBM_CONFIGS}
    per_cfg_scores: Dict[str, List[Tuple[int, float]]] = {c["name"]: [] for c in LGBM_CONFIGS}

    for cfg in LGBM_CONFIGS:
        cfg_name = cfg["name"]
        for seed in SEEDS:
            oof, test, meta = train_one_cfg_seed(
                X_train=X_train,
                y=y,
                X_test=X_test,
                classes=classes,
                cfg=cfg,
                seed=seed,
            )

            save_cfg_seed_artifacts(cfg_name, seed, oof, test, meta)

            score = float(meta["cv_logloss_nonrare"])
            all_models.append({"model": f"{cfg_name}_seed{seed}", "cv_logloss_nonrare": score})

            per_seed_oofs[seed].append(oof)
            per_seed_tests[seed].append(test)
            per_seed_scores[seed].append((cfg_name, score))

            per_cfg_oofs[cfg_name].append(oof)
            per_cfg_tests[cfg_name].append(test)
            per_cfg_scores[cfg_name].append((seed, score))

    # ---- per-seed mean over cfgs ----
    for seed in SEEDS:
        oof_seed = np.mean(np.stack(per_seed_oofs[seed], axis=0), axis=0)
        test_seed = np.mean(np.stack(per_seed_tests[seed], axis=0), axis=0)
        ll_seed = float(log_loss(y, oof_seed, labels=np.arange(n_classes)))

        np.save(os.path.join(OUT_DIR, f"oof_seed{seed}.npy"), oof_seed)
        np.save(os.path.join(OUT_DIR, f"test_seed{seed}.npy"), test_seed)

        sub_path = os.path.join(OUT_DIR, f"submission_seed{seed}.csv")
        make_submission(test_seed, classes, test_ids, sub_path)

        dump = {
            "seed": int(seed),
            "seed_ensemble_oof_logloss_allrows": ll_seed,
            "cfg_scores_nonrare_sorted": [
                {"cfg": n, "cv_logloss_nonrare": sc}
                for (n, sc) in sorted(per_seed_scores[seed], key=lambda x: x[1])
            ],
        }
        save_json(os.path.join(OUT_DIR, f"scores_seed{seed}.json"), dump)

        print(f"\n[SEED {seed}] mean-over-cfg OOF logloss (all rows): {ll_seed:.6f}")
        print("Saved:", sub_path)

    # ---- per-cfg mean over seeds ----
    for cfg in LGBM_CONFIGS:
        cfg_name = cfg["name"]
        oof_cfg = np.mean(np.stack(per_cfg_oofs[cfg_name], axis=0), axis=0)
        test_cfg = np.mean(np.stack(per_cfg_tests[cfg_name], axis=0), axis=0)
        ll_cfg = float(log_loss(y, oof_cfg, labels=np.arange(n_classes)))

        np.save(os.path.join(OUT_DIR, f"oof_cfg_{cfg_name}.npy"), oof_cfg)
        np.save(os.path.join(OUT_DIR, f"test_cfg_{cfg_name}.npy"), test_cfg)

        sub_path = os.path.join(OUT_DIR, f"submission_cfg_{cfg_name}.csv")
        make_submission(test_cfg, classes, test_ids, sub_path)

        dump = {
            "cfg": cfg_name,
            "cfg_ensemble_oof_logloss_allrows": ll_cfg,
            "seed_scores_nonrare_sorted": [
                {"seed": int(s), "cv_logloss_nonrare": sc}
                for (s, sc) in sorted(per_cfg_scores[cfg_name], key=lambda x: x[1])
            ],
            "cfg_params": {k: v for k, v in cfg.items() if k != "name"},
        }
        save_json(os.path.join(OUT_DIR, f"scores_cfg_{cfg_name}.json"), dump)

        print(f"\n[CFG {cfg_name}] mean-over-seeds OOF logloss (all rows): {ll_cfg:.6f}")
        print("Saved:", sub_path)

    # ---- optional global mean (cfg×seed) ----
    if SAVE_GLOBAL_MEAN:
        all_oof = []
        all_test = []
        for cfg in LGBM_CONFIGS:
            for seed in SEEDS:
                mdir = os.path.join(OUT_DIR, f"{cfg['name']}_seed{seed}")
                all_oof.append(np.load(os.path.join(mdir, "oof.npy")))
                all_test.append(np.load(os.path.join(mdir, "test.npy")))
        oof_mean = np.mean(np.stack(all_oof, axis=0), axis=0)
        test_mean = np.mean(np.stack(all_test, axis=0), axis=0)
        ll_mean = float(log_loss(y, oof_mean, labels=np.arange(n_classes)))

        np.save(os.path.join(OUT_DIR, "oof_mean_all.npy"), oof_mean)
        np.save(os.path.join(OUT_DIR, "test_mean_all.npy"), test_mean)

        sub_path = os.path.join(OUT_DIR, "submission_mean_all.csv")
        make_submission(test_mean, classes, test_ids, sub_path)
        save_json(os.path.join(OUT_DIR, "scores_mean_all.json"), {"oof_logloss_allrows": ll_mean})

        print("\n[GLOBAL MEAN] OOF logloss:", ll_mean)
        print("Saved:", sub_path)

    save_json(os.path.join(OUT_DIR, "models_summary.json"), {"models": all_models})
    print("\nDone. Artifacts in:", OUT_DIR)


if __name__ == "__main__":
    main()
