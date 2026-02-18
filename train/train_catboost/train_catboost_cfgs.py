# train/train_catboost_3cfg_x2seed.py
# Train CatBoost configs × seeds with StratifiedKFold OOF + Test proba saving.
# Saves per-model outputs AND per-seed ensemble outputs (mean over configs for that seed).
# NO global ensemble across all seeds (so you can blend later with your own weights script).

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from catboost import CatBoostClassifier


# -----------------------
# CONFIG (edit here)
# -----------------------
DATA_DIR = "../../data/processed/best"
TARGET_COL = "bird_group"
ID_COL = "track_id"

N_SPLITS = 5
USE_GPU = True

# feature drop list (optional). set to None to disable
FEATURES_TO_DROP_CSV = "../features_to_drop.csv"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

RUN_NAME = "cat_3cfg_x2seed"
OUT_DIR = os.path.join("out", RUN_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 1488, 228, 777]

CONFIGS: List[Dict] = [
    dict(
        name="B0_cpu_rsm085",
        task_type="CPU",
        iterations=9000,
        learning_rate=0.03,
        depth=6,
        rsm=0.85,
        l2_leaf_reg=15.0,
        min_data_in_leaf=40,
        bootstrap_type="Bayesian",
        bagging_temperature=0.3,
        random_strength=0.5,
        od_wait=250,
    ),

    # B1: CPU rsm чуть сильнее шум (ансабль любит)
    dict(
        name="B1_cpu_rsm080",
        task_type="CPU",
        iterations=11000,
        learning_rate=0.027,
        depth=6,
        rsm=0.80,
        l2_leaf_reg=18.0,
        min_data_in_leaf=45,
        bootstrap_type="Bayesian",
        bagging_temperature=0.45,
        random_strength=0.8,
        od_wait=350,
    ),

    # C1: CPU depth7 + rsm (максимально “иная” модель)
    dict(
        name="C1_cpu_d7_rsm090",
        task_type="CPU",
        iterations=12000,
        learning_rate=0.022,
        depth=7,
        rsm=0.90,
        l2_leaf_reg=24.0,
        min_data_in_leaf=50,
        bootstrap_type="Bayesian",
        bagging_temperature=0.4,
        random_strength=1.0,
        od_wait=450,
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
    print(f"Categorical columns: {len(cat_cols)}")

    return X_train, X_test, cat_idx


def ensure_required_labels(classes: np.ndarray):
    extra = sorted(set(classes) - set(REQUIRED))
    missing = sorted(set(REQUIRED) - set(classes))
    if extra or missing:
        raise ValueError(f"Label mismatch. extra={extra}, missing={missing}")


def save_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_model_outputs(run_dir: str, model_name: str, oof: np.ndarray, test: np.ndarray, meta: Dict):
    mdir = os.path.join(run_dir, model_name)
    os.makedirs(mdir, exist_ok=True)

    np.save(os.path.join(mdir, "oof.npy"), oof)
    np.save(os.path.join(mdir, "test.npy"), test)
    save_json(os.path.join(mdir, "meta.json"), meta)


def make_submission(test_proba: np.ndarray, classes: np.ndarray, test_ids: pd.DataFrame, out_csv: str):
    proba_df = pd.DataFrame(test_proba, columns=classes)[REQUIRED]
    sub = pd.concat(
        [test_ids[[ID_COL]].reset_index(drop=True), proba_df.reset_index(drop=True)],
        axis=1
    )
    sub.to_csv(out_csv, index=False)


# -----------------------
# TRAINING
# -----------------------
def build_params(cfg: Dict, seed: int) -> Dict:
    params = dict(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=int(cfg["iterations"]),
        learning_rate=float(cfg["learning_rate"]),
        depth=int(cfg["depth"]),
        l2_leaf_reg=float(cfg["l2_leaf_reg"]),
        min_data_in_leaf=int(cfg["min_data_in_leaf"]),
        bootstrap_type=str(cfg["bootstrap_type"]),
        bagging_temperature=float(cfg["bagging_temperature"]),
        random_strength=float(cfg["random_strength"]),
        random_seed=int(seed),
        od_type="Iter",
        od_wait=int(cfg["od_wait"]),
        task_type=str(cfg.get("task_type", "GPU" if USE_GPU else "CPU")),
        thread_count=-1,
        verbose=200,
    )

    # optional rsm
    if "rsm" in cfg and cfg["rsm"] is not None:
        params["rsm"] = float(cfg["rsm"])

    return params


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

    params = build_params(cfg, seed)

    for fold, (tr, va) in enumerate(skf.split(X_train, y), 1):
        X_tr, y_tr = X_train.iloc[tr], y[tr]
        X_va, y_va = X_train.iloc[va], y[va]

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
    pd.DataFrame({"label": classes}).to_csv(os.path.join(OUT_DIR, "label_mapping.csv"), index=False)

    # We'll save per-seed ensembles (mean over cfgs for that seed)
    per_seed_scores: Dict[int, Dict] = {}

    for seed in SEEDS:
        print(f"\n#############################")
        print(f"########## SEED {seed} ##########")
        print(f"#############################\n")

        seed_oof_list: List[np.ndarray] = []
        seed_test_list: List[np.ndarray] = []
        seed_model_scores: List[Tuple[str, float]] = []

        for cfg in CONFIGS:
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
                "use_gpu_default": USE_GPU,
                "task_type": str(cfg.get("task_type", "GPU" if USE_GPU else "CPU")),
                "n_features": int(X_train.shape[1]),
                "n_classes": int(n_classes),
            }

            save_model_outputs(OUT_DIR, model_name, oof, test, meta)

            seed_oof_list.append(oof)
            seed_test_list.append(test)
            seed_model_scores.append((model_name, float(score)))

        # ---- per-seed ensemble (mean over configs for THIS seed) ----
        oof_seed = np.mean(np.stack(seed_oof_list, axis=0), axis=0)
        test_seed = np.mean(np.stack(seed_test_list, axis=0), axis=0)

        ll_seed = float(log_loss(y, oof_seed, labels=np.arange(n_classes)))

        np.save(os.path.join(OUT_DIR, f"oof_seed{seed}.npy"), oof_seed)
        np.save(os.path.join(OUT_DIR, f"test_seed{seed}.npy"), test_seed)

        sub_path = os.path.join(OUT_DIR, f"submission_seed{seed}.csv")
        make_submission(test_seed, classes, test_ids, sub_path)

        score_dump = {
            "seed": seed,
            "seed_ensemble_oof_logloss": ll_seed,
            "models_sorted": [
                {"model": n, "cv_logloss": sc}
                for (n, sc) in sorted(seed_model_scores, key=lambda x: x[1])
            ],
        }
        save_json(os.path.join(OUT_DIR, f"scores_seed{seed}.json"), score_dump)

        per_seed_scores[seed] = score_dump

        print(f"\n========== SEED {seed} ENSEMBLE (mean over cfgs) ==========")
        for row in score_dump["models_sorted"]:
            print(f" - {row['model']}: {row['cv_logloss']:.6f}")
        print(f"seed ensemble OOF logloss: {ll_seed:.6f}")
        print("Saved:", sub_path)

    # summary json for all seeds
    save_json(os.path.join(OUT_DIR, "scores_all_seeds.json"), {"seeds": per_seed_scores})
    print("\nDone. Saved per-seed oof/test/submissions in:", OUT_DIR)


if __name__ == "__main__":
    main()
