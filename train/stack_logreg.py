# train/stack_blend_auto.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "../data/processed"
ID_COL = "track_id"
TARGET_COL = "bird_group"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

BASE_MODELS = [
    ("cat", "out/result12(5266) - overfit/oof_proba_cat_temp.npy", "out/result12(5266) - overfit/test_proba_cat_temp.npy", "out/result12(5266) - overfit/label_mapping_cat_temp.csv"),
    ("lgb", "out/result12(5266) - overfit/oof_proba_lgbm_weighted.npy", "out/result12(5266) - overfit/test_proba_lgbm_weighted.npy", "out/result12(5266) - overfit/label_mapping_lgbm.csv"),

]

SEED = 42
N_SPLITS_META = 5

# blending search budgets (можешь поднять)
COARSE_STEPS = 21              # 0..1 step for coarse grid
RANDOM_TRIES = 20000           # random simplex samples

EPS = 1e-7


# -----------------------
# UTIL
# -----------------------
def clip_norm(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return p / s


def load_y_required() -> np.ndarray:
    y_raw = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")[TARGET_COL].astype(str).values
    missing = sorted(set(y_raw) - set(REQUIRED))
    if missing:
        raise ValueError(f"Train labels contain unknown classes not in REQUIRED: {missing}")
    label_to_idx = {c: i for i, c in enumerate(REQUIRED)}
    return np.array([label_to_idx[c] for c in y_raw], dtype=np.int64)


def align_proba(proba: np.ndarray, mapping_csv: str) -> np.ndarray:
    classes = pd.read_csv(mapping_csv)["label"].astype(str).tolist()
    df = pd.DataFrame(proba, columns=classes)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"{mapping_csv} missing classes: {miss}")
    return df[REQUIRED].to_numpy(dtype=np.float64)


@dataclass
class ModelProba:
    name: str
    oof: np.ndarray
    test: np.ndarray


def load_models() -> List[ModelProba]:
    models: List[ModelProba] = []
    for name, oof_path, test_path, mapping_csv in BASE_MODELS:
        if not os.path.exists(oof_path):
            raise FileNotFoundError(oof_path)
        if not os.path.exists(test_path):
            raise FileNotFoundError(test_path)
        if not os.path.exists(mapping_csv):
            raise FileNotFoundError(mapping_csv)

        oof = np.load(oof_path)
        test = np.load(test_path)

        oof = clip_norm(align_proba(oof, mapping_csv))
        test = clip_norm(align_proba(test, mapping_csv))

        print(f"[{name}] oof={oof.shape} test={test.shape}")
        models.append(ModelProba(name=name, oof=oof, test=test))

    # sanity checks
    n = models[0].oof.shape[0]
    m = models[0].test.shape[0]
    k = models[0].oof.shape[1]
    for mm in models:
        if mm.oof.shape != (n, k):
            raise ValueError(f"OOF shape mismatch for {mm.name}: {mm.oof.shape} expected {(n,k)}")
        if mm.test.shape != (m, k):
            raise ValueError(f"TEST shape mismatch for {mm.name}: {mm.test.shape} expected {(m,k)}")
    return models


def evaluate_models(y: np.ndarray, models: List[ModelProba]) -> Dict[str, float]:
    out = {}
    for mm in models:
        ll = log_loss(y, mm.oof, labels=np.arange(len(REQUIRED)))
        out[mm.name] = float(ll)
    return out


def blend(models: List[ModelProba], w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (oof_blend, test_blend). w must sum to 1."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    oof = np.zeros_like(models[0].oof)
    test = np.zeros_like(models[0].test)
    for i, mm in enumerate(models):
        oof += w[i] * mm.oof
        test += w[i] * mm.test
    return clip_norm(oof), clip_norm(test)


def coarse_grid_best(y: np.ndarray, models: List[ModelProba], steps: int = COARSE_STEPS):
    n = len(models)
    if n == 1:
        return np.array([1.0]), log_loss(y, models[0].oof, labels=np.arange(len(REQUIRED)))

    best_ll = 1e9
    best_w = None

    grid = np.linspace(0.0, 1.0, steps)

    if n == 2:
        for a in grid:
            w = np.array([a, 1 - a])
            oof, _ = blend(models, w)
            ll = log_loss(y, oof, labels=np.arange(len(REQUIRED)))
            if ll < best_ll:
                best_ll, best_w = ll, w
        return best_w, best_ll

    if n == 3:
        for a in grid:
            for b in grid:
                if a + b > 1:
                    continue
                c = 1 - a - b
                w = np.array([a, b, c])
                oof, _ = blend(models, w)
                ll = log_loss(y, oof, labels=np.arange(len(REQUIRED)))
                if ll < best_ll:
                    best_ll, best_w = ll, w
        return best_w, best_ll

    # n>3: fallback random (coarse grid explodes)
    return None, None


def random_simplex_best(y: np.ndarray, models: List[ModelProba], tries: int = RANDOM_TRIES, seed: int = SEED,
                        init_w: np.ndarray | None = None):
    rng = np.random.default_rng(seed)
    best_ll = 1e9
    best_w = None

    # try init
    if init_w is not None:
        oof, _ = blend(models, init_w)
        ll = log_loss(y, oof, labels=np.arange(len(REQUIRED)))
        best_ll, best_w = ll, init_w / init_w.sum()

    n = len(models)

    # Dirichlet samples (sum=1)
    alpha = np.ones(n, dtype=np.float64)
    for _ in range(tries):
        w = rng.dirichlet(alpha)
        oof, _ = blend(models, w)
        ll = log_loss(y, oof, labels=np.arange(len(REQUIRED)))
        if ll < best_ll:
            best_ll, best_w = ll, w

    return best_w, float(best_ll)


def make_meta_X(models: List[ModelProba]) -> Tuple[np.ndarray, np.ndarray]:
    X_oof = np.concatenate([m.oof for m in models], axis=1)   # (n_train, n_models*k)
    X_test = np.concatenate([m.test for m in models], axis=1) # (n_test,  n_models*k)
    return X_oof, X_test


def meta_logreg_cv(y: np.ndarray, X_oof: np.ndarray, X_test: np.ndarray, n_splits: int = N_SPLITS_META, seed: int = SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best = {"logloss": 1e9, "C": None, "meta_oof": None, "meta_test": None}

    for C in [0.2, 0.5, 1.0, 2.0, 5.0]:
        meta_oof = np.zeros((X_oof.shape[0], len(REQUIRED)), dtype=np.float64)

        for f, (tr, va) in enumerate(skf.split(X_oof, y), 1):
            meta = LogisticRegression(
                max_iter=5000,
                solver="lbfgs",
                C=C,
            )
            meta.fit(X_oof[tr], y[tr])
            meta_oof[va] = meta.predict_proba(X_oof[va])

        meta_oof = clip_norm(meta_oof)
        ll = log_loss(y, meta_oof, labels=np.arange(len(REQUIRED)))
        print(f"[meta logreg] C={C} OOF logloss={ll:.6f}")

        if ll < best["logloss"]:
            # refit on full
            meta_full = LogisticRegression(max_iter=5000, solver="lbfgs", C=C)
            meta_full.fit(X_oof, y)
            meta_test = clip_norm(meta_full.predict_proba(X_test))

            best.update({"logloss": float(ll), "C": C, "meta_oof": meta_oof, "meta_test": meta_test})

    return best


def meta_hgb_cv(y: np.ndarray, X_oof: np.ndarray, X_test: np.ndarray, n_splits: int = N_SPLITS_META, seed: int = SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best = {"logloss": 1e9, "params": None, "meta_oof": None, "meta_test": None}

    # небольшой safe grid
    param_grid = [
        dict(max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=0.0),
        dict(max_depth=3, learning_rate=0.05, max_iter=600, l2_regularization=0.0),
        dict(max_depth=4, learning_rate=0.05, max_iter=400, l2_regularization=0.0),
        dict(max_depth=3, learning_rate=0.03, max_iter=800, l2_regularization=0.0),
        dict(max_depth=3, learning_rate=0.05, max_iter=600, l2_regularization=1.0),
    ]

    for params in param_grid:
        meta_oof = np.zeros((X_oof.shape[0], len(REQUIRED)), dtype=np.float64)

        for f, (tr, va) in enumerate(skf.split(X_oof, y), 1):
            meta = HistGradientBoostingClassifier(
                random_state=seed,
                loss="log_loss",
                **params
            )
            meta.fit(X_oof[tr], y[tr])
            meta_oof[va] = meta.predict_proba(X_oof[va])

        meta_oof = clip_norm(meta_oof)
        ll = log_loss(y, meta_oof, labels=np.arange(len(REQUIRED)))
        print(f"[meta hgb] {params} OOF logloss={ll:.6f}")

        if ll < best["logloss"]:
            meta_full = HistGradientBoostingClassifier(
                random_state=seed,
                loss="log_loss",
                **params
            )
            meta_full.fit(X_oof, y)
            meta_test = clip_norm(meta_full.predict_proba(X_test))
            best.update({"logloss": float(ll), "params": params, "meta_oof": meta_oof, "meta_test": meta_test})

    return best


def save_outputs(test_ids: pd.DataFrame, test_proba: np.ndarray, prefix: str):
    np.save(f"{prefix}_test_proba.npy", test_proba)

    sub_proba = pd.concat([test_ids.reset_index(drop=True),
                           pd.DataFrame(test_proba, columns=REQUIRED)], axis=1)
    sub_proba.to_csv(f"{prefix}_submission_proba.csv", index=False)

    pred_idx = test_proba.argmax(axis=1)
    pred_label = [REQUIRED[i] for i in pred_idx]
    sub_label = test_ids.copy()
    sub_label["bird_group"] = pred_label
    sub_label.to_csv(f"{prefix}_submission_label.csv", index=False)


def main():
    y = load_y_required()
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")[[ID_COL]]

    models = load_models()
    per_model = evaluate_models(y, models)
    print("\n=== BASE MODELS OOF LOGLOSS ===")
    for k, v in per_model.items():
        print(f"{k:>6}: {v:.6f}")

    report = {"base_oof": per_model}

    # -----------------------
    # BLENDING (best for non-synced OOF)
    # -----------------------
    print("\n=== BLEND SEARCH ===")
    w0, ll0 = coarse_grid_best(y, models, steps=COARSE_STEPS)
    if w0 is not None:
        print(f"[coarse grid] best logloss={ll0:.6f} w={w0}")
    w_best, ll_best = random_simplex_best(y, models, tries=RANDOM_TRIES, seed=SEED, init_w=w0)
    oof_blend, test_blend = blend(models, w_best)
    ll_blend = log_loss(y, oof_blend, labels=np.arange(len(REQUIRED)))
    print(f"[random simplex] best logloss={ll_blend:.6f} w={w_best}")

    report["blend"] = {"oof_logloss": float(ll_blend), "weights": [float(x) for x in w_best]}

    # -----------------------
    # STACKING (может быть лучше, но иногда хуже на несинхронных OOF)
    # -----------------------
    print("\n=== STACKING ===")
    X_oof, X_test = make_meta_X(models)

    lr_res = meta_logreg_cv(y, X_oof, X_test)
    report["stack_logreg"] = {"oof_logloss": lr_res["logloss"], "C": lr_res["C"]}

    hgb_res = meta_hgb_cv(y, X_oof, X_test)
    report["stack_hgb"] = {"oof_logloss": hgb_res["logloss"], "params": hgb_res["params"]}

    # -----------------------
    # PICK BEST BY OOF
    # -----------------------
    candidates = [
        ("blend", ll_blend, test_blend),
        ("stack_logreg", lr_res["logloss"], lr_res["meta_test"]),
        ("stack_hgb", hgb_res["logloss"], hgb_res["meta_test"]),
    ]
    best_name, best_ll, best_test = sorted(candidates, key=lambda x: x[1])[0]

    print("\n=== BEST METHOD ===")
    print(f"{best_name} with OOF logloss={best_ll:.6f}")

    report["best"] = {"method": best_name, "oof_logloss": float(best_ll)}

    # save outputs
    prefix = "best"
    save_outputs(test_ids, best_test, prefix=prefix)

    with open("out/result12(5266) - overfit/report_best.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(" - best_test_proba.npy")
    print(" - best_submission_proba_stack.csv")
    print(" - best_submission_label.csv")
    print(" - report_best.json")


if __name__ == "__main__":
    main()
