# train/tune_lgbm_seed_weights.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

DATA_DIR = "../../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

SEEDS = [1, 42, 1337, 2026, 777]
SEED_DIR = "../out/result12(5266)/lgbm_seeds"

EPS = 1e-15


def load_y_and_classes():
    y_raw = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")[TARGET_COL].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    classes = le.classes_.tolist()
    return y, classes


def load_seed_arrays(seeds):
    oofs, tests = [], []
    for s in seeds:
        oof_path = os.path.join(SEED_DIR, f"oof_proba_lgbm_seed{s}.npy")
        test_path = os.path.join(SEED_DIR, f"test_proba_lgbm_seed{s}.npy")
        if not os.path.exists(oof_path):
            raise FileNotFoundError(f"Missing {oof_path} (run train_lgbm_ensemble_strong.py first)")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing {test_path} (run train_lgbm_ensemble_strong.py first)")
        oofs.append(np.load(oof_path))
        tests.append(np.load(test_path))
    return np.stack(oofs, axis=0), np.stack(tests, axis=0)


def blend(probas: np.ndarray, w: np.ndarray) -> np.ndarray:
    p = np.tensordot(w, probas, axes=(0, 0))  # (n, k)
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def objective(w: np.ndarray, oofs: np.ndarray, y: np.ndarray) -> float:
    p = blend(oofs, w)
    return log_loss(y, p, labels=np.arange(p.shape[1]))


def random_dirichlet_search(oofs, y, n_iter=40000, seed=42):
    rng = np.random.default_rng(seed)
    m = oofs.shape[0]
    best_w = np.ones(m) / m
    best = objective(best_w, oofs, y)

    for i in range(n_iter):
        w = rng.dirichlet(np.ones(m))
        val = objective(w, oofs, y)
        if val < best:
            best, best_w = val, w
            if i % 200 == 0:
                print(f"[random] iter={i} best_logloss={best:.6f} w={np.round(best_w, 4)}")
    return best_w, best


def scipy_simplex_opt(oofs, y):
    try:
        from scipy.optimize import minimize
    except Exception:
        return None

    m = oofs.shape[0]
    x0 = np.ones(m) / m

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * m

    res = minimize(
        fun=lambda w: objective(w, oofs, y),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-12, "disp": True},
    )
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum()
    best = objective(w, oofs, y)
    return w, best


def main():
    y, classes = load_y_and_classes()
    oofs, tests = load_seed_arrays(SEEDS)

    m, n, k = oofs.shape
    print(f"Loaded: seeds={m} train={n} classes={k}")

    w_eq = np.ones(m) / m
    base = objective(w_eq, oofs, y)
    print(f"Equal-weights logloss: {base:.6f}")

    best_w = None
    best_ll = None

    got = scipy_simplex_opt(oofs, y)
    if got is not None:
        best_w, best_ll = got
        print(f"[scipy] best logloss: {best_ll:.6f} w={np.round(best_w, 6)}")
    else:
        print("[scipy] SciPy not found, using random search")

    w_r, ll_r = random_dirichlet_search(oofs, y, n_iter=40000, seed=42)
    print(f"[random] best logloss: {ll_r:.6f} w={np.round(w_r, 6)}")

    if best_w is None or ll_r < best_ll:
        best_w, best_ll = w_r, ll_r

    print("\n========== BEST LGBM SEED WEIGHTS ==========")
    for s, w in zip(SEEDS, best_w):
        print(f"seed {s}: {w:.6f}")
    print(f"best_logloss: {best_ll:.6f}")

    pd.DataFrame({"seed": SEEDS, "weight": best_w}).to_csv("result_weight/weights_lgbm_seeds.csv", index=False)
    print("Saved weights_lgbm_seeds.csv")

    oof_best = blend(oofs, best_w)
    test_best = blend(tests, best_w)

    np.save("result_weight/oof_proba_lgbm_weighted.npy", oof_best)
    np.save("result_weight/test_proba_lgbm_weighted.npy", test_best)

    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")[[ID_COL]]
    sub_proba = pd.DataFrame(test_best, columns=classes)
    sub_proba.insert(0, ID_COL, test_ids[ID_COL].values)
    sub_proba.to_csv("submission_lgbm_weighted_proba.csv", index=False)

    print("Saved submission_lgbm_weighted_proba.csv")


if __name__ == "__main__":
    main()
