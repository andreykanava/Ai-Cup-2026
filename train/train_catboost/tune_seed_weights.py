# train/tune_seed_weights.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

DATA_DIR = "../../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

# Должно совпадать с train_catboost_seed_ensemble.py
SEEDS = [42, 1337, 2024, 777, 999]

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

EPS = 1e-15  # чтобы log_loss не умирал на нулях


def load_y_and_classes() -> tuple[np.ndarray, np.ndarray]:
    y_raw = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")[TARGET_COL].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    return y, classes


def load_seed_arrays(seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    oofs = []
    tests = []
    for s in seeds:
        oof_path = f"oof_seed_{s}.npy"
        test_path = f"test_seed_{s}.npy"
        if not os.path.exists(oof_path):
            raise FileNotFoundError(f"Missing {oof_path}. Add np.save in your trainer.")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing {test_path}. Add np.save in your trainer.")
        oofs.append(np.load(oof_path))
        tests.append(np.load(test_path))
    # shape: (n_models, n_samples, n_classes)
    return np.stack(oofs, axis=0), np.stack(tests, axis=0)


def blend(probas: np.ndarray, w: np.ndarray) -> np.ndarray:
    # probas: (m, n, k), w: (m,)
    p = np.tensordot(w, probas, axes=(0, 0))  # -> (n, k)
    # safety clip + renorm
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def objective(w: np.ndarray, oofs: np.ndarray, y: np.ndarray) -> float:
    p = blend(oofs, w)
    return log_loss(y, p, labels=np.arange(p.shape[1]))


def random_dirichlet_search(oofs: np.ndarray, y: np.ndarray, n_iter: int = 30000, seed: int = 42):
    rng = np.random.default_rng(seed)
    m = oofs.shape[0]
    best_w = np.ones(m) / m
    best = objective(best_w, oofs, y)

    for i in range(n_iter):
        # Dirichlet: все веса >=0 и сумма=1
        w = rng.dirichlet(np.ones(m))
        val = objective(w, oofs, y)
        if val < best:
            best = val
            best_w = w
            # чуть-чуть логов
            if i % 200 == 0:
                print(f"[random] iter={i} best_logloss={best:.6f} w={np.round(best_w, 4)}")

    return best_w, best


def scipy_simplex_opt(oofs: np.ndarray, y: np.ndarray):
    try:
        from scipy.optimize import minimize
    except Exception:
        return None

    m = oofs.shape[0]
    x0 = np.ones(m) / m

    # constraints: sum(w)=1, w>=0
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * m

    res = minimize(
        fun=lambda w: objective(w, oofs, y),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-12, "disp": True},
    )
    if not res.success:
        print("[scipy] Optimization failed:", res.message)
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum()
    best = objective(w, oofs, y)
    return w, best


def main():
    y, classes = load_y_and_classes()
    oofs, tests = load_seed_arrays(SEEDS)

    m, n, k = oofs.shape
    print(f"Loaded: models={m} train={n} classes={k}")

    # baseline equal weights
    w_eq = np.ones(m) / m
    base = objective(w_eq, oofs, y)
    print(f"Equal-weights logloss: {base:.6f}")

    # 1) SciPy opt (если есть)
    best_w = None
    best_ll = None
    got = scipy_simplex_opt(oofs, y)
    if got is not None:
        best_w, best_ll = got
        print(f"[scipy] best logloss: {best_ll:.6f} w={np.round(best_w, 6)}")
    else:
        print("[scipy] not available, using random search only")

    # 2) Random Dirichlet search (как страховка / улучшение)
    w_r, ll_r = random_dirichlet_search(oofs, y, n_iter=30000, seed=42)
    print(f"[random] best logloss: {ll_r:.6f} w={np.round(w_r, 6)}")

    # выберем лучшее
    if best_w is None or ll_r < best_ll:
        best_w, best_ll = w_r, ll_r

    print("\n========== BEST WEIGHTS ==========")
    for s, w in zip(SEEDS, best_w):
        print(f"seed {s}: {w:.6f}")
    print(f"best_logloss: {best_ll:.6f}")

    # save weights
    pd.DataFrame({"seed": SEEDS, "weight": best_w}).to_csv("weights_seed_ensemble.csv", index=False)
    print("Saved weights_seed_ensemble.csv")

    # blended outputs
    oof_best = blend(oofs, best_w)
    test_best = blend(tests, best_w)

    np.save("oof_proba_cat_weighted.npy", oof_best)
    np.save("test_proba_cat_weighted.npy", test_best)

    # submission
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")[[ID_COL]]

    proba_df = pd.DataFrame(test_best, columns=classes)
    # sanity: REQUIRED must exist
    missing = sorted(set(REQUIRED) - set(classes))
    if missing:
        raise ValueError(f"Missing REQUIRED classes in model outputs: {missing}")

    proba_df = proba_df[REQUIRED]
    sub = pd.concat([test_ids.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
    sub.to_csv("submission_cat_weighted.csv", index=False)
    print("Saved submission_cat_weighted.csv")


if __name__ == "__main__":
    main()
