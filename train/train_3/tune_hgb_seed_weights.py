# train/tune_hgb_seed_weights.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

DATA_DIR = "../../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

# Должно совпадать с твоим HGB-скриптом
SEEDS = [42, 1337, 2026, 7777]
SEED_DIR = "hgb_seeds"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

EPS = 1e-15


def load_y_and_classes():
    y_raw = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")[TARGET_COL].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    classes = le.classes_.astype(str)
    return y, classes


def load_seed_arrays(seeds):
    oofs, tests = [], []
    for s in seeds:
        oof_path = os.path.join(SEED_DIR, f"oof_proba_hgb_seed{s}.npy")
        test_path = os.path.join(SEED_DIR, f"test_proba_hgb_seed{s}.npy")
        if not os.path.exists(oof_path):
            raise FileNotFoundError(f"Missing {oof_path}. Save per-seed OOF in training first.")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing {test_path}. Save per-seed TEST in training first.")
        oofs.append(np.load(oof_path))
        tests.append(np.load(test_path))
    return np.stack(oofs, axis=0), np.stack(tests, axis=0)


def probs_to_logits(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return np.log(p)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    logits = probs_to_logits(proba)
    p = softmax(logits / T)
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def find_best_temperature(oof_proba: np.ndarray, y: np.ndarray, n_classes: int):
    best_T, best_ll = 1.0, 1e9
    grid = np.unique(np.round(np.concatenate([
        np.linspace(0.7, 1.6, 46),
        np.linspace(1.6, 3.0, 36),
    ]), 4))
    for T in grid:
        p = apply_temperature(oof_proba, float(T))
        ll = log_loss(y, p, labels=np.arange(n_classes))
        if ll < best_ll:
            best_ll = ll
            best_T = float(T)
    return best_T, best_ll


def blend(probas: np.ndarray, w: np.ndarray) -> np.ndarray:
    # probas: (m, n, k), w: (m,)
    p = np.tensordot(w, probas, axes=(0, 0))  # (n, k)
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def objective(w: np.ndarray, oofs: np.ndarray, y: np.ndarray) -> float:
    p = blend(oofs, w)
    return log_loss(y, p, labels=np.arange(p.shape[1]))


def random_dirichlet_search(oofs, y, n_iter=60000, seed=42):
    rng = np.random.default_rng(seed)
    m = oofs.shape[0]
    best_w = np.ones(m) / m
    best = objective(best_w, oofs, y)

    for i in range(n_iter):
        w = rng.dirichlet(np.ones(m))
        val = objective(w, oofs, y)
        if val < best:
            best, best_w = val, w
            if i % 500 == 0:
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
        options={"maxiter": 400, "ftol": 1e-12, "disp": True},
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

    # baseline equal
    w_eq = np.ones(m) / m
    base_ll = objective(w_eq, oofs, y)
    print(f"Equal weights logloss: {base_ll:.6f}")

    # SciPy opt (если есть)
    best_w, best_ll = None, None
    got = scipy_simplex_opt(oofs, y)
    if got is not None:
        best_w, best_ll = got
        print(f"[scipy] best logloss: {best_ll:.6f} w={np.round(best_w, 6)}")
    else:
        print("[scipy] SciPy not found -> skip SLSQP")

    # Random search (страховка и часто улучшает)
    w_r, ll_r = random_dirichlet_search(oofs, y, n_iter=60000, seed=42)
    print(f"[random] best logloss: {ll_r:.6f} w={np.round(w_r, 6)}")

    if best_w is None or ll_r < best_ll:
        best_w, best_ll = w_r, ll_r

    print("\n========== BEST HGB SEED WEIGHTS ==========")
    for s, w in zip(SEEDS, best_w):
        print(f"seed {s}: {w:.6f}")
    print(f"best_logloss_raw: {best_ll:.6f}")

    pd.DataFrame({"seed": SEEDS, "weight": best_w}).to_csv("weights_hgb_seeds.csv", index=False)
    print("Saved weights_hgb_seeds.csv")

    # Blend with best weights
    oof_w = blend(oofs, best_w)
    test_w = blend(tests, best_w)

    raw_ll = log_loss(y, oof_w, labels=np.arange(k))
    print(f"Weighted (raw) OOF logloss: {raw_ll:.6f}")

    # Temperature scaling on weighted OOF
    best_T, best_T_ll = find_best_temperature(oof_w, y, k)
    print(f"Best temperature: T={best_T:.4f} -> logloss={best_T_ll:.6f}")

    oof_ts = apply_temperature(oof_w, best_T).astype(np.float32)
    test_ts = apply_temperature(test_w, best_T).astype(np.float32)

    ts_ll = log_loss(y, oof_ts, labels=np.arange(k))
    print(f"Weighted + TS OOF logloss: {ts_ll:.6f}")

    # Save arrays
    np.save("result10(537)/oof_proba_hgb_weighted.npy", oof_w)
    np.save("result10(537)/test_proba_hgb_weighted.npy", test_w)
    np.save("result10(537)/oof_proba_hgb_weighted_ts.npy", oof_ts)
    np.save("result10(537)/test_proba_hgb_weighted_ts.npy", test_ts)

    # Submission (probabilities)
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")[[ID_COL]]

    # sanity REQUIRED
    missing = sorted(set(REQUIRED) - set(classes.tolist()))
    if missing:
        raise ValueError(f"Missing REQUIRED classes in model outputs: {missing}")

    proba_df = pd.DataFrame(test_ts, columns=classes)
    proba_df = proba_df[REQUIRED]

    s = proba_df.sum(axis=1).values
    print("proba sum check:", float(np.min(s)), float(np.max(s)), float(np.mean(s)))

    sub = pd.concat([test_ids.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
    sub.to_csv("submission_hgb_weighted_ts.csv", index=False)
    print("Saved submission_hgb_weighted_ts.csv")


if __name__ == "__main__":
    main()
