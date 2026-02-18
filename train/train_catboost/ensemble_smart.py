# train/ensemble_smart.py
# "Smart" ensemble for your CatBoost multi-run outputs:
# 1) Optimized convex weights on OOF (non-negative, sum=1) via Dirichlet random search + local jitter
# 2) Optional stacking (multinomial logistic regression) on OOF logits
#
# Expects folder like:
#   result/
#     label_mapping.csv
#     <model_name_1>/oof.npy, test.npy, meta.json
#     <model_name_2>/...
#
# Produces:
#   result/ensemble_smart/
#     weights.json  (for weight method)
#     oof_weighted.npy / test_weighted.npy / submission_weighted.csv
#     stack_meta.joblib (for stack method)
#     oof_stacked.npy / test_stacked.npy / submission_stacked.csv

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import joblib


# ---- adjust if needed ----
DATA_DIR = "../../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def safe_clip_proba(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    p = p / np.sum(p, axis=1, keepdims=True)
    return p


def list_models(result_dir: str) -> List[str]:
    # model dirs are those that contain oof.npy and test.npy
    dirs = []
    for d in sorted(glob(os.path.join(result_dir, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "oof.npy")) and os.path.exists(os.path.join(d, "test.npy")):
            dirs.append(d)
    if not dirs:
        raise FileNotFoundError(f"No model dirs with oof.npy/test.npy found in: {result_dir}")
    return dirs


def load_mapping(result_dir: str) -> List[str]:
    path = os.path.join(result_dir, "label_mapping.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run your training script first.")
    classes = pd.read_csv(path)["label"].astype(str).tolist()
    # sanity
    extra = sorted(set(classes) - set(REQUIRED))
    missing = sorted(set(REQUIRED) - set(classes))
    if extra or missing:
        raise ValueError(f"Label mismatch vs REQUIRED. extra={extra}, missing={missing}")
    return classes


def load_y(result_dir: str, classes: List[str]) -> np.ndarray:
    y_raw = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")[TARGET_COL].astype(str).values
    label_to_idx = {c: i for i, c in enumerate(classes)}
    unknown = sorted(set(y_raw) - set(label_to_idx))
    if unknown:
        raise ValueError(f"Train labels contain unknown classes not in label_mapping.csv: {unknown}")
    y = np.array([label_to_idx[c] for c in y_raw], dtype=np.int64)
    return y


def load_test_ids() -> pd.DataFrame:
    return pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")[[ID_COL]].reset_index(drop=True)


def load_preds(model_dirs: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    oofs, tests, names, metas = [], [], [], []
    for d in model_dirs:
        name = os.path.basename(d.rstrip("/"))
        oof = np.load(os.path.join(d, "oof.npy"))
        test = np.load(os.path.join(d, "test.npy"))
        meta_path = os.path.join(d, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        oofs.append(oof.astype(np.float64))
        tests.append(test.astype(np.float64))
        names.append(name)
        metas.append(meta)
    # stack -> (M, N, C)
    oof_stack = np.stack(oofs, axis=0)
    test_stack = np.stack(tests, axis=0)
    return oof_stack, test_stack, names, metas


def weighted_ensemble(p_stack: np.ndarray, w: np.ndarray) -> np.ndarray:
    # p_stack: (M, N, C), w: (M,)
    p = np.tensordot(w, p_stack, axes=(0, 0))  # (N, C)
    return safe_clip_proba(p)


def init_weights(names: List[str], metas: List[Dict]) -> np.ndarray:
    # try use meta cv_logloss if present; else uniform
    losses = []
    ok = True
    for m in metas:
        if isinstance(m, dict) and "cv_logloss" in m and m["cv_logloss"] is not None:
            losses.append(float(m["cv_logloss"]))
        else:
            ok = False
            break
    if not ok or len(losses) != len(names):
        return np.ones(len(names), dtype=np.float64) / len(names)

    # better models -> bigger weight: use softmax(-loss / temp)
    losses = np.array(losses, dtype=np.float64)
    temp = max(1e-6, float(np.std(losses)) + 1e-6)
    w = softmax(-(losses - losses.min()) / temp)
    w = np.maximum(w, 1e-12)
    w = w / w.sum()
    return w


def search_weights(
    oof_stack: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    n_iter: int = 20000,
    alpha: float = 0.4,
    seed: int = 42,
    jitter_rounds: int = 4000,
    jitter_scale: float = 0.08,
) -> Tuple[np.ndarray, float]:
    """
    Dirichlet search around:
      - global random dirichlet (alpha)
      - plus local jitter around current best
    """
    rng = np.random.default_rng(seed)
    M = oof_stack.shape[0]

    best_w = w0.copy()
    best_p = weighted_ensemble(oof_stack, best_w)
    best_ll = float(log_loss(y, best_p, labels=np.arange(best_p.shape[1])))

    # global dirichlet search
    for i in range(n_iter):
        w = rng.dirichlet(np.ones(M) * alpha)
        p = weighted_ensemble(oof_stack, w)
        ll = float(log_loss(y, p, labels=np.arange(p.shape[1])))
        if ll < best_ll:
            best_ll, best_w = ll, w

    # local jitter refinement (multiplicative noise)
    for i in range(jitter_rounds):
        noise = rng.normal(0.0, jitter_scale, size=M)
        w = best_w * np.exp(noise)
        w = np.maximum(w, 1e-12)
        w = w / w.sum()
        p = weighted_ensemble(oof_stack, w)
        ll = float(log_loss(y, p, labels=np.arange(p.shape[1])))
        if ll < best_ll:
            best_ll, best_w = ll, w

    return best_w, best_ll


def save_submission(out_dir: str, filename: str, test_proba: np.ndarray, classes: List[str]):
    os.makedirs(out_dir, exist_ok=True)
    test_ids = load_test_ids()

    proba_df = pd.DataFrame(test_proba, columns=classes)[REQUIRED]
    sub = pd.concat([test_ids, proba_df.reset_index(drop=True)], axis=1)
    sub.to_csv(os.path.join(out_dir, filename), index=False)


def do_weight_ensemble(result_dir: str, out_dir: str, n_iter: int, seed: int):
    classes = load_mapping(result_dir)
    y = load_y(result_dir, classes)

    model_dirs = list_models(result_dir)
    oof_stack, test_stack, names, metas = load_preds(model_dirs)

    # sanity shapes
    M, N, C = oof_stack.shape
    if C != len(classes):
        raise ValueError(f"Class count mismatch: preds C={C}, mapping={len(classes)}")
    if len(y) != N:
        raise ValueError(f"Row mismatch: y={len(y)} oof={N}")

    # baselines
    w_uniform = np.ones(M) / M
    ll_uniform = log_loss(y, weighted_ensemble(oof_stack, w_uniform), labels=np.arange(C))

    w0 = init_weights(names, metas)
    ll_w0 = log_loss(y, weighted_ensemble(oof_stack, w0), labels=np.arange(C))

    print("\n=== WEIGHTED ENSEMBLE ===")
    print(f"models: {M}  rows: {N}  classes: {C}")
    print(f"uniform OOF logloss: {ll_uniform:.6f}")
    print(f"init    OOF logloss: {ll_w0:.6f}")

    best_w, best_ll = search_weights(
        oof_stack=oof_stack,
        y=y,
        w0=w0,
        n_iter=n_iter,
        alpha=0.35,
        seed=seed,
        jitter_rounds=max(2000, n_iter // 5),
        jitter_scale=0.07,
    )

    print(f"BEST    OOF logloss: {best_ll:.6f}\n")
    print("weights:")
    for name, w in sorted(zip(names, best_w), key=lambda x: -x[1]):
        print(f"  {w:.5f}  {name}")

    # build final
    oof_best = weighted_ensemble(oof_stack, best_w)
    test_best = weighted_ensemble(test_stack, best_w)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "oof_weighted.npy"), oof_best.astype(np.float32))
    np.save(os.path.join(out_dir, "test_weighted.npy"), test_best.astype(np.float32))

    with open(os.path.join(out_dir, "weights.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "optimized_convex_weights",
                "best_oof_logloss": float(best_ll),
                "n_iter": int(n_iter),
                "seed": int(seed),
                "models": [{"name": n, "weight": float(w)} for n, w in zip(names, best_w)],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    save_submission(out_dir, "submission_weighted.csv", test_best, classes)
    print(f"Saved: {out_dir}/submission_weighted.csv")


def logits_from_proba(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    # log(p) is fine as features; multinomial LR will learn relative weights.
    p = np.clip(p, eps, 1 - eps)
    return np.log(p)


def do_stacking(result_dir: str, out_dir: str, seed: int):
    """
    Stacking on OOF with multinomial logistic regression.
    NOTE: This can overfit a bit; we use CV inside stacking to generate stacked OOF cleanly.
    """
    classes = load_mapping(result_dir)
    y = load_y(result_dir, classes)

    model_dirs = list_models(result_dir)
    oof_stack, test_stack, names, metas = load_preds(model_dirs)

    M, N, C = oof_stack.shape
    print("\n=== STACKING ENSEMBLE ===")
    print(f"models: {M}  rows: {N}  classes: {C}")

    X_all = logits_from_proba(oof_stack).transpose(1, 0, 2).reshape(N, M * C)
    X_test_all = logits_from_proba(test_stack).transpose(1, 0, 2).reshape(test_stack.shape[1], M * C)

    # inner CV to get honest stacked OOF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_meta = np.zeros((N, C), dtype=np.float64)

    # a pretty sane regularization
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=0.5,
        max_iter=2000,
        n_jobs=-1,
    )

    for fold, (tr, va) in enumerate(skf.split(X_all, y), 1):
        clf_fold = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=0.5,
            max_iter=2000,
            n_jobs=-1,
        )
        clf_fold.fit(X_all[tr], y[tr])
        p_va = clf_fold.predict_proba(X_all[va])
        oof_meta[va] = p_va
        ll = log_loss(y[va], p_va, labels=np.arange(C))
        print(f"[stack fold {fold}] logloss={ll:.6f}")

    ll_oof = log_loss(y, oof_meta, labels=np.arange(C))
    print(f"STACK OOF logloss: {ll_oof:.6f}")

    # fit final stacker on full data, predict test
    clf.fit(X_all, y)
    test_meta = clf.predict_proba(X_test_all)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "oof_stacked.npy"), safe_clip_proba(oof_meta).astype(np.float32))
    np.save(os.path.join(out_dir, "test_stacked.npy"), safe_clip_proba(test_meta).astype(np.float32))
    joblib.dump(
        {"model": clf, "names": names, "classes": classes, "seed": seed},
        os.path.join(out_dir, "stack_meta.joblib"),
    )

    save_submission(out_dir, "submission_stacked.csv", test_meta, classes)
    print(f"Saved: {out_dir}/submission_stacked.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", type=str, default="result", help="Folder with label_mapping.csv and model subfolders")
    ap.add_argument("--out_dir", type=str, default="result/ensemble_smart", help="Where to write outputs")
    ap.add_argument("--method", type=str, default="weights", choices=["weights", "stack"], help="Ensembling method")
    ap.add_argument("--n_iter", type=int, default=25000, help="Dirichlet samples for weight search (weights method)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = ap.parse_args()

    if args.method == "weights":
        do_weight_ensemble(args.result_dir, args.out_dir, n_iter=args.n_iter, seed=args.seed)
    else:
        do_stacking(args.result_dir, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
