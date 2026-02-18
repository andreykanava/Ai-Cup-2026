# ensemble/fit_weights_many_and_submit.py
# Fits weights on OOF and (optionally) builds weighted submission from TEST probas.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

DATA_DIR = "../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

EPS = 1e-15


# -------------------------
# CONFIG: ADD MODELS HERE
# -------------------------
@dataclass
class ModelSpec:
    name: str
    oof_path: str
    test_path: Optional[str]  # set to None if you don't have it
    mapping_csv: str


MODELS: List[ModelSpec] = [
    ModelSpec(
        name="LGBM_cfg0_safe_reg",
        oof_path="out/result15/out/lgbm_6cfg_xseed/oof_cfg_cfg0_safe_reg.npy",
        test_path="out/result15/out/lgbm_6cfg_xseed/test_cfg_cfg0_safe_reg.npy",
        mapping_csv="out/result15/out/lgbm_6cfg_xseed/label_mapping_lgbm.csv",
    ),
    ModelSpec(
        name="LGBM_cfg1_capacity",
        oof_path="out/result15/out/lgbm_6cfg_xseed/oof_cfg_cfg1_capacity.npy",
        test_path="out/result15/out/lgbm_6cfg_xseed/test_cfg_cfg1_capacity.npy",
        mapping_csv="out/result15/out/lgbm_6cfg_xseed/label_mapping_lgbm.csv",
    ),
    ModelSpec(
        name="LGBM_cfg2_extratrees",
        oof_path="out/result15/out/lgbm_6cfg_xseed/oof_cfg_cfg2_extratrees.npy",
        test_path="out/result15/out/lgbm_6cfg_xseed/test_cfg_cfg2_extratrees.npy",
        mapping_csv="out/result15/out/lgbm_6cfg_xseed/label_mapping_lgbm.csv",
    ),
    ModelSpec(
        name="LGBM_cfg3_dart",
        oof_path="out/result15/out/lgbm_6cfg_xseed/oof_cfg_cfg3_dart.npy",
        test_path="out/result15/out/lgbm_6cfg_xseed/test_cfg_cfg3_dart.npy",
        mapping_csv="out/result15/out/lgbm_6cfg_xseed/label_mapping_lgbm.csv",
    ),
    ModelSpec(
        name="LGBM_cfg4_strong_reg",
        oof_path="out/result15/out/lgbm_6cfg_xseed/oof_cfg_cfg4_strong_reg.npy",
        test_path="out/result15/out/lgbm_6cfg_xseed/test_cfg_cfg4_strong_reg.npy",
        mapping_csv="out/result15/out/lgbm_6cfg_xseed/label_mapping_lgbm.csv",
    ),
    ModelSpec(
        name="LGBM_cfg5_leafy_stoch",
        oof_path="out/result15/out/lgbm_6cfg_xseed/oof_cfg_cfg5_leafy_stoch.npy",
        test_path="out/result15/out/lgbm_6cfg_xseed/test_cfg_cfg5_leafy_stoch.npy",
        mapping_csv="out/result15/out/lgbm_6cfg_xseed/label_mapping_lgbm.csv",
    ),

    # -----------------------------
    # LGBM seed ensemble outputs
    # -----------------------------
    ModelSpec(
        name="LGBM_seed214",
        oof_path="out/result15/out/lgbm_6cfg_xseed/oof_seed214.npy",
        test_path="out/result15/out/lgbm_6cfg_xseed/test_seed214.npy",
        mapping_csv="out/result15/out/lgbm_6cfg_xseed/label_mapping_lgbm.csv",
    ),
    ModelSpec(
        name="LGBM_seed235",
        oof_path="out/result15/out/lgbm_6cfg_xseed/oof_seed235.npy",
        test_path="out/result15/out/lgbm_6cfg_xseed/test_seed235.npy",
        mapping_csv="out/result15/out/lgbm_6cfg_xseed/label_mapping_lgbm.csv",
    ),
    ModelSpec(
        name="LGBM_seed463",
        oof_path="out/result15/out/lgbm_6cfg_xseed/oof_seed463.npy",
        test_path="out/result15/out/lgbm_6cfg_xseed/test_seed463.npy",
        mapping_csv="out/result15/out/lgbm_6cfg_xseed/label_mapping_lgbm.csv",
    ),
    ModelSpec(
        name="LGBM_seed743",
        oof_path="out/result15/out/lgbm_6cfg_xseed/oof_seed743.npy",
        test_path="out/result15/out/lgbm_6cfg_xseed/test_seed743.npy",
        mapping_csv="out/result15/out/lgbm_6cfg_xseed/label_mapping_lgbm.csv",
    ),
    ModelSpec(
        name="LGBM_seed756",
        oof_path="out/result15/out/lgbm_6cfg_xseed/oof_seed756.npy",
        test_path="out/result15/out/lgbm_6cfg_xseed/test_seed756.npy",
        mapping_csv="out/result15/out/lgbm_6cfg_xseed/label_mapping_lgbm.csv",
    ),

    ModelSpec(
        name="Cat_seed42",
        oof_path="out/result15/out/cat_3cfg_x2seed/oof_seed42.npy",
        test_path="out/result15/out/cat_3cfg_x2seed/test_seed42.npy",
        mapping_csv="out/result15/out/cat_3cfg_x2seed/label_mapping.csv",
    ),
    ModelSpec(
        name="Cat_seed228",
        oof_path="out/result15/out/cat_3cfg_x2seed/oof_seed228.npy",
        test_path="out/result15/out/cat_3cfg_x2seed/test_seed228.npy",
        mapping_csv="out/result15/out/cat_3cfg_x2seed/label_mapping.csv",
    ),
    ModelSpec(
        name="Cat_seed777",
        oof_path="out/result15/out/cat_3cfg_x2seed/oof_seed777.npy",
        test_path="out/result15/out/cat_3cfg_x2seed/test_seed777.npy",
        mapping_csv="out/result15/out/cat_3cfg_x2seed/label_mapping.csv",
    ),
    ModelSpec(
        name="Cat_seed1488",
        oof_path="out/result15/out/cat_3cfg_x2seed/oof_seed1488.npy",
        test_path="out/result15/out/cat_3cfg_x2seed/test_seed1488.npy",
        mapping_csv="out/result15/out/cat_3cfg_x2seed/label_mapping.csv",
    ),

    ModelSpec(
        name="Cat_seed42cpu",
        oof_path="out/result15/out/cat_3cfg_x2seed_cpu/oof_seed42.npy",
        test_path="out/result15/out/cat_3cfg_x2seed_cpu/test_seed42.npy",
        mapping_csv="out/result15/out/cat_3cfg_x2seed_cpu/label_mapping.csv",
    ),
    ModelSpec(
        name="Cat_seed228cpu",
        oof_path="out/result15/out/cat_3cfg_x2seed_cpu/oof_seed228.npy",
        test_path="out/result15/out/cat_3cfg_x2seed_cpu/test_seed228.npy",
        mapping_csv="out/result15/out/cat_3cfg_x2seed_cpu/label_mapping.csv",
    ),
    ModelSpec(
        name="Cat_seed777cpu",
        oof_path="out/result15/out/cat_3cfg_x2seed_cpu/oof_seed777.npy",
        test_path="out/result15/out/cat_3cfg_x2seed_cpu/test_seed777.npy",
        mapping_csv="out/result15/out/cat_3cfg_x2seed_cpu/label_mapping.csv",
    ),
    ModelSpec(
        name="Cat_seed1488cpu",
        oof_path="out/result15/out/cat_3cfg_x2seed_cpu/oof_seed1488.npy",
        test_path="out/result15/out/cat_3cfg_x2seed_cpu/test_seed1488.npy",
        mapping_csv="out/result15/out/cat_3cfg_x2seed_cpu/label_mapping.csv",
    ),

    ModelSpec(
        name="Old_LGBM",
        oof_path="out/result11(536)/oof_proba_lgbm_weighted.npy",
        test_path="out/result11(536)/test_proba_lgbm_weighted.npy",
        mapping_csv="out/result11(536)/label_mapping_lgbm.csv",
    ),
]

OUT_DIR = "result/ensemble_many"
os.makedirs(OUT_DIR, exist_ok=True)

# weight fitting knobs
MAX_WEIGHT: Optional[float] = 0.80   # None disables cap
N_GREEDY_PASSES = 3
ALPHA_GRID = np.linspace(0.0, 1.0, 401)  # ~0.0025 step


# -------------------------
# UTIL
# -------------------------
def load_y_as_required_indices() -> np.ndarray:
    y_raw = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")[TARGET_COL].astype(str).values
    extra = sorted(set(y_raw) - set(REQUIRED))
    if extra:
        raise ValueError(f"Train labels contain unknown classes not in REQUIRED: {extra}")
    label_to_idx = {c: i for i, c in enumerate(REQUIRED)}
    return np.array([label_to_idx[c] for c in y_raw], dtype=np.int64)


def normalize_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return p / s


def align_proba_to_required(proba: np.ndarray, mapping_csv: str) -> np.ndarray:
    classes = pd.read_csv(mapping_csv)["label"].astype(str).tolist()
    df = pd.DataFrame(proba, columns=classes)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{mapping_csv}: missing columns {missing}")

    return normalize_rows(df[REQUIRED].to_numpy(dtype=np.float64))


def apply_max_weight_cap(w: np.ndarray, max_w: Optional[float]) -> np.ndarray:
    if max_w is None:
        s = w.sum()
        return (w / s) if s > 0 else np.ones_like(w) / len(w)

    w = np.clip(w, 0.0, float(max_w))
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def ens_from_weights(probas: List[np.ndarray], w: np.ndarray) -> np.ndarray:
    p = np.zeros_like(probas[0], dtype=np.float64)
    for wi, pi in zip(w, probas):
        if wi != 0:
            p += wi * pi
    return normalize_rows(p)


# -------------------------
# WEIGHT FITTING
# -------------------------
def greedy_fit_weights(
    oofs: List[np.ndarray],
    y: np.ndarray,
    labels: np.ndarray,
    max_weight: Optional[float],
    alpha_grid: np.ndarray,
    n_passes: int,
) -> Tuple[np.ndarray, float]:
    m = len(oofs)
    w = np.ones(m, dtype=np.float64) / m
    w = apply_max_weight_cap(w, max_weight)

    p = ens_from_weights(oofs, w)
    best_ll = float(log_loss(y, p, labels=labels))

    for _ in range(n_passes):
        improved = False
        for i in range(m):
            p_cur = p
            pi = oofs[i]

            best_a = 0.0
            best_here = best_ll

            for a in alpha_grid:
                p_try = normalize_rows((1.0 - a) * p_cur + a * pi)
                ll = log_loss(y, p_try, labels=labels)
                if ll < best_here:
                    best_here = ll
                    best_a = float(a)

            if best_here + 1e-12 < best_ll:
                a = best_a
                w = (1.0 - a) * w
                w[i] = w[i] + a
                w = apply_max_weight_cap(w, max_weight)

                p = ens_from_weights(oofs, w)
                best_ll = float(log_loss(y, p, labels=labels))
                improved = True

        if not improved:
            break

    return w, best_ll


def main():
    y = load_y_as_required_indices()
    labels = np.arange(len(REQUIRED))

    # ----- LOAD OOF (+ TEST if present) -----
    oofs: List[np.ndarray] = []
    tests: List[Optional[np.ndarray]] = []
    names: List[str] = []

    for ms in MODELS:
        if not os.path.exists(ms.oof_path):
            raise FileNotFoundError(f"OOF not found: {ms.oof_path}")
        if not os.path.exists(ms.mapping_csv):
            raise FileNotFoundError(f"mapping_csv not found: {ms.mapping_csv}")

        oof_raw = np.load(ms.oof_path)
        oof = align_proba_to_required(oof_raw, ms.mapping_csv)

        if oof.shape[0] != y.shape[0]:
            raise ValueError(f"{ms.name}: OOF rows {oof.shape[0]} != y rows {y.shape[0]}")

        test = None
        if ms.test_path is not None:
            if not os.path.exists(ms.test_path):
                raise FileNotFoundError(f"TEST not found: {ms.test_path} (model {ms.name})")
            test_raw = np.load(ms.test_path)
            test = align_proba_to_required(test_raw, ms.mapping_csv)

        oofs.append(oof)
        tests.append(test)
        names.append(ms.name)

    # ----- SINGLE SCORES -----
    print("OOF logloss (single models):")
    for name, p in zip(names, oofs):
        ll = float(log_loss(y, p, labels=labels))
        print(f"  {name:28s} {ll:.6f}")

    # ----- UNIFORM -----
    w0 = np.ones(len(oofs), dtype=np.float64) / len(oofs)
    w0 = apply_max_weight_cap(w0, MAX_WEIGHT)
    ll0 = float(log_loss(y, ens_from_weights(oofs, w0), labels=labels))
    print(f"\nUniform ensemble logloss: {ll0:.6f}")

    # ----- FIT -----
    w_best, ll_best = greedy_fit_weights(
        oofs=oofs, y=y, labels=labels,
        max_weight=MAX_WEIGHT,
        alpha_grid=ALPHA_GRID,
        n_passes=N_GREEDY_PASSES
    )

    order = np.argsort(-w_best)
    print("\nBEST ensemble logloss:", f"{ll_best:.6f}")
    print("weights:")
    for idx in order:
        print(f"  {w_best[idx]:.5f}  {names[idx]}")

    # ----- SAVE WEIGHTS + OOF ENS -----
    out = {
        "required": REQUIRED,
        "max_weight": MAX_WEIGHT,
        "uniform_logloss": ll0,
        "best_logloss": ll_best,
        "models": [{"name": n, "weight": float(w)} for n, w in zip(names, w_best)],
    }
    with open(os.path.join(OUT_DIR, "weights.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    oof_ens = ens_from_weights(oofs, w_best)
    np.save(os.path.join(OUT_DIR, "oof_ens.npy"), oof_ens)

    # ----- SUBMISSION (if we have ALL test arrays) -----
    if any(t is None for t in tests):
        missing = [names[i] for i, t in enumerate(tests) if t is None]
        print("\nNo submission created: missing test proba for models:", missing)
        print("Set test_path for those models to generate submission.")
        return

    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")[[ID_COL]].reset_index(drop=True)
    test_ens = ens_from_weights([t for t in tests if t is not None], w_best)

    sub = pd.concat([test_ids, pd.DataFrame(test_ens, columns=REQUIRED)], axis=1)
    sub_path = os.path.join(OUT_DIR, "submission_weighted.csv")
    sub.to_csv(sub_path, index=False)

    np.save(os.path.join(OUT_DIR, "test_ens.npy"), test_ens)

    print("\nSaved:")
    print(" -", os.path.join(OUT_DIR, "weights.json"))
    print(" -", os.path.join(OUT_DIR, "oof_ens.npy"))
    print(" -", os.path.join(OUT_DIR, "test_ens.npy"))
    print(" -", sub_path)


if __name__ == "__main__":
    main()
