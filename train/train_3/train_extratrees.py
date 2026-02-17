# train/train_extratrees.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

DATA_DIR = "../../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

N_SPLITS = 5
RANDOM_STATE = 42

# Порядок классов как в твоих других скриптах (важно для ансамбля)
REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

OUT_DIR = "result_et"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    y_train_df = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")
    X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")[[ID_COL]]

    y_raw = y_train_df[TARGET_COL].astype(str).values
    return X_train, y_raw, X_test, test_ids


def ensure_required_labels(y_raw: np.ndarray) -> np.ndarray:
    missing = sorted(set(y_raw) - set(REQUIRED))
    if missing:
        raise ValueError(f"Train labels contain unknown classes not in REQUIRED: {missing}")
    label_to_idx = {c: i for i, c in enumerate(REQUIRED)}
    y = np.array([label_to_idx[c] for c in y_raw], dtype=np.int64)
    return y


def encode_non_numeric_together(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Факторизуем все нечисловые колонки совместно train+test, чтобы категории совпали."""
    Xtr = X_train.copy()
    Xte = X_test.copy()

    non_num = [c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])]
    if non_num:
        print(f"Non-numeric cols -> factorize: {len(non_num)} {non_num[:10]}{'...' if len(non_num) > 10 else ''}")

    for c in non_num:
        combined = pd.concat([Xtr[c], Xte[c]], axis=0).astype(str).fillna("__NA__")
        codes, uniques = pd.factorize(combined, sort=True)
        Xtr[c] = codes[: len(Xtr)].astype(np.int32)
        Xte[c] = codes[len(Xtr) :].astype(np.int32)

    # NaN в numeric -> заполним медианой по train
    for c in Xtr.columns:
        if pd.api.types.is_numeric_dtype(Xtr[c]):
            med = np.nanmedian(Xtr[c].to_numpy(dtype=np.float64))
            if np.isnan(med):
                med = 0.0
            Xtr[c] = Xtr[c].fillna(med)
            Xte[c] = Xte[c].fillna(med)

    return Xtr, Xte


def make_model(seed: int) -> ExtraTreesClassifier:
    # Это хорошие стартовые параметры для tabular + logloss
    return ExtraTreesClassifier(
        n_estimators=2000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,

        n_jobs=-1,
        random_state=seed,
        class_weight=None,
    )


def main():
    X_train, y_raw, X_test, test_ids = load_data()
    y = ensure_required_labels(y_raw)

    X_train, X_test = encode_non_numeric_together(X_train, X_test)

    Xtr = X_train.to_numpy(dtype=np.float32)
    Xte = X_test.to_numpy(dtype=np.float32)

    print(f"X_train={Xtr.shape} X_test={Xte.shape} classes={len(REQUIRED)}")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros((Xtr.shape[0], len(REQUIRED)), dtype=np.float64)
    test_sum = np.zeros((Xte.shape[0], len(REQUIRED)), dtype=np.float64)

    rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y), 1):
        model = make_model(RANDOM_STATE + fold)

        model.fit(Xtr[tr_idx], y[tr_idx])

        p_va = model.predict_proba(Xtr[va_idx])
        p_te = model.predict_proba(Xte)

        # safety нормализация
        p_va = np.clip(p_va, 1e-7, 1 - 1e-7)
        p_va /= p_va.sum(axis=1, keepdims=True)
        p_te = np.clip(p_te, 1e-7, 1 - 1e-7)
        p_te /= p_te.sum(axis=1, keepdims=True)

        oof[va_idx] = p_va
        test_sum += p_te / N_SPLITS

        ll = log_loss(y[va_idx], p_va, labels=np.arange(len(REQUIRED)))
        acc = accuracy_score(y[va_idx], np.argmax(p_va, axis=1))

        rows.append({"fold": fold, "logloss": ll, "acc": acc})
        print(f"[fold {fold}] acc={acc:.4f} logloss={ll:.4f}")

    cv = pd.DataFrame(rows)
    oof_ll = log_loss(y, oof, labels=np.arange(len(REQUIRED)))
    oof_acc = accuracy_score(y, np.argmax(oof, axis=1))

    print("\n=== CV SUMMARY (ExtraTrees) ===")
    print(f"acc:     {oof_acc:.4f}")
    print(f"logloss: {oof_ll:.4f}")
    print("\nproba sum check:", float(oof.sum(axis=1).min()), float(oof.sum(axis=1).max()))

    # save artifacts
    np.save(f"{OUT_DIR}/oof_proba_et.npy", oof)
    np.save(f"{OUT_DIR}/test_proba_et.npy", test_sum)

    pd.DataFrame({"label": REQUIRED}).to_csv(f"{OUT_DIR}/label_mapping_et.csv", index=False)
    cv.to_csv(f"{OUT_DIR}/cv_metrics_et.csv", index=False)

    # submission proba
    sub_proba = pd.concat(
        [test_ids.reset_index(drop=True), pd.DataFrame(test_sum, columns=REQUIRED)],
        axis=1
    )
    sub_proba.to_csv(f"{OUT_DIR}/submission_et_proba.csv", index=False)

    # submission label
    pred_idx = test_sum.argmax(axis=1)
    sub_label = test_ids.copy()
    sub_label[TARGET_COL] = [REQUIRED[i] for i in pred_idx]
    sub_label.to_csv(f"{OUT_DIR}/submission_et_label.csv", index=False)

    print("\nSaved files:")
    print(f" - {OUT_DIR}/oof_proba_et.npy")
    print(f" - {OUT_DIR}/test_proba_et.npy")
    print(f" - {OUT_DIR}/label_mapping_et.csv")
    print(f" - {OUT_DIR}/cv_metrics_et.csv")
    print(f" - {OUT_DIR}/submission_et_proba.csv")
    print(f" - {OUT_DIR}/submission_et_label.csv")


if __name__ == "__main__":
    main()
