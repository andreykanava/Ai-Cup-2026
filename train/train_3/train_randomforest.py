import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import HistGradientBoostingClassifier


DATA_DIR = "../../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

N_SPLITS = 5
BASE_RANDOM_STATE = 42

MAPPING_PATH_PREFERRED = "../out/result9(537)/label_mapping_cat.csv"
MAPPING_PATH_OUT = "../out/result9(537)/label_mapping_hgb_ms_ts.csv"

REQUIRED = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds"
]

SEEDS = [42, 1337, 2026, 7777]  # try 3-5, 4 is a good start


def load_data():
    X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    y_train_df = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")
    X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")
    y_raw = y_train_df[TARGET_COL].astype(str).values
    return X_train, y_raw, X_test, test_ids


def load_or_build_label_mapping(y_raw: np.ndarray) -> np.ndarray:
    os.makedirs(os.path.dirname(MAPPING_PATH_OUT), exist_ok=True)

    if os.path.exists(MAPPING_PATH_PREFERRED):
        df = pd.read_csv(MAPPING_PATH_PREFERRED)
        classes = df["label"].astype(str).values
        pd.DataFrame({"label": classes}).to_csv(MAPPING_PATH_OUT, index=False)
        return classes

    unique = sorted(set(map(str, y_raw)))
    if set(REQUIRED).issubset(set(unique)) and len(unique) == len(REQUIRED):
        classes = np.array(REQUIRED, dtype=str)
    else:
        classes = np.array(unique, dtype=str)

    pd.DataFrame({"label": classes}).to_csv(MAPPING_PATH_OUT, index=False)
    return classes


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def probs_to_logits(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return np.log(p)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    logits = probs_to_logits(proba)
    return softmax(logits / T)


def find_best_temperature(oof_proba: np.ndarray, y: np.ndarray, n_classes: int):
    best_T, best_ll = None, 1e9
    # tighter search around likely range
    grid = np.concatenate([
        np.linspace(0.8, 1.6, 41),
        np.linspace(1.6, 2.6, 21)
    ])
    grid = np.unique(np.round(grid, 4))
    for T in grid:
        p = apply_temperature(oof_proba, float(T))
        ll = log_loss(y, p, labels=np.arange(n_classes))
        if ll < best_ll:
            best_ll = ll
            best_T = float(T)
    return best_T, best_ll


def make_model(pre, seed: int):
    # keep this close to your good config
    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_depth=8,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", hgb)])


def main():
    X_train, y_raw, X_test, test_ids = load_data()

    drop_list_path = "../features_to_drop.csv"
    if os.path.exists(drop_list_path):
        drop_list = pd.read_csv(drop_list_path)["feature"].astype(str).tolist()
        drop_list = [c for c in drop_list if c in X_train.columns]
        print(f"Dropping {len(drop_list)} features")
        X_train = X_train.drop(columns=drop_list)
        X_test = X_test.drop(columns=drop_list)
    else:
        print("features_to_drop.csv not found -> not dropping anything")

    classes = load_or_build_label_mapping(y_raw)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[str(lbl)] for lbl in y_raw], dtype=np.int32)
    n_classes = len(classes)

    print(f"X_train={X_train.shape} X_test={X_test.shape} classes={n_classes}")

    pre = build_preprocessor(X_train)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=BASE_RANDOM_STATE)

    oof_avg = np.zeros((len(X_train), n_classes), dtype=np.float32)
    test_avg = np.zeros((len(X_test), n_classes), dtype=np.float32)

    per_seed = []

    for si, seed in enumerate(SEEDS, 1):
        print(f"\n=== SEED {seed} ({si}/{len(SEEDS)}) ===")
        oof = np.zeros((len(X_train), n_classes), dtype=np.float32)
        test = np.zeros((len(X_test), n_classes), dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
            model = make_model(pre, seed + fold)
            X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y[va_idx]

            model.fit(X_tr, y_tr)
            p_va = model.predict_proba(X_va)
            oof[va_idx] = p_va

            ll = log_loss(y_va, p_va, labels=np.arange(n_classes))
            acc = accuracy_score(y_va, p_va.argmax(axis=1))
            print(f"[seed {seed} fold {fold}] acc={acc:.4f} logloss={ll:.5f}")

            test += model.predict_proba(X_test) / N_SPLITS

        seed_ll = log_loss(y, oof, labels=np.arange(n_classes))
        print(f"--- SEED {seed} OOF logloss: {seed_ll:.5f}")
        per_seed.append({"seed": seed, "oof_logloss": seed_ll})

        np.save(f"hgb_seeds/oof_proba_hgb_seed{seed}.npy", oof)
        np.save(f"hgb_seeds/test_proba_hgb_seed{seed}.npy", test)

        oof_avg += oof / len(SEEDS)
        test_avg += test / len(SEEDS)

    raw_ll = log_loss(y, oof_avg, labels=np.arange(n_classes))
    raw_acc = accuracy_score(y, oof_avg.argmax(axis=1))
    print("\n=== AVG (raw) ===")
    print(f"acc: {raw_acc:.4f} logloss: {raw_ll:.5f}")

    # temperature on averaged oof
    best_T, best_ll = find_best_temperature(oof_avg, y, n_classes)
    print(f"\nBest temperature on AVG: T={best_T:.4f} -> logloss={best_ll:.5f}")

    oof_ts = apply_temperature(oof_avg, best_T).astype(np.float32)
    test_ts = apply_temperature(test_avg, best_T).astype(np.float32)

    ts_ll = log_loss(y, oof_ts, labels=np.arange(n_classes))
    print(f"Temp-scaled AVG OOF logloss: {ts_ll:.5f}")

    np.save("result9(537)/oof_proba_hgb_ms.npy", oof_avg)
    np.save("result9(537)/test_proba_hgb_ms.npy", test_avg)
    np.save("result9(537)/oof_proba_hgb_ms_ts.npy", oof_ts)
    np.save("result9(537)/test_proba_hgb_ms_ts.npy", test_ts)

    pd.DataFrame(per_seed).to_csv("cv_metrics_hgb_ms.csv", index=False)

    proba_df = pd.DataFrame(test_ts, columns=classes)
    proba_df = proba_df[REQUIRED]

    s = proba_df.sum(axis=1).values
    print("\nproba sum check:", float(np.min(s)), float(np.max(s)), float(np.mean(s)))

    sub = pd.concat(
        [test_ids[[ID_COL]].reset_index(drop=True), proba_df.reset_index(drop=True)],
        axis=1
    )
    sub.to_csv("submission_hgb_ms_ts.csv", index=False)

    print("\nSaved files:")
    print(" - oof_proba_hgb_ms.npy / test_proba_hgb_ms.npy")
    print(" - oof_proba_hgb_ms_ts.npy / test_proba_hgb_ms_ts.npy")
    print(f" - {MAPPING_PATH_OUT}")
    print(" - cv_metrics_hgb_ms.csv")
    print(" - submission_hgb_ms_ts.csv")
    print(sub.head())


if __name__ == "__main__":
    main()
