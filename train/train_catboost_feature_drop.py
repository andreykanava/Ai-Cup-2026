# train/train_catboost_feature_drop.py

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

from catboost import CatBoostClassifier, Pool


DATA_DIR = "../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

N_SPLITS = 5
RANDOM_STATE = 42


def load_data():
    X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    y_train_df = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")
    X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")
    y_raw = y_train_df[TARGET_COL].astype(str).values
    return X_train, y_raw, X_test, test_ids


def main():
    X_train, y_raw, X_test, test_ids = load_data()

    # Make non-numeric cols categorical for CatBoost
    for df in (X_train, X_test):
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype("string").fillna("__MISSING__")

    cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(int)
    n_classes = len(le.classes_)

    counts = np.bincount(y)
    rare_classes = np.where(counts < 2)[0]
    rare_mask = np.isin(y, rare_classes)
    rare_idx = np.where(rare_mask)[0]
    ok_idx = np.where(~rare_mask)[0]

    ok_counts = np.bincount(y[ok_idx])
    ok_min = ok_counts[ok_counts > 0].min()
    n_splits = min(N_SPLITS, ok_min)
    if n_splits < 2:
        n_splits = 2

    print(f"X_train={X_train.shape} X_test={X_test.shape} classes={n_classes}")
    print(f"rare_classes={len(rare_classes)} rare_samples={len(rare_idx)} n_splits={n_splits}")
    print(f"categorical cols: {len(cat_cols)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_proba = np.zeros((len(X_train), n_classes), dtype=np.float32)
    test_proba = np.zeros((len(X_test), n_classes), dtype=np.float32)

    fold_acc, fold_ll = [], []

    # сюда будем собирать importance по фолдам
    fold_importances = []

    X_ok = X_train.iloc[ok_idx]
    y_ok = y[ok_idx]

    for fold, (tr_local, va_local) in enumerate(skf.split(X_ok, y_ok), 1):
        tr_idx = ok_idx[tr_local]
        va_idx = ok_idx[va_local]
        tr_idx = np.concatenate([tr_idx, rare_idx])  # rare always in train

        X_tr = X_train.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y[va_idx]

        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            classes_count=n_classes,

            iterations=5000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=6.0,

            random_seed=RANDOM_STATE,
            od_type="Iter",
            od_wait=100,
            verbose=100,

            task_type="CPU",
            thread_count=-1,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            cat_features=cat_idx if cat_idx else None,
            use_best_model=True,
        )

        va_proba = model.predict_proba(X_va)
        oof_proba[va_idx] = va_proba

        acc = accuracy_score(y_va, va_proba.argmax(axis=1))
        ll = log_loss(y_va, va_proba, labels=np.arange(n_classes))

        fold_acc.append(acc)
        fold_ll.append(ll)
        print(f"[fold {fold}] acc={acc:.4f} logloss={ll:.4f}")

        test_proba += model.predict_proba(X_test) / n_splits

        # -------- Feature importance: LossFunctionChange on validation --------
        va_pool = Pool(X_va, y_va, cat_features=cat_idx if cat_idx else None)

        imp = model.get_feature_importance(
            data=va_pool,
            type="LossFunctionChange"
        )

        imp_df = pd.DataFrame({
            "feature": X_train.columns,
            f"fold_{fold}": imp.astype(float)
        }).set_index("feature")

        fold_importances.append(imp_df)

    overall_acc = accuracy_score(y[ok_idx], oof_proba[ok_idx].argmax(axis=1))
    overall_ll = log_loss(y[ok_idx], oof_proba[ok_idx], labels=np.arange(n_classes))

    print("\n=== CV SUMMARY (non-rare only) ===")
    print(f"acc:     {overall_acc:.4f} (folds: {', '.join(f'{x:.4f}' for x in fold_acc)})")
    print(f"logloss: {overall_ll:.4f} (folds: {', '.join(f'{x:.4f}' for x in fold_ll)})")

    # -------- Aggregate importances across folds --------
    lfc = pd.concat(fold_importances, axis=1)

    # mean / std / positive ratio across folds
    fold_cols = [c for c in lfc.columns if c.startswith("fold_")]
    lfc["mean_imp"] = lfc[fold_cols].mean(axis=1)
    lfc["std_imp"] = lfc[fold_cols].std(axis=1)
    lfc["pos_ratio"] = (lfc[fold_cols] > 0).mean(axis=1)

    # save full summary
    lfc_sorted = lfc.sort_values("mean_imp", ascending=False)
    lfc_sorted.to_csv("lfc_summary.csv")

    print("\n=== TOP 25 helpful features (mean_imp desc) ===")
    print(lfc_sorted.head(25)[["mean_imp", "std_imp", "pos_ratio"]])

    print("\n=== TOP 25 harmful features (mean_imp asc) ===")
    print(lfc_sorted.tail(25)[["mean_imp", "std_imp", "pos_ratio"]].sort_values("mean_imp"))

    # -------- Decide what to drop --------
    # базовое правило: в среднем не помогает или вредно
    to_drop = lfc[(lfc["mean_imp"] <= 0)].index.tolist()

    # чуть более жёсткий режим (если хочешь):
    # to_drop = lfc[(lfc["mean_imp"] <= 0) | (lfc["pos_ratio"] < 0.4)].index.tolist()

    pd.Series(to_drop, name="feature").to_csv("features_to_drop.csv", index=False)

    print("\nSaved: lfc_summary.csv")
    print(f"Saved: features_to_drop.csv  (count={len(to_drop)})")

    # ---- submission (оставил как у тебя) ----
    REQUIRED = [
        "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
        "Gulls", "Birds of Prey", "Waders", "Songbirds"
    ]

    got = list(le.classes_)
    missing = [c for c in REQUIRED if c not in got]
    extra = [c for c in got if c not in REQUIRED]

    print("le.classes_:", got)
    if missing:
        raise ValueError(f"Missing classes in training labels: {missing}")
    if extra:
        print("Warning: extra classes in training labels (not required):", extra)

    proba_df = pd.DataFrame(test_proba, columns=le.classes_)
    proba_df = proba_df[REQUIRED]

    sub = pd.concat(
        [test_ids[[ID_COL]].reset_index(drop=True), proba_df.reset_index(drop=True)],
        axis=1
    )

    np.save("test_proba_cat.npy", test_proba)
    np.save("oof_proba_cat.npy", oof_proba)
    pd.DataFrame({"label": le.classes_}).to_csv("label_mapping_cat.csv", index=False)
    sub.to_csv("submission_cat.csv", index=False)

    print("Saved: test_proba_cat.npy, oof_proba_cat.npy, label_mapping_cat.csv")
    print("Saved: submission_cat.csv", sub.shape)
    print(sub.head())


if __name__ == "__main__":
    main()
