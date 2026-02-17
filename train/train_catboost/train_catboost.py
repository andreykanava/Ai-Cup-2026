# train/train_catboost_seed_ensemble.py

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold


DATA_DIR = "../../data/processed"
TARGET_COL = "bird_group"
ID_COL = "track_id"

N_SPLITS = 5

SEEDS = [42, 1337, 2024, 777, 999]   # ensemble seeds


def load_data():

    X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
    y_train_df = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")

    X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
    test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")

    y_raw = y_train_df[TARGET_COL].astype(str).values

    return X_train, y_raw, X_test, test_ids


def preprocess(X_train, X_test):

    drop_list = pd.read_csv("../features_to_drop.csv")["feature"].astype(str).tolist()
    drop_list = [c for c in drop_list if c in X_train.columns]

    print(f"Dropping {len(drop_list)} features")

    X_train = X_train.drop(columns=drop_list)
    X_test = X_test.drop(columns=drop_list)

    for df in (X_train, X_test):
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype("string").fillna("__MISSING__")

    cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    return X_train, X_test, cat_idx


def train_single_seed(seed, X_train, y, X_test, cat_idx, n_classes):

    print(f"\n========== SEED {seed} ==========")

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed
    )

    oof = np.zeros((len(X_train), n_classes), dtype=np.float32)
    test = np.zeros((len(X_test), n_classes), dtype=np.float32)

    fold_ll = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):

        X_tr = X_train.iloc[tr_idx]
        y_tr = y[tr_idx]

        X_va = X_train.iloc[va_idx]
        y_va = y[va_idx]

        model = CatBoostClassifier(

            loss_function="MultiClass",
            eval_metric="MultiClass",

            iterations=8000,
            learning_rate=0.01,
            depth=6,

            l2_leaf_reg=15.0,
            min_data_in_leaf=20,

            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
            random_strength=1.5,

            random_seed=seed,

            od_type="Iter",
            od_wait=200,

            task_type="GPU",
            thread_count=-1,

            verbose=200,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            cat_features=cat_idx if cat_idx else None,
            use_best_model=True,
        )

        va_proba = model.predict_proba(X_va)

        oof[va_idx] = va_proba

        ll = log_loss(y_va, va_proba, labels=np.arange(n_classes))

        fold_ll.append(ll)

        print(f"[seed {seed} fold {fold}] logloss={ll:.5f}")

        test += model.predict_proba(X_test) / N_SPLITS

    full_ll = log_loss(y, oof, labels=np.arange(n_classes))

    print(f"\nSEED {seed} CV logloss = {full_ll:.5f}")

    return oof, test, full_ll


def main():

    X_train, y_raw, X_test, test_ids = load_data()

    X_train, X_test, cat_idx = preprocess(X_train, X_test)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    n_classes = len(le.classes_)

    print("classes:", list(le.classes_))
    print("X_train:", X_train.shape)

    oof_all = np.zeros((len(X_train), n_classes), dtype=np.float32)
    test_all = np.zeros((len(X_test), n_classes), dtype=np.float32)

    seed_scores = []

    for seed in SEEDS:

        oof, test, score = train_single_seed(
            seed,
            X_train,
            y,
            X_test,
            cat_idx,
            n_classes
        )

        np.save(f"oof_seed_{seed}.npy", oof)
        np.save(f"test_seed_{seed}.npy", test)

        oof_all += oof
        test_all += test

        seed_scores.append(score)

    oof_all /= len(SEEDS)
    test_all /= len(SEEDS)

    final_ll = log_loss(y, oof_all, labels=np.arange(n_classes))


    print("\n========== FINAL ENSEMBLE ==========")
    print("seed scores:", seed_scores)
    print("final logloss:", final_ll)

    # save
    np.save("../out/result12(5266) - overfit/cat_files/oof_proba_cat.npy", oof_all)
    np.save("../out/result12(5266) - overfit/cat_files/test_proba_cat.npy", test_all)

    pd.DataFrame({"label": le.classes_}).to_csv(
        "../out/result12(5266) - overfit/cat_files/label_mapping_cat.csv",
        index=False
    )

    REQUIRED = [
        "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
        "Gulls", "Birds of Prey", "Waders", "Songbirds"
    ]

    proba_df = pd.DataFrame(test_all, columns=le.classes_)
    proba_df = proba_df[REQUIRED]

    submission = pd.concat(
        [test_ids[[ID_COL]].reset_index(drop=True),
         proba_df.reset_index(drop=True)],
        axis=1
    )

    submission.to_csv("submission_cat.csv", index=False)

    print("Saved submission_cat.csv")


if __name__ == "__main__":
    main()
