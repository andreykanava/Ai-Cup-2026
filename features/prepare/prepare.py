# prepare_dataset.py
# Full dataset prep for tabular ML (LightGBM/CatBoost/XGBoost-ready)

import re
import numpy as np
import pandas as pd


TRAIN_PATH = "../../data/interim/train/train_13.parquet"
TEST_PATH  = "../../data/interim/test/test_13.parquet"

TARGET_SPECIES = "bird_species"
TARGET_GROUP   = "bird_group"


# --- columns that are not usable / risky / not present in test ---
ALWAYS_DROP = [
    # IDs / leakage / human annotations (present only in train)
    "observation_id",
    "primary_observation_id",
    "observer_position",
    "observer_comment",
    "n_birds_observed",
    "bird_group",   # drop from features; keep separately if you want hierarchical
    "bird_species", # target, not a feature

    # Non-tabular heavy objects / geometry
    "trajectory",
    "trajectory_time",

    # Timestamps (usually not worth using raw; we already have hour/day_of_year)
    "timestamp_start_radar_utc",
    "timestamp_end_radar_utc",
]

# keep track_id for submission mapping but don't use as a feature
ID_COL = "track_id"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # basic sanity: ensure string column names, no weird spaces
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df.drop(columns=cols)


def _coerce_bools_to_int(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "bool":
            df[c] = df[c].astype(np.int8)
    return df


def _coerce_objects_to_numeric_or_category(df: pd.DataFrame, max_unique_for_cat: int = 1000) -> pd.DataFrame:
    """
    If any object columns remain (should be rare after dropping), convert:
    - if numeric-like -> numeric
    - else -> category (tree models can handle via encoding later)
    """
    df = df.copy()
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_cols:
        # try numeric conversion
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.98:  # mostly numeric
            df[c] = s
        else:
            # category for later encoding (CatBoost can use it directly)
            nunique = df[c].nunique(dropna=True)
            if nunique <= max_unique_for_cat:
                df[c] = df[c].astype("category")
            else:
                # too high-cardinality -> drop (usually IDs / text)
                df = df.drop(columns=[c])
    return df


def _handle_infs_and_nans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # replace infs
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _align_train_test(train_X: pd.DataFrame, test_X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make sure train/test have identical feature columns and order.
    - columns missing in test are dropped from train
    - columns missing in train are dropped from test
    """
    train_cols = set(train_X.columns)
    test_cols = set(test_X.columns)
    common = sorted(train_cols & test_cols)

    train_X = train_X[common].copy()
    test_X = test_X[common].copy()
    return train_X, test_X


def _report_schema(train: pd.DataFrame, test: pd.DataFrame) -> None:
    missing_in_test = sorted(set(train.columns) - set(test.columns))
    missing_in_train = sorted(set(test.columns) - set(train.columns))

    print("\n--- SCHEMA CHECK ---")
    print(f"train cols: {len(train.columns)} | test cols: {len(test.columns)}")
    if missing_in_test:
        print("Columns present in train but missing in test:")
        for c in missing_in_test:
            print("  -", c)
    if missing_in_train:
        print("Columns present in test but missing in train:")
        for c in missing_in_train:
            print("  -", c)
    if not missing_in_test and not missing_in_train:
        print("Train/Test schemas match.")


def prepare(
    train_path: str = TRAIN_PATH,
    test_path: str = TEST_PATH,
    target: str = TARGET_SPECIES,
    save: bool = True,
    out_dir: str = "../data/processed",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns:
      X_train, y_train, X_test, train_ids, test_ids
    """
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    train = _normalize_columns(train)
    test = _normalize_columns(test)

    _report_schema(train, test)

    # Keep ids (for debugging/submission mapping)
    train_ids = train[ID_COL].copy() if ID_COL in train.columns else pd.Series(np.arange(len(train)))
    test_ids = test[ID_COL].copy() if ID_COL in test.columns else pd.Series(np.arange(len(test)))

    # Extract target
    if target not in train.columns:
        raise KeyError(f"Target '{target}' not found in train. Available: {list(train.columns)}")
    y_train = train[target].copy()

    # Drop trash / leakage / non-tabular / target columns
    train_X = _drop_columns(train, ALWAYS_DROP + ([ID_COL] if ID_COL in train.columns else []))
    test_X  = _drop_columns(test,  ALWAYS_DROP + ([ID_COL] if ID_COL in test.columns else []))

    # If any leftover target columns exist (just in case)
    if target in train_X.columns:
        train_X = train_X.drop(columns=[target])

    # Coerce types
    train_X = _coerce_bools_to_int(train_X)
    test_X  = _coerce_bools_to_int(test_X)

    train_X = _coerce_objects_to_numeric_or_category(train_X)
    test_X  = _coerce_objects_to_numeric_or_category(test_X)

    # Clean infs
    train_X = _handle_infs_and_nans(train_X)
    test_X  = _handle_infs_and_nans(test_X)

    # Align feature columns
    train_X, test_X = _align_train_test(train_X, test_X)

    # Optional: basic missing value imputation (safe default for tree models)
    # We'll fill numeric NaNs with median from train; category NaNs with "__MISSING__"
    train_X = train_X.copy()
    test_X = test_X.copy()

    num_cols = [c for c in train_X.columns if pd.api.types.is_numeric_dtype(train_X[c])]
    cat_cols = [c for c in train_X.columns if str(train_X[c].dtype) == "category"]

    for c in num_cols:
        med = train_X[c].median(skipna=True)
        train_X[c] = train_X[c].fillna(med)
        test_X[c] = test_X[c].fillna(med)

    for c in cat_cols:
        train_X[c] = train_X[c].cat.add_categories(["__MISSING__"]).fillna("__MISSING__")
        test_X[c] = test_X[c].cat.add_categories(["__MISSING__"]).fillna("__MISSING__")

    # Final sanity
    assert list(train_X.columns) == list(test_X.columns), "Train/Test feature columns mismatch after alignment!"

    print("\n--- FINAL FEATURES ---")
    print("X_train shape:", train_X.shape)
    print("X_test  shape:", test_X.shape)
    print("Target unique:", y_train.nunique())
    print("Example columns:", list(train_X.columns[:15]), "..." if train_X.shape[1] > 15 else "")

    if save:
        import os
        os.makedirs(out_dir, exist_ok=True)

        # Save as parquet for speed and preserving dtypes
        train_X.to_parquet(f"{out_dir}/X_train.parquet", index=False)
        pd.DataFrame({target: y_train}).to_parquet(f"{out_dir}/y_train_{target}.parquet", index=False)
        test_X.to_parquet(f"{out_dir}/X_test.parquet", index=False)

        # Save ids (useful for submission)
        pd.DataFrame({ID_COL: train_ids}).to_parquet(f"{out_dir}/train_ids.parquet", index=False)
        pd.DataFrame({ID_COL: test_ids}).to_parquet(f"{out_dir}/test_ids.parquet", index=False)

        print(f"\nSaved to: {out_dir}/X_train.parquet, y_train.parquet, X_test.parquet (+ ids)")

    return train_X, y_train, test_X, train_ids, test_ids


if __name__ == "__main__":
    # 1) species
    prepare(target=TARGET_SPECIES, save=True, out_dir="../../data/processed")

    # 2) group (нужен для Kaggle сабмита)
    prepare(target=TARGET_GROUP, save=True, out_dir="../../data/processed")

