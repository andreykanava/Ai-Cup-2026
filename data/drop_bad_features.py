import pandas as pd

DATA_DIR = "processed"

X_TRAIN_PATH = f"{DATA_DIR}/X_train.parquet"
X_TEST_PATH  = f"{DATA_DIR}/X_test.parquet"

OUT_TRAIN_PATH = f"{DATA_DIR}/X_train_clean.parquet"
OUT_TEST_PATH  = f"{DATA_DIR}/X_test_clean.parquet"

DROP_LIST_MANUAL = [
    "acc_energy_vertical_p90",
    "acc_energy_xy_p90",
    "acc_energy_xy_std",
    "alt_kurt",
    "boost_glide_ratio",
    "boost_heading_acf10",
    "boost_speed_acf5",
    "boost_vertical_oscillation_freq",
    "climb_episode_count",
    "climb_episode_mean_len",
    "curv_kurt",
    "curv_skew",
    "descent_episode_max_len",
    "descent_segments_count",
    "duration_sec",
    "energy_xy_max",
    "energy_xy_std",
    "hd_large_turn_ratio",
    "hd_turn_rate_cv",
    "is_evening",
    "is_night",
    "jerk_energy_vertical_mean",
    "jerk_energy_xy_std",
    "jerk_rv_mean",
    "jerk_rv_p90",
    "max_speed",
    "rv_acf1",
    "rv_dom_freq",
    "rv_spec_entropy",
    "speed_burst_ratio_p90",
    "speed_skew",
    "speed_spec_entropy",
    "speed_std",
    "turn_kurt",
    "turn_skew",
    "vertical_burst_ratio_p90",
    "z_range",
]


def build_drop_list(columns):

    auto_drop = []

    for col in columns:

        if col.startswith("boost_"):
            auto_drop.append(col)

        elif col.startswith("jerk_"):
            auto_drop.append(col)

        elif col.endswith("_kurt"):
            auto_drop.append(col)

        elif col.endswith("_skew"):
            auto_drop.append(col)

        elif col.endswith("_spec_entropy"):
            auto_drop.append(col)

    full_drop = set(auto_drop) | set(DROP_LIST_MANUAL)

    full_drop = [c for c in full_drop if c in columns]

    return sorted(full_drop)


def main():

    print("Loading data...")

    X_train = pd.read_parquet(X_TRAIN_PATH)
    X_test  = pd.read_parquet(X_TEST_PATH)

    print("Original shape:")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    drop_cols = build_drop_list(X_train.columns)

    print(f"\nDropping {len(drop_cols)} features")

    for c in drop_cols[:20]:
        print(" -", c)

    X_train_clean = X_train.drop(columns=drop_cols)
    X_test_clean  = X_test.drop(columns=drop_cols)

    print("\nNew shape:")
    print("Train:", X_train_clean.shape)
    print("Test :", X_test_clean.shape)

    print("\nSaving...")

    X_train_clean.to_parquet(OUT_TRAIN_PATH)
    X_test_clean.to_parquet(OUT_TEST_PATH)

    pd.Series(drop_cols, name="feature").to_csv("dropped_features_full.csv", index=False)

    print("\nSaved:")
    print(" -", OUT_TRAIN_PATH)
    print(" -", OUT_TEST_PATH)
    print(" - dropped_features_full.csv")


if __name__ == "__main__":
    main()
