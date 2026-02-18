import numpy as np
import pandas as pd


EPS = 1e-12


def _safe_div(a, b, eps=EPS):
    a = a.astype(float)
    b = b.astype(float)
    return a / (b + eps)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds strong interaction/normalized/cyclic/log features in the SAME style:
    - returns df with new columns appended
    - silently skips features if required base columns are missing
    """

    X = df.copy()

    def has(*cols):
        return all(c in X.columns for c in cols)

    feats = {}

    # --- 1) Speed × Altitude ---
    if has("speed_mean", "alt_mean"):
        feats["ix_speed_alt_ratio"]   = _safe_div(X["speed_mean"], X["alt_mean"])
        feats["ix_speed_alt_product"] = X["speed_mean"].astype(float) * X["alt_mean"].astype(float)
        feats["ix_speed_alt_diff"]    = X["speed_mean"].astype(float) - X["alt_mean"].astype(float)

    if has("speed_cv", "alt_cv"):
        feats["ix_speed_alt_cv_ratio"] = _safe_div(X["speed_cv"], X["alt_cv"])

    # --- 2) Vertical vs horizontal ---
    if has("energy_vertical_mean", "energy_xy_mean"):
        feats["ix_vertical_xy_energy_ratio"] = _safe_div(X["energy_vertical_mean"], X["energy_xy_mean"])

    if has("energy_vertical_max", "energy_xy_mean"):
        feats["ix_verticalmax_xymean_ratio"] = _safe_div(X["energy_vertical_max"], X["energy_xy_mean"])

    if has("alt_range", "path_length"):
        feats["ix_vertical_efficiency"] = _safe_div(X["alt_range"], X["path_length"])

    # --- 3) Trajectory efficiency / geometry normalization ---
    if has("displacement", "path_length"):
        feats["ix_path_efficiency"] = _safe_div(X["displacement"], X["path_length"])

    if has("hull_area", "path_length"):
        feats["ix_area_per_length"] = _safe_div(X["hull_area"], X["path_length"])

    if has("displacement", "hull_perimeter"):
        feats["ix_disp_per_perimeter"] = _safe_div(X["displacement"], X["hull_perimeter"])

    if has("path_length", "hull_area"):
        feats["ix_movement_density"] = _safe_div(X["path_length"], X["hull_area"])

    # --- 4) Turn × speed / curvature × speed ---
    if has("turn_rate_mean", "speed_mean"):
        feats["ix_turn_speed_ratio"]   = _safe_div(X["turn_rate_mean"], X["speed_mean"])
        feats["ix_turn_speed_product"] = X["turn_rate_mean"].astype(float) * X["speed_mean"].astype(float)

    if has("curvature_mean", "speed_mean"):
        feats["ix_curv_speed_ratio"] = _safe_div(X["curvature_mean"], X["speed_mean"])

    if has("curvature_mean", "alt_mean"):
        feats["ix_curv_alt_ratio"] = _safe_div(X["curvature_mean"], X["alt_mean"])

    if has("hd_total_turn", "path_length"):
        feats["ix_turn_per_length"] = _safe_div(X["hd_total_turn"], X["path_length"])

    # --- 5) Shape features ---
    if has("hull_perimeter", "hull_area"):
        feats["ix_shape_compactness"]    = _safe_div(X["hull_perimeter"].astype(float) ** 2, X["hull_area"])
        feats["ix_perimeter_area_ratio"] = _safe_div(X["hull_perimeter"], X["hull_area"])

    # bbox fill ratio: prefer bounding_box_area if exists, else bbox_area
    bbox_col = None
    if "bounding_box_area" in X.columns:
        bbox_col = "bounding_box_area"
    elif "bbox_area" in X.columns:
        bbox_col = "bbox_area"

    if bbox_col is not None and has("hull_area"):
        feats["ix_bbox_fill_ratio"] = _safe_div(X["hull_area"], X[bbox_col])

    # --- 6) Distribution shape ratios (safe, low-risk) ---
    if has("speed_p90", "speed_mean"):
        feats["ix_speed_peak_ratio"] = _safe_div(X["speed_p90"], X["speed_mean"])

    if has("speed_std", "speed_mean"):
        feats["ix_speed_spread_ratio"] = _safe_div(X["speed_std"], X["speed_mean"])

    if has("alt_p90", "alt_mean"):
        feats["ix_alt_peak_ratio"] = _safe_div(X["alt_p90"], X["alt_mean"])

    if has("alt_std", "alt_mean"):
        feats["ix_alt_spread_ratio"] = _safe_div(X["alt_std"], X["alt_mean"])

    # --- 7) Cyclic time encodings ---
    if has("hour"):
        h = X["hour"].astype(float)
        feats["ix_hour_sin"] = np.sin(2 * np.pi * h / 24.0)
        feats["ix_hour_cos"] = np.cos(2 * np.pi * h / 24.0)

    if has("day_of_year"):
        d = X["day_of_year"].astype(float)
        feats["ix_day_sin"] = np.sin(2 * np.pi * d / 365.0)
        feats["ix_day_cos"] = np.cos(2 * np.pi * d / 365.0)

    # --- 8) Log transforms (only if positive-ish) ---
    if has("speed_mean"):
        feats["ix_speed_mean_log"] = np.log1p(np.clip(X["speed_mean"].astype(float), 0, None))

    if has("alt_mean"):
        feats["ix_alt_mean_log"] = np.log1p(np.clip(X["alt_mean"].astype(float), 0, None))

    if has("path_length"):
        feats["ix_path_length_log"] = np.log1p(np.clip(X["path_length"].astype(float), 0, None))

    if has("hull_area"):
        feats["ix_hull_area_log"] = np.log1p(np.clip(X["hull_area"].astype(float), 0, None))

    # --- append & cleanup ---
    if feats:
        feat_df = pd.DataFrame(feats, index=X.index)
        X = pd.concat([X, feat_df], axis=1)

    # remove accidental duplicates if any
    X = X.loc[:, ~X.columns.duplicated(keep="first")]

    return X


def main():
    IN_PATH  = "../../data/interim/train/train_11-optimal.parquet"  # поменяй на свой
    OUT_PATH = "../../data/interim/train/train_12.parquet"  # поменяй на свой

    df = pd.read_parquet(IN_PATH)

    df2 = add_interaction_features(df)

    print("Before:", df.shape)
    print("After :", df2.shape)
    added = [c for c in df2.columns if c not in df.columns]
    print(f"Added {len(added)} features")
    for c in added[:30]:
        print(" -", c)
    if len(added) > 30:
        print(f" ... +{len(added)-30} more")

    df2.to_parquet(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
