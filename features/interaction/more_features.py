import pandas as pd


import numpy as np
import pandas as pd

EPS = 1e-12

def _safe_div(a, b, eps=EPS):
    a = a.astype(float)
    b = b.astype(float)
    return a / (b + eps)

def add_more_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds another batch of high-signal features (stability/efficiency/heading-interactions/size-normalization).
    Same style as before:
      - returns df with new columns appended
      - skips anything if required base columns are missing
      - dedups column names at the end
    """
    X = df.copy()

    def has(*cols):
        return all(c in X.columns for c in cols)

    feats = {}

    # -------------------------
    # 1) Stability / smoothness
    # -------------------------
    if has("speed_mean", "speed_std"):
        feats["ix_speed_stability"] = _safe_div(X["speed_mean"], X["speed_std"])

    if has("alt_mean", "alt_std"):
        feats["ix_alt_stability"] = _safe_div(X["alt_mean"], X["alt_std"])

    # heading stability directly
    if has("hd_heading_r"):
        feats["ix_heading_stability"] = X["hd_heading_r"].astype(float)

    if has("hd_turn_abs_mean", "hd_heading_r"):
        feats["ix_motion_smoothness"] = _safe_div(X["hd_heading_r"], X["hd_turn_abs_mean"])

    if has("hd_turn_abs_std"):
        feats["ix_turn_stability_invstd"] = _safe_div(pd.Series(1.0, index=X.index), X["hd_turn_abs_std"])

    if has("hd_turn_abs_std", "hd_turn_abs_mean"):
        feats["ix_turn_chaos"] = _safe_div(X["hd_turn_abs_std"], X["hd_turn_abs_mean"])

    # -------------------------
    # 2) Energy / distance efficiency
    # -------------------------
    if has("energy_total_mean", "path_length"):
        feats["ix_energy_per_distance"] = _safe_div(X["energy_total_mean"], X["path_length"])

    if has("energy_xy_mean", "path_length"):
        feats["ix_horizontal_energy_per_distance"] = _safe_div(X["energy_xy_mean"], X["path_length"])

    if has("energy_vertical_mean", "alt_range"):
        feats["ix_vertical_energy_per_alt"] = _safe_div(X["energy_vertical_mean"], X["alt_range"])

    # -------------------------
    # 3) Geometry / compactness vs displacement
    # -------------------------
    if has("displacement", "hull_area"):
        feats["ix_displacement_area_ratio"] = _safe_div(X["displacement"], X["hull_area"])

    if has("displacement", "hull_perimeter"):
        feats["ix_displacement_perimeter_ratio"] = _safe_div(X["displacement"], X["hull_perimeter"])

    bbox_col = None
    if "bounding_box_area" in X.columns:
        bbox_col = "bounding_box_area"
    elif "bbox_area" in X.columns:
        bbox_col = "bbox_area"

    if bbox_col is not None and has("path_length"):
        feats["ix_path_vs_bbox_area"] = _safe_div(X["path_length"], X[bbox_col])

    # -------------------------
    # 4) Heading-based interactions (gold mine)
    # -------------------------
    if has("hd_heading_r", "speed_mean"):
        feats["ix_heading_speed_alignment"] = X["hd_heading_r"].astype(float) * X["speed_mean"].astype(float)

    if has("hd_turn_efficiency", "speed_mean"):
        feats["ix_heading_turn_efficiency_x_speed"] = (
            X["hd_turn_efficiency"].astype(float) * X["speed_mean"].astype(float)
        )

    if has("hd_heading_r", "curvature_mean"):
        feats["ix_heading_over_curvature"] = _safe_div(X["hd_heading_r"], X["curvature_mean"])

    if has("hd_heading_r", "hd_turn_abs_mean"):
        feats["ix_directional_persistence"] = _safe_div(X["hd_heading_r"], X["hd_turn_abs_mean"])

    # -------------------------
    # 5) Peak vs median ratios (distribution shape)
    # -------------------------
    if has("speed_p90", "speed_p50"):
        feats["ix_speed_peak_vs_median"] = _safe_div(X["speed_p90"], X["speed_p50"])

    if has("alt_p90", "alt_p50"):
        feats["ix_alt_peak_vs_median"] = _safe_div(X["alt_p90"], X["alt_p50"])

    if has("hd_turn_abs_p90", "hd_turn_abs_mean"):
        feats["ix_turn_peak_vs_mean"] = _safe_div(X["hd_turn_abs_p90"], X["hd_turn_abs_mean"])

    # -------------------------
    # 6) Normalization by bird size (if numeric)
    # -------------------------
    # radar_bird_size might be categorical; only do these if it looks numeric
    if "radar_bird_size" in X.columns:
        s = X["radar_bird_size"]
        if pd.api.types.is_numeric_dtype(s):
            if has("speed_mean"):
                feats["ix_speed_per_size"] = _safe_div(X["speed_mean"], s)
            if has("alt_mean"):
                feats["ix_alt_per_size"] = _safe_div(X["alt_mean"], s)
            if has("energy_total_mean"):
                feats["ix_energy_per_size"] = _safe_div(X["energy_total_mean"], s)

    # -------------------------
    # 7) Motion anisotropy / dominance
    # -------------------------
    if has("energy_vertical_std", "energy_xy_std"):
        feats["ix_vertical_horizontal_std_ratio"] = _safe_div(X["energy_vertical_std"], X["energy_xy_std"])

    if has("alt_range", "displacement"):
        feats["ix_vertical_dominance"] = _safe_div(X["alt_range"], X["displacement"])

    # -------------------------
    # 8) Nonlinear boosts (safe)
    # -------------------------
    if has("speed_mean"):
        feats["ix_speed_sq"] = X["speed_mean"].astype(float) ** 2

    if has("alt_mean"):
        feats["ix_alt_sq"] = X["alt_mean"].astype(float) ** 2

    if has("hd_turn_abs_mean"):
        feats["ix_turn_abs_mean_sq"] = X["hd_turn_abs_mean"].astype(float) ** 2

    # -------------------------
    # 9) Interactions with cyclic time features (if you added ix_hour_sin/day_sin before)
    # -------------------------
    if has("speed_mean", "ix_hour_sin"):
        feats["ix_speed_x_hour_sin"] = X["speed_mean"].astype(float) * X["ix_hour_sin"].astype(float)

    if has("alt_mean", "ix_day_sin"):
        feats["ix_alt_x_day_sin"] = X["alt_mean"].astype(float) * X["ix_day_sin"].astype(float)

    # append
    if feats:
        feat_df = pd.DataFrame(feats, index=X.index)
        X = pd.concat([X, feat_df], axis=1)

    # remove accidental duplicates
    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    return X


def main():

    IN_PATH  = "../../data/interim/test/test_12.parquet"
    OUT_PATH = "../../data/interim/test/test_13.parquet"

    print("Loading:", IN_PATH)

    df = pd.read_parquet(IN_PATH)

    print("Original shape:", df.shape)

    cols_before = set(df.columns)

    # --- add new features ---
    df2 = add_more_interaction_features(df)

    cols_after = set(df2.columns)
    added = sorted(cols_after - cols_before)

    print("\nAdded features:", len(added))

    for c in added[:30]:
        print(" -", c)

    if len(added) > 30:
        print(f" ... +{len(added) - 30} more")

    print("\nNew shape:", df2.shape)

    print("\nSaving:", OUT_PATH)

    df2.to_parquet(OUT_PATH, index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()



