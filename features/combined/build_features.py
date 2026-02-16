import numpy as np
import pandas as pd

def extract_combined_features(
    curvature_mean,
    curvature_std,
    speed_mean,
    speed_std,
    tortuosity,
    abs_rv_mean,
    mean_speed_real,
    vertical_activity_ratio,
    eps=1e-6
):
    """
    Combined features computed from already-extracted columns.
    Pass scalar values from a row.
    """
    feats = {
        "curvature_speed_product": np.nan,
        "rv_curvature_product": np.nan,
        "tortuosity_speed_ratio": np.nan,
        "vertical_horizontal_motion_ratio": np.nan,
    }

    try:
        # curvature_speed_product: more turning * more movement
        if curvature_mean is not None and speed_mean is not None:
            feats["curvature_speed_product"] = float(curvature_mean) * float(speed_mean)

        # rv_curvature_product: vertical activity times curvature
        if abs_rv_mean is not None and curvature_mean is not None:
            feats["rv_curvature_product"] = float(abs_rv_mean) * float(curvature_mean)

        # tortuosity_speed_ratio: how "wiggly" per speed
        if tortuosity is not None and speed_mean is not None:
            feats["tortuosity_speed_ratio"] = float(tortuosity) / (float(speed_mean) + eps)

        # vertical_horizontal_motion_ratio:
        # prefer physically meaningful ratio if mean_speed_real exists;
        # fallback to vertical_activity_ratio if you already computed it.
        if abs_rv_mean is not None:
            if mean_speed_real is not None and np.isfinite(mean_speed_real) and mean_speed_real > 0:
                feats["vertical_horizontal_motion_ratio"] = float(abs_rv_mean) / (float(mean_speed_real) + eps)
            elif vertical_activity_ratio is not None and np.isfinite(vertical_activity_ratio):
                feats["vertical_horizontal_motion_ratio"] = float(vertical_activity_ratio)

        return feats

    except Exception:
        return feats

train = pd.read_parquet("../../data/interim/train/train_5.parquet")
comb_df = train.apply(
    lambda row: extract_combined_features(
        row.get("curvature_mean", np.nan),
        row.get("curvature_std", np.nan),
        row.get("speed_mean", np.nan),
        row.get("speed_std", np.nan),
        row.get("tortuosity", np.nan),
        row.get("abs_rv_mean", np.nan),
        row.get("mean_speed_real", np.nan),
        row.get("vertical_activity_ratio", np.nan),
    ),
    axis=1
).apply(pd.Series)

train = pd.concat([train, comb_df], axis=1)
train.to_parquet("../../data/interim/train/train_6.parquet", index=False)
