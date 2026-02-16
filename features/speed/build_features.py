import binascii
from shapely import wkb
import numpy as np
import pandas as pd

def extract_motion_features(hex_wkb, duration_sec):
    feats = {
        "mean_speed_real": np.nan,
        "speed_cv": np.nan,
        "acceleration_mean_xy": np.nan,
        "acceleration_std_xy": np.nan,
    }

    try:
        geom = wkb.loads(binascii.unhexlify(hex_wkb))
        coords = np.asarray(geom.coords, dtype=float)

        if coords.shape[0] < 2:
            return feats

        # --- SPEED ---
        dx = np.diff(coords[:, 0])
        dy = np.diff(coords[:, 1])

        speeds = np.sqrt(dx*dx + dy*dy)

        # mean_speed_real (path_length / duration)
        if duration_sec and duration_sec > 0:
            path_length = np.sum(speeds)
            feats["mean_speed_real"] = float(path_length / duration_sec)

        # speed_cv = std / mean
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)

        if mean_speed > 0:
            feats["speed_cv"] = float(std_speed / mean_speed)

        # --- ACCELERATION ---
        if speeds.shape[0] >= 2:
            acc = np.diff(speeds)

            feats["acceleration_mean_xy"] = float(np.mean(acc))
            feats["acceleration_std_xy"]  = float(np.std(acc))

        return feats

    except Exception:
        return feats



train = pd.read_parquet("../../data/interim/train/train_2.parquet")
motion_df = train.apply(
    lambda row: extract_motion_features(
        row["trajectory"],
        row["duration_sec"]
    ),
    axis=1
).apply(pd.Series)

print(motion_df.columns)

train = pd.concat([train, motion_df], axis=1)
train.to_parquet("../../data/interim/train/train_3.parquet", index=False)
