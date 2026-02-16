import binascii
from shapely import wkb
import numpy as np
import pandas as pd

def extract_geometry_radar_derived_features(
    hex_wkb,
    duration_sec,
    airspeed,
    min_z,
    max_z,
    eps=1e-12
):
    feats = {
        # GEOMETRY ADVANCED
        "net_displacement_speed_ratio": np.nan,
        "curvature_std": np.nan,
        "turn_rate_cv": np.nan,

        # RADAR DERIVED
        "z_range": np.nan,
        "airspeed_path_speed_ratio": np.nan,
    }

    try:
        # --- RADAR derived ---
        if min_z is not None and max_z is not None:
            feats["z_range"] = float(max_z - min_z)

        # --- Geometry derived from trajectory ---
        geom = wkb.loads(binascii.unhexlify(hex_wkb))
        coords = np.asarray(geom.coords, dtype=float)

        if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
            return feats

        x = coords[:, 0]
        y = coords[:, 1]
        n = coords.shape[0]

        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.sqrt(dx*dx + dy*dy)

        path_length = float(np.sum(seg_len))
        displacement = float(np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))

        # net_displacement_speed_ratio:
        # (displacement / duration) / (path_length / duration) == displacement / path_length
        # but computed robustly with duration when possible
        if duration_sec and duration_sec > 0 and path_length > 0:
            net_speed = displacement / duration_sec
            path_speed = path_length / duration_sec
            feats["net_displacement_speed_ratio"] = float(net_speed / (path_speed + eps))
        elif path_length > 0:
            feats["net_displacement_speed_ratio"] = float(displacement / (path_length + eps))

        # airspeed_path_speed_ratio = airspeed / (path_length/duration)
        if duration_sec and duration_sec > 0 and path_length > 0 and airspeed is not None:
            path_speed = path_length / duration_sec
            if path_speed > 0:
                feats["airspeed_path_speed_ratio"] = float(float(airspeed) / (path_speed + eps))

        # Turn rate and curvature distributions
        if seg_len.size >= 2:
            angles = np.arctan2(dy, dx)
            angles = np.unwrap(angles)
            dtheta = np.abs(np.diff(angles))  # size = len(seg_len)-1

            # curvature ~ |Δθ| / segment_length (aligned with seg_len[1:])
            curvature = dtheta / (seg_len[1:] + eps)

            feats["curvature_std"] = float(np.std(curvature)) if curvature.size else np.nan

            turn_mean = float(np.mean(dtheta)) if dtheta.size else np.nan
            turn_std = float(np.std(dtheta)) if dtheta.size else np.nan
            if turn_mean and turn_mean > 0:
                feats["turn_rate_cv"] = float(turn_std / (turn_mean + eps))

        return feats

    except Exception:
        return feats



train = pd.read_parquet("../../data/interim/train/train_3.parquet")
geo_radar_df = train.apply(
    lambda row: extract_geometry_radar_derived_features(
        row["trajectory"],
        row.get("duration_sec", np.nan),
        row.get("airspeed", np.nan),
        row.get("min_z", np.nan),
        row.get("max_z", np.nan),
    ),
    axis=1
).apply(pd.Series)
print(geo_radar_df.columns)

train = pd.concat([train, geo_radar_df], axis=1)

train.to_parquet("../../data/interim/train/train_4.parquet", index=False)