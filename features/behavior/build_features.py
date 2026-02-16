import binascii
from shapely import wkb
import numpy as np
import pandas as pd

def extract_behavior_structure_features(hex_wkb, duration_sec):
    feats = {
        # VERTICAL BEHAVIOR
        "alt_cv": np.nan,
        "rv_cv": np.nan,
        "vertical_activity_ratio": np.nan,
        "climb_descent_ratio": np.nan,

        # TRAJECTORY STRUCTURE
        "segment_density": np.nan,
        "heading_entropy": np.nan,
        "radius_of_gyration": np.nan,
        "bounding_box_area": np.nan,
        "movement_efficiency": np.nan,
    }

    try:
        geom = wkb.loads(binascii.unhexlify(hex_wkb))
        coords = np.asarray(geom.coords, dtype=float)

        if coords.ndim != 2 or coords.shape[0] == 0:
            return feats

        n = coords.shape[0]
        dim = coords.shape[1]

        # -------------------------
        # XY basics
        # -------------------------
        if dim >= 2 and n >= 2:
            x = coords[:, 0]
            y = coords[:, 1]

            dx = np.diff(x)
            dy = np.diff(y)
            seg_len = np.sqrt(dx*dx + dy*dy)

            path_length = float(np.sum(seg_len))
            displacement = float(np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))

            # movement_efficiency == straightness_index == displacement / path_length
            if path_length > 0:
                feats["movement_efficiency"] = float(displacement / path_length)

            # bounding box area
            feats["bounding_box_area"] = float((np.max(x) - np.min(x)) * (np.max(y) - np.min(y)))

            # radius of gyration (spatial spread)
            cx = float(np.mean(x))
            cy = float(np.mean(y))
            rg2 = np.mean((x - cx) ** 2 + (y - cy) ** 2)
            feats["radius_of_gyration"] = float(np.sqrt(rg2))

            # segment_density: number of points per second (or segments per second)
            if duration_sec and duration_sec > 0:
                feats["segment_density"] = float((n - 1) / duration_sec)

            # heading_entropy: entropy of heading angles distribution
            angles = np.arctan2(dy, dx)  # [-pi, pi]
            # ignore NaNs (shouldn't happen) and optionally ignore zero-length segments
            if angles.size > 0:
                # 16 bins over [-pi, pi]
                hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi))
                p = hist.astype(float)
                s = p.sum()
                if s > 0:
                    p /= s
                    p = p[p > 0]
                    feats["heading_entropy"] = float(-(p * np.log(p)).sum())

        # -------------------------
        # VERTICAL behavior (alt + rv)
        # -------------------------
        # alt_cv
        if dim >= 3:
            alt = coords[:, 2]
            alt_mean = float(np.mean(alt))
            alt_std = float(np.std(alt))
            if abs(alt_mean) > 0:
                feats["alt_cv"] = float(alt_std / (abs(alt_mean) + 1e-6))

        # rv-based
        if dim >= 4:
            rv = coords[:, 3]
            abs_rv = np.abs(rv)

            abs_rv_mean = float(np.mean(abs_rv))
            rv_std = float(np.std(rv))

            # rv_cv = rv_std / abs_rv_mean
            if abs_rv_mean > 0:
                feats["rv_cv"] = float(rv_std / (abs_rv_mean + 1e-6))

            # vertical_activity_ratio = abs_rv_mean / mean horizontal speed
            if dim >= 2 and n >= 2:
                x = coords[:, 0]
                y = coords[:, 1]
                dx = np.diff(x)
                dy = np.diff(y)
                speed_xy = np.sqrt(dx*dx + dy*dy)
                mean_speed_xy = float(np.mean(speed_xy)) if speed_xy.size else np.nan
                if mean_speed_xy and mean_speed_xy > 0:
                    feats["vertical_activity_ratio"] = float(abs_rv_mean / (mean_speed_xy + 1e-6))

            # climb_descent_ratio = total positive rv / total abs negative rv
            pos_sum = float(np.sum(rv[rv > 0])) if np.any(rv > 0) else 0.0
            neg_sum = float(np.sum(-rv[rv < 0])) if np.any(rv < 0) else 0.0
            if neg_sum > 0:
                feats["climb_descent_ratio"] = float(pos_sum / (neg_sum + 1e-6))
            else:
                # if there's no descent at all but climb exists -> big ratio
                if pos_sum > 0:
                    feats["climb_descent_ratio"] = float(pos_sum / 1e-6)
                else:
                    feats["climb_descent_ratio"] = np.nan

        return feats

    except Exception:
        return feats

train = pd.read_parquet("../../data/interim/train/train_4.parquet")
beh_df = train.apply(
    lambda row: extract_behavior_structure_features(row["trajectory"], row["duration_sec"]),
    axis=1
).apply(pd.Series)
print(beh_df.columns)

train = pd.concat([train, beh_df], axis=1)
train.to_parquet("../../data/interim/train/train_5.parquet", index=False)