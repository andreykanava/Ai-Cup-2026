import binascii
from shapely import wkb
import pandas as pd
import numpy as np




def extract_features(hex_wkb, stop_eps=1e-6, eps=1e-12):
    """
    Extract a bunch of trajectory features in ONE WKB parse.

    Expected coord layout (example you gave):
        (x, y, alt, rv)
    where x=lon, y=lat (or any planar coords), alt=coords[:,2], rv=coords[:,3].

    Returns: dict[str, float|int]
    """
    feats = {
        # Alt
        "alt_mean": np.nan,
        "alt_std": np.nan,
        "alt_max": np.nan,
        "alt_min": np.nan,
        "alt_range": np.nan,
        "climb_rate_mean": np.nan,
        "climb_rate_std": np.nan,

        # RV
        "rv_mean": np.nan,
        "rv_std": np.nan,
        "rv_max": np.nan,
        "rv_min": np.nan,
        "rv_range": np.nan,
        "rv_acc_mean": np.nan,
        "rv_acc_std": np.nan,
        "abs_rv_mean": np.nan,
        "abs_rv_max": np.nan,
        "positive_rv_ratio": np.nan,
        "negative_rv_ratio": np.nan,
        "climb_segments_count": np.nan,
        "descent_segments_count": np.nan,
        "rv_sign_changes": np.nan,

        # XY
        "path_length": np.nan,
        "displacement": np.nan,
        "speed_mean": np.nan,
        "speed_std": np.nan,
        "max_speed": np.nan,
        "stop_ratio": np.nan,
        "turn_rate_mean": np.nan,
        "turn_rate_std": np.nan,
        "tortuosity": np.nan,
        "straightness_index": np.nan,
        "curvature_mean": np.nan,
    }

    try:
        geom = wkb.loads(binascii.unhexlify(hex_wkb))
        coords = np.asarray(geom.coords, dtype=float)

        if coords.ndim != 2 or coords.shape[0] == 0:
            return feats

        n = coords.shape[0]
        dim = coords.shape[1]

        # --- ALT ---
        if dim >= 3:
            alt = coords[:, 2]
            feats["alt_mean"] = float(np.mean(alt))
            feats["alt_std"]  = float(np.std(alt))
            feats["alt_max"]  = float(np.max(alt))
            feats["alt_min"]  = float(np.min(alt))
            feats["alt_range"] = float(np.max(alt) - np.min(alt))

            if n >= 2:
                dalt = np.diff(alt)
                feats["climb_rate_mean"] = float(np.mean(dalt))
                feats["climb_rate_std"]  = float(np.std(dalt))

        # --- RV ---
        if dim >= 4:
            rv = coords[:, 3]
            feats["rv_mean"] = float(np.mean(rv))
            feats["rv_std"]  = float(np.std(rv))
            feats["rv_max"]  = float(np.max(rv))
            feats["rv_min"]  = float(np.min(rv))
            feats["rv_range"] = float(np.max(rv) - np.min(rv))

            abs_rv = np.abs(rv)
            feats["abs_rv_mean"] = float(np.mean(abs_rv))
            feats["abs_rv_max"]  = float(np.max(abs_rv))

            if n > 0:
                feats["positive_rv_ratio"] = float(np.sum(rv > 0) / n)
                feats["negative_rv_ratio"] = float(np.sum(rv < 0) / n)

            if n >= 2:
                drv = np.diff(rv)
                feats["rv_acc_mean"] = float(np.mean(drv))
                feats["rv_acc_std"]  = float(np.std(drv))

                # segment counts: count rising edges into >0 and <0 states
                pos = rv > 0
                neg = rv < 0
                feats["climb_segments_count"] = int(np.sum(pos[1:] & (~pos[:-1])))
                feats["descent_segments_count"] = int(np.sum(neg[1:] & (~neg[:-1])))

                # sign changes (ignore exact zeros by treating sign(0)=0; still counts transitions)
                sgn = np.sign(rv)
                feats["rv_sign_changes"] = int(np.sum(sgn[1:] != sgn[:-1]))

        # --- XY / GEOMETRY ---
        if dim >= 2 and n >= 2:
            x = coords[:, 0]
            y = coords[:, 1]

            dx = np.diff(x)
            dy = np.diff(y)

            seg_len = np.sqrt(dx*dx + dy*dy)
            path_length = float(np.sum(seg_len))
            feats["path_length"] = path_length

            displacement = float(np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))
            feats["displacement"] = displacement

            feats["speed_mean"] = float(np.mean(seg_len)) if seg_len.size else np.nan
            feats["speed_std"]  = float(np.std(seg_len)) if seg_len.size else np.nan
            feats["max_speed"]  = float(np.max(seg_len)) if seg_len.size else np.nan
            feats["stop_ratio"] = float(np.mean(seg_len <= stop_eps)) if seg_len.size else np.nan

            # Tortuosity / straightness
            if displacement > 0 and path_length > 0:
                feats["tortuosity"] = float(path_length / displacement)
                feats["straightness_index"] = float(displacement / path_length)
            else:
                feats["tortuosity"] = np.nan
                feats["straightness_index"] = np.nan

            # Turn rate: based on heading angles between segments
            if seg_len.size >= 2:
                angles = np.arctan2(dy, dx)
                angles = np.unwrap(angles)
                dtheta = np.abs(np.diff(angles))

                feats["turn_rate_mean"] = float(np.mean(dtheta)) if dtheta.size else np.nan
                feats["turn_rate_std"]  = float(np.std(dtheta)) if dtheta.size else np.nan

                # Curvature ~ |Δθ| / segment_length (use seg_len[1:] aligned with dtheta)
                denom = seg_len[1:] + eps
                curvature = dtheta / denom
                feats["curvature_mean"] = float(np.mean(curvature)) if curvature.size else np.nan

        return feats

    except Exception:
        return feats

train = pd.read_csv("../../data/raw/train.csv")
feat_df = train["trajectory"].apply(extract_features).apply(pd.Series)
train = pd.concat([train, feat_df], axis=1)

print(train.filter(regex="^(alt_|rv_|abs_rv_|path_|disp|speed_|turn_|tort|straight|curv)").head())

train.to_parquet("../../data/interim/train/train_1.parquet", index=False)
