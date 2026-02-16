import binascii
from shapely import wkb
import numpy as np
import pandas as pd


def extract_energy_features(hex_wkb, eps=1e-12):
    """
    Energy / physics based features from trajectory.

    coord format expected:
        (x, y, alt, rv)

    Returns dict[str, float]
    """

    feats = {
        # horizontal kinetic energy proxy
        "energy_xy_mean": np.nan,
        "energy_xy_std": np.nan,
        "energy_xy_max": np.nan,
        "energy_xy_p90": np.nan,

        # vertical kinetic energy proxy
        "energy_vertical_mean": np.nan,
        "energy_vertical_std": np.nan,
        "energy_vertical_max": np.nan,
        "energy_vertical_p90": np.nan,

        # ratios
        "energy_vertical_horizontal_ratio": np.nan,
        "energy_total_mean": np.nan,
        "energy_total_std": np.nan,

        # acceleration energy
        "acc_energy_xy_mean": np.nan,
        "acc_energy_xy_std": np.nan,
        "acc_energy_xy_max": np.nan,
        "acc_energy_xy_p90": np.nan,

        "acc_energy_vertical_mean": np.nan,
        "acc_energy_vertical_std": np.nan,
        "acc_energy_vertical_max": np.nan,
        "acc_energy_vertical_p90": np.nan,

        # jerk energy
        "jerk_energy_xy_mean": np.nan,
        "jerk_energy_xy_std": np.nan,
        "jerk_energy_xy_max": np.nan,

        "jerk_energy_vertical_mean": np.nan,
        "jerk_energy_vertical_std": np.nan,
        "jerk_energy_vertical_max": np.nan,
    }

    try:
        geom = wkb.loads(binascii.unhexlify(hex_wkb))
        coords = np.asarray(geom.coords, dtype=float)

        if coords.ndim != 2 or coords.shape[0] < 3:
            return feats

        n = coords.shape[0]
        dim = coords.shape[1]

        x = coords[:, 0]
        y = coords[:, 1]

        dx = np.diff(x)
        dy = np.diff(y)

        speed_xy = np.sqrt(dx*dx + dy*dy)

        # ------------------------
        # horizontal energy
        # ------------------------

        if speed_xy.size >= 2:

            energy_xy = speed_xy ** 2

            feats["energy_xy_mean"] = float(np.mean(energy_xy))
            feats["energy_xy_std"]  = float(np.std(energy_xy))
            feats["energy_xy_max"]  = float(np.max(energy_xy))
            feats["energy_xy_p90"]  = float(np.percentile(energy_xy, 90))

        # ------------------------
        # vertical energy
        # ------------------------

        if dim >= 4:

            rv = coords[:, 3]
            rv = rv[np.isfinite(rv)]

            if rv.size >= 2:

                energy_vertical = rv ** 2

                feats["energy_vertical_mean"] = float(np.mean(energy_vertical))
                feats["energy_vertical_std"]  = float(np.std(energy_vertical))
                feats["energy_vertical_max"]  = float(np.max(energy_vertical))
                feats["energy_vertical_p90"]  = float(np.percentile(energy_vertical, 90))

        # ------------------------
        # total energy + ratio
        # ------------------------

        if speed_xy.size >= 2 and dim >= 4:

            rv = coords[:, 3]

            min_len = min(speed_xy.size, rv.size)

            total_energy = speed_xy[:min_len]**2 + rv[:min_len]**2

            feats["energy_total_mean"] = float(np.mean(total_energy))
            feats["energy_total_std"]  = float(np.std(total_energy))

            vertical_mean = feats["energy_vertical_mean"]
            horizontal_mean = feats["energy_xy_mean"]

            if horizontal_mean > eps:
                feats["energy_vertical_horizontal_ratio"] = float(
                    vertical_mean / horizontal_mean
                )

        # ------------------------
        # acceleration energy
        # ------------------------

        if speed_xy.size >= 3:

            acc_xy = np.diff(speed_xy)
            acc_energy_xy = acc_xy ** 2

            feats["acc_energy_xy_mean"] = float(np.mean(acc_energy_xy))
            feats["acc_energy_xy_std"]  = float(np.std(acc_energy_xy))
            feats["acc_energy_xy_max"]  = float(np.max(acc_energy_xy))
            feats["acc_energy_xy_p90"]  = float(np.percentile(acc_energy_xy, 90))

        if dim >= 4:

            rv = coords[:, 3]

            if rv.size >= 3:

                acc_rv = np.diff(rv)
                acc_energy_vertical = acc_rv ** 2

                feats["acc_energy_vertical_mean"] = float(np.mean(acc_energy_vertical))
                feats["acc_energy_vertical_std"]  = float(np.std(acc_energy_vertical))
                feats["acc_energy_vertical_max"]  = float(np.max(acc_energy_vertical))
                feats["acc_energy_vertical_p90"]  = float(np.percentile(acc_energy_vertical, 90))

        # ------------------------
        # jerk energy
        # ------------------------

        if speed_xy.size >= 4:

            jerk_xy = np.diff(np.diff(speed_xy))
            jerk_energy_xy = jerk_xy ** 2

            feats["jerk_energy_xy_mean"] = float(np.mean(jerk_energy_xy))
            feats["jerk_energy_xy_std"]  = float(np.std(jerk_energy_xy))
            feats["jerk_energy_xy_max"]  = float(np.max(jerk_energy_xy))

        if dim >= 4:

            rv = coords[:, 3]

            if rv.size >= 4:

                jerk_rv = np.diff(np.diff(rv))
                jerk_energy_vertical = jerk_rv ** 2

                feats["jerk_energy_vertical_mean"] = float(np.mean(jerk_energy_vertical))
                feats["jerk_energy_vertical_std"]  = float(np.std(jerk_energy_vertical))
                feats["jerk_energy_vertical_max"]  = float(np.max(jerk_energy_vertical))

        return feats

    except Exception:
        return feats


train = pd.read_parquet("../../data/interim/train/train_7.parquet")

feat_energy = train["trajectory"].apply(extract_energy_features).apply(pd.Series)

feat_energy = feat_energy.loc[:, ~feat_energy.columns.isin(train.columns)]

train = pd.concat([train, feat_energy], axis=1)

train.to_parquet("../../data/interim/train/train_8.parquet", index=False)