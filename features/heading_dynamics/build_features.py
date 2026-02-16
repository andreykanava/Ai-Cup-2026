import binascii
from shapely import wkb
import numpy as np
import pandas as pd


def extract_heading_features(hex_wkb, eps=1e-12, bins=16):
    """
    Heading dynamics features from ONE WKB parse.
    Uses only XY (x,y) coords.

    Returns dict[str, float|int]
    """
    feats = {
        # circular mean / dispersion
        "heading_mean": np.nan,
        "heading_r": np.nan,              # mean resultant length (0..1), higher = more consistent direction
        "heading_circ_std": np.nan,       # circular std proxy

        # heading distribution
        "heading_entropy": np.nan,        # entropy over heading histogram
        "heading_mode_bin": np.nan,       # most frequent bin index [0..bins-1]
        "heading_mode_ratio": np.nan,     # share of samples in modal bin

        # turn dynamics (delta heading)
        "turn_abs_mean": np.nan,
        "turn_abs_std": np.nan,
        "turn_abs_p90": np.nan,
        "turn_abs_p99": np.nan,
        "turn_rate_cv": np.nan,           # std/mean of |dtheta| (not per second)

        # “zigzag / flip” style
        "turn_sign_change_ratio": np.nan, # how often direction of turning switches
        "large_turn_ratio": np.nan,       # fraction |dtheta| > threshold

        # persistence / correlation
        "heading_acf1_cos": np.nan,       # autocorr of cos(theta)
        "heading_acf1_sin": np.nan,       # autocorr of sin(theta)
        "turn_acf1": np.nan,              # autocorr of |dtheta|

        # directional projection features
        "mean_cos_heading": np.nan,
        "mean_sin_heading": np.nan,
        "mean_cos_turn": np.nan,          # mean cos(dtheta) -> ~1 straight, smaller = wiggly

        # integrated turning
        "total_turn": np.nan,             # sum |dtheta|
        "net_turn": np.nan,               # abs(sum signed dtheta)
        "turn_efficiency": np.nan,        # net_turn / total_turn
    }

    def _acf1(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            return np.nan
        x = x - x.mean()
        denom = np.dot(x, x) + eps
        return float(np.dot(x[1:], x[:-1]) / denom)

    try:
        geom = wkb.loads(binascii.unhexlify(hex_wkb))
        coords = np.asarray(geom.coords, dtype=float)

        if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
            return feats

        x = coords[:, 0]
        y = coords[:, 1]

        dx = np.diff(x)
        dy = np.diff(y)

        seg_len = np.sqrt(dx*dx + dy*dy)

        # need at least 2 segments for turns
        if seg_len.size < 2:
            return feats

        # heading angles per segment
        theta = np.arctan2(dy, dx)                # [-pi, pi]
        theta_u = np.unwrap(theta)

        # circular mean using sin/cos
        s = np.sin(theta)
        c = np.cos(theta)
        mean_s = float(np.mean(s))
        mean_c = float(np.mean(c))
        R = float(np.sqrt(mean_s**2 + mean_c**2))  # mean resultant length

        feats["mean_sin_heading"] = mean_s
        feats["mean_cos_heading"] = mean_c

        feats["heading_mean"] = float(np.arctan2(mean_s, mean_c))
        feats["heading_r"] = R
        feats["heading_circ_std"] = float(np.sqrt(-2.0 * np.log(R + eps))) if R > 0 else np.nan

        # heading entropy (histogram on [0,2pi))
        if theta.size >= 8:
            th = (theta + np.pi) % (2*np.pi)
            hist, _ = np.histogram(th, bins=bins, range=(0, 2*np.pi))
            total = hist.sum()
            if total > 0:
                p = hist / (total + eps)
                feats["heading_entropy"] = float(-np.sum(p * np.log(p + eps)) / np.log(bins))
                mode_bin = int(np.argmax(hist))
                feats["heading_mode_bin"] = mode_bin
                feats["heading_mode_ratio"] = float(hist[mode_bin] / (total + eps))

        # turns
        dtheta = np.diff(theta_u)  # signed
        abs_dtheta = np.abs(dtheta)

        if abs_dtheta.size:
            feats["turn_abs_mean"] = float(np.mean(abs_dtheta))
            feats["turn_abs_std"] = float(np.std(abs_dtheta))
            feats["turn_abs_p90"] = float(np.percentile(abs_dtheta, 90))
            feats["turn_abs_p99"] = float(np.percentile(abs_dtheta, 99))

            m = feats["turn_abs_mean"]
            if np.isfinite(m) and m > eps:
                feats["turn_rate_cv"] = float(feats["turn_abs_std"] / (m + eps))

            # big turns ratio: threshold = 90th percentile or fixed rad threshold
            thr = feats["turn_abs_p90"]
            if np.isfinite(thr):
                feats["large_turn_ratio"] = float(np.mean(abs_dtheta > thr))

            # turning direction switches (zigzag)
            # ignore near-zero turns
            signed = np.sign(dtheta)
            nz = signed != 0
            signed_nz = signed[nz]
            if signed_nz.size >= 2:
                feats["turn_sign_change_ratio"] = float(np.mean(signed_nz[1:] != signed_nz[:-1]))

            # persistence proxy: mean cos(dtheta)
            feats["mean_cos_turn"] = float(np.mean(np.cos(dtheta)))

            # integrated turning
            total_turn = float(np.sum(abs_dtheta))
            net_turn = float(np.abs(np.sum(dtheta)))
            feats["total_turn"] = total_turn
            feats["net_turn"] = net_turn
            feats["turn_efficiency"] = float(net_turn / (total_turn + eps)) if total_turn > 0 else np.nan

            # ACFs
            feats["turn_acf1"] = _acf1(abs_dtheta)

        feats["heading_acf1_cos"] = _acf1(np.cos(theta))
        feats["heading_acf1_sin"] = _acf1(np.sin(theta))

        return feats

    except Exception:
        return feats


train = pd.read_parquet("../../data/interim/test/test_8.parquet")

feat_head = train["trajectory"].apply(extract_heading_features).apply(pd.Series).add_prefix("hd_")
train = pd.concat([train, feat_head], axis=1)
train = train.loc[:, ~train.columns.duplicated(keep="first")]

train.to_parquet("../../data/interim/test/test_9.parquet", index=False)