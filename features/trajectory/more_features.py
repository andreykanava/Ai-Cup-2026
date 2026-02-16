import binascii
from shapely import wkb
from shapely.geometry import LineString
import numpy as np
import pandas as pd


def extract_species_features(hex_wkb, eps=1e-12, stop_eps=1e-6):
    """
    More species-oriented trajectory features from ONE WKB parse.
    Expected coord layout: (x, y, alt, rv) if available.
    Returns: dict[str, float|int]
    """
    feats = {
        # --- Distribution stats (speed / turn / curvature / rv / alt) ---
        "speed_p10": np.nan, "speed_p50": np.nan, "speed_p90": np.nan, "speed_iqr": np.nan,
        "speed_skew": np.nan, "speed_kurt": np.nan, "speed_burst_ratio_p90": np.nan,

        "rv_p10": np.nan, "rv_p50": np.nan, "rv_p90": np.nan, "rv_iqr": np.nan,
        "rv_skew": np.nan, "rv_kurt": np.nan,
        "abs_rv_p90": np.nan,

        "alt_p10": np.nan, "alt_p50": np.nan, "alt_p90": np.nan, "alt_iqr": np.nan,
        "alt_skew": np.nan, "alt_kurt": np.nan,

        "turn_p10": np.nan, "turn_p50": np.nan, "turn_p90": np.nan, "turn_iqr": np.nan,
        "turn_skew": np.nan, "turn_kurt": np.nan,

        "curv_p10": np.nan, "curv_p50": np.nan, "curv_p90": np.nan, "curv_iqr": np.nan,
        "curv_skew": np.nan, "curv_kurt": np.nan,

        # --- Heading / geometry ---
        "heading_mean": np.nan,
        "heading_std": np.nan,
        "heading_entropy": np.nan,           # circular-ish histogram entropy
        "bearing_change_rate": np.nan,       # mean |Δθ|
        "bearing_change_p90": np.nan,

        "bbox_area": np.nan,
        "bbox_aspect_ratio": np.nan,
        "pca_elongation": np.nan,            # lambda1 / lambda2
        "hull_area": np.nan,
        "hull_perimeter": np.nan,
        "hull_to_bbox_ratio": np.nan,

        # --- Jerk (acceleration derivative proxy) ---
        "jerk_xy_mean": np.nan,
        "jerk_xy_std": np.nan,
        "jerk_xy_p90": np.nan,
        "jerk_rv_mean": np.nan,
        "jerk_rv_std": np.nan,
        "jerk_rv_p90": np.nan,

        # --- Autocorrelation / spectral ---
        "speed_acf1": np.nan,
        "rv_acf1": np.nan,
        "speed_dom_freq": np.nan,
        "speed_spec_entropy": np.nan,
        "rv_dom_freq": np.nan,
        "rv_spec_entropy": np.nan,

        # --- Episodes / states ---
        "climb_episode_count": np.nan,
        "descent_episode_count": np.nan,
        "climb_episode_mean_len": np.nan,
        "climb_episode_max_len": np.nan,
        "descent_episode_mean_len": np.nan,
        "descent_episode_max_len": np.nan,
        "vertical_burst_ratio_p90": np.nan,  # fraction |rv| > p90(|rv|)
    }

    def _skew_kurt(x):
        # population skewness/kurtosis (not Fisher corrected), robust for small n
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        n = x.size
        if n < 3:
            return np.nan, np.nan
        mu = x.mean()
        m2 = np.mean((x - mu) ** 2)
        if m2 <= 0:
            return np.nan, np.nan
        m3 = np.mean((x - mu) ** 3)
        m4 = np.mean((x - mu) ** 4)
        skew = m3 / (m2 ** 1.5 + eps)
        kurt = m4 / (m2 ** 2 + eps)
        return float(skew), float(kurt)

    def _quantiles_iqr(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return (np.nan, np.nan, np.nan, np.nan)
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        p25, p75 = np.percentile(x, [25, 75])
        return float(p10), float(p50), float(p90), float(p75 - p25)

    def _acf1(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            return np.nan
        x = x - x.mean()
        denom = np.dot(x, x) + eps
        return float(np.dot(x[1:], x[:-1]) / denom)

    def _spec_feats(x):
        """
        Return (dominant_freq_index, spectral_entropy) using FFT magnitudes.
        Since you don't pass time deltas, freq is 'per-sample'. Still useful.
        """
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size < 8:
            return np.nan, np.nan
        x = x - x.mean()
        # rfft -> positive freqs
        mag = np.abs(np.fft.rfft(x))
        mag[0] = 0.0  # remove DC
        power = mag ** 2
        s = power.sum()
        if s <= 0:
            return np.nan, np.nan
        p = power / (s + eps)
        spec_entropy = -np.sum(p * np.log(p + eps)) / (np.log(p.size + eps))
        dom = int(np.argmax(power))
        return float(dom), float(spec_entropy)

    def _episode_lengths(mask):
        """
        mask: boolean array length n (states per point)
        returns: (count, mean_len, max_len) in samples
        """
        mask = np.asarray(mask, bool)
        if mask.size < 2:
            return 0, np.nan, np.nan
        # find runs of True
        d = np.diff(mask.astype(int))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, mask.size]
        if starts.size == 0 or ends.size == 0:
            return 0, np.nan, np.nan
        lens = (ends - starts).astype(float)
        return int(lens.size), float(lens.mean()), float(lens.max())

    try:
        geom = wkb.loads(binascii.unhexlify(hex_wkb))
        coords = np.asarray(geom.coords, dtype=float)

        if coords.ndim != 2 or coords.shape[0] < 2:
            return feats

        n = coords.shape[0]
        dim = coords.shape[1]

        x = coords[:, 0]
        y = coords[:, 1]

        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.sqrt(dx * dx + dy * dy)

        # Heading / turn series
        if seg_len.size >= 1:
            angles = np.arctan2(dy, dx)
            angles_u = np.unwrap(angles)
            dtheta = np.abs(np.diff(angles_u)) if angles_u.size >= 2 else np.array([], dtype=float)

            # heading mean/std (use sin/cos mean for circular mean)
            s = np.sin(angles)
            c = np.cos(angles)
            mean_ang = np.arctan2(np.mean(s), np.mean(c))
            # circular std proxy: 1 - R
            R = np.sqrt(np.mean(s)**2 + np.mean(c)**2)
            circ_std = np.sqrt(-2.0 * np.log(R + eps)) if R > 0 else np.nan

            feats["heading_mean"] = float(mean_ang)
            feats["heading_std"] = float(circ_std)

            # heading entropy over bins
            if angles.size >= 8:
                bins = 16
                hist, _ = np.histogram((angles + np.pi) % (2*np.pi), bins=bins, range=(0, 2*np.pi))
                p = hist / (hist.sum() + eps)
                feats["heading_entropy"] = float(-np.sum(p * np.log(p + eps)) / np.log(bins))

            # bearing changes
            if dtheta.size:
                feats["bearing_change_rate"] = float(np.mean(dtheta))
                feats["bearing_change_p90"] = float(np.percentile(dtheta, 90))

            # turn-rate distribution stats from dtheta
            if dtheta.size >= 2:
                p10, p50, p90, iqr = _quantiles_iqr(dtheta)
                feats.update({
                    "turn_p10": p10, "turn_p50": p50, "turn_p90": p90, "turn_iqr": iqr
                })
                sk, ku = _skew_kurt(dtheta)
                feats["turn_skew"] = sk
                feats["turn_kurt"] = ku

            # curvature series: dtheta / seg_len[1:]
            if dtheta.size >= 1 and seg_len.size >= 2:
                curv = dtheta / (seg_len[1:] + eps)
                if curv.size >= 2:
                    p10, p50, p90, iqr = _quantiles_iqr(curv)
                    feats.update({
                        "curv_p10": p10, "curv_p50": p50, "curv_p90": p90, "curv_iqr": iqr
                    })
                    sk, ku = _skew_kurt(curv)
                    feats["curv_skew"] = sk
                    feats["curv_kurt"] = ku

        # Speed distribution
        if seg_len.size >= 2:
            p10, p50, p90, iqr = _quantiles_iqr(seg_len)
            feats.update({"speed_p10": p10, "speed_p50": p50, "speed_p90": p90, "speed_iqr": iqr})
            sk, ku = _skew_kurt(seg_len)
            feats["speed_skew"] = sk
            feats["speed_kurt"] = ku
            feats["speed_burst_ratio_p90"] = float(np.mean(seg_len > p90))

            feats["speed_acf1"] = _acf1(seg_len)
            dom, se = _spec_feats(seg_len)
            feats["speed_dom_freq"] = dom
            feats["speed_spec_entropy"] = se

        # Geometry: bbox + PCA elongation + hull
        if n >= 3:
            x_min, x_max = float(np.min(x)), float(np.max(x))
            y_min, y_max = float(np.min(y)), float(np.max(y))
            w = (x_max - x_min)
            h = (y_max - y_min)
            feats["bbox_area"] = float(w * h)
            feats["bbox_aspect_ratio"] = float((max(w, h) + eps) / (min(w, h) + eps))

            # PCA elongation
            pts = np.column_stack([x, y])
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            if pts.shape[0] >= 3:
                pts_c = pts - pts.mean(axis=0, keepdims=True)
                cov = np.cov(pts_c.T)
                vals = np.linalg.eigvalsh(cov)  # ascending
                lam1 = float(vals[-1])
                lam2 = float(vals[-2]) if vals.size >= 2 else np.nan
                if np.isfinite(lam1) and np.isfinite(lam2):
                    feats["pca_elongation"] = float((lam1 + eps) / (lam2 + eps))

            # Convex hull features
            try:
                line = LineString(np.column_stack([x, y]))
                hull = line.convex_hull
                feats["hull_area"] = float(getattr(hull, "area", np.nan))
                feats["hull_perimeter"] = float(getattr(hull, "length", np.nan))
                if np.isfinite(feats["bbox_area"]) and feats["bbox_area"] > 0:
                    feats["hull_to_bbox_ratio"] = float(feats["hull_area"] / (feats["bbox_area"] + eps))
            except Exception:
                pass

        # RV / ALT distributions + jerk + episodes
        if dim >= 4:
            rv = coords[:, 3]
            rv = rv[np.isfinite(rv)]
            if rv.size >= 2:
                p10, p50, p90, iqr = _quantiles_iqr(rv)
                feats.update({"rv_p10": p10, "rv_p50": p50, "rv_p90": p90, "rv_iqr": iqr})
                sk, ku = _skew_kurt(rv)
                feats["rv_skew"] = sk
                feats["rv_kurt"] = ku

                abs_rv = np.abs(rv)
                feats["abs_rv_p90"] = float(np.percentile(abs_rv, 90)) if abs_rv.size else np.nan

                feats["rv_acf1"] = _acf1(rv)
                dom, se = _spec_feats(rv)
                feats["rv_dom_freq"] = dom
                feats["rv_spec_entropy"] = se

                # vertical burstiness: |rv| > p90(|rv|)
                thr = np.percentile(abs_rv, 90)
                feats["vertical_burst_ratio_p90"] = float(np.mean(abs_rv > thr)) if np.isfinite(thr) else np.nan

                # jerk in rv: diff(diff(rv))
                if rv.size >= 3:
                    jrv = np.diff(np.diff(rv))
                    feats["jerk_rv_mean"] = float(np.mean(np.abs(jrv)))
                    feats["jerk_rv_std"] = float(np.std(jrv))
                    feats["jerk_rv_p90"] = float(np.percentile(np.abs(jrv), 90))

                # episodes from sign of rv
                rv_full = coords[:, 3]
                rv_full = rv_full[np.isfinite(rv_full)]
                if rv_full.size >= 2:
                    climb_mask = rv_full > 0
                    desc_mask = rv_full < 0
                    c_cnt, c_mean, c_max = _episode_lengths(climb_mask)
                    d_cnt, d_mean, d_max = _episode_lengths(desc_mask)
                    feats["climb_episode_count"] = c_cnt
                    feats["climb_episode_mean_len"] = c_mean
                    feats["climb_episode_max_len"] = c_max
                    feats["descent_episode_count"] = d_cnt
                    feats["descent_episode_mean_len"] = d_mean
                    feats["descent_episode_max_len"] = d_max

        if dim >= 3:
            alt = coords[:, 2]
            alt = alt[np.isfinite(alt)]
            if alt.size >= 2:
                p10, p50, p90, iqr = _quantiles_iqr(alt)
                feats.update({"alt_p10": p10, "alt_p50": p50, "alt_p90": p90, "alt_iqr": iqr})
                sk, ku = _skew_kurt(alt)
                feats["alt_skew"] = sk
                feats["alt_kurt"] = ku

        # Jerk_xy: use second diff of x,y as crude jerk proxy
        if n >= 4:
            # velocity approx: diff positions
            vx = np.diff(x)
            vy = np.diff(y)
            # acceleration approx: diff velocity
            ax = np.diff(vx)
            ay = np.diff(vy)
            # jerk approx: diff acceleration
            jx = np.diff(ax)
            jy = np.diff(ay)
            if jx.size >= 2:
                jmag = np.sqrt(jx*jx + jy*jy)
                feats["jerk_xy_mean"] = float(np.mean(jmag))
                feats["jerk_xy_std"] = float(np.std(jmag))
                feats["jerk_xy_p90"] = float(np.percentile(jmag, 90))

        return feats

    except Exception:
        return feats



train = pd.read_parquet("../../data/interim/test/test_6.parquet")

feat_df2 = train["trajectory"].apply(extract_species_features).apply(pd.Series)

# выкидываем те, что уже есть в train
feat_df2 = feat_df2.loc[:, ~feat_df2.columns.isin(train.columns)]

train = pd.concat([train, feat_df2], axis=1)

# на всякий случай прибиваем дубли (если они были уже в train_6)
train = train.loc[:, ~train.columns.duplicated(keep="first")]

train.to_parquet("../../data/interim/test/test_7.parquet", index=False)
