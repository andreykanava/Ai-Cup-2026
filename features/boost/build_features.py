import binascii
from shapely import wkb
import numpy as np
import pandas as pd


def extract_species_boost_features(hex_wkb, trajectory_time=None, eps=1e-12):
    """
    Extra strong features for bird species detection from ONE WKB parse.
    Supports 2D/3D LineString. Uses trajectory_time if provided.

    Returns dict[str, float|int]
    """
    feats = {
        "vertical_oscillation_freq": np.nan,
        "speed_acf5": np.nan,
        "heading_acf10": np.nan,
        "fractal_dimension": np.nan,
        "maneuver_intensity": np.nan,
        "control_smoothness": np.nan,
        "state_transition_entropy": np.nan,
        "glide_ratio": np.nan,
        "speed_peak_count": np.nan,
        "acf_decay_tau": np.nan,
        "trajectory_planarity": np.nan,
        "energy_per_distance": np.nan,
    }

    def _to_time_array(t, n_pts):
        # expects t as list/np array of timestamps (seconds) length n_pts
        if t is None:
            return None
        try:
            t = np.asarray(t, dtype=float)
            if t.ndim != 1 or t.size != n_pts:
                return None
            # ensure monotonic-ish; if broken, fallback None
            if not np.all(np.isfinite(t)) or np.nanmax(np.diff(t)) <= 0:
                return None
            return t
        except Exception:
            return None

    def _acf_lag(x, lag):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size <= lag + 2:
            return np.nan
        x = x - np.mean(x)
        denom = np.dot(x, x) + eps
        return float(np.dot(x[lag:], x[:-lag]) / denom)

    def _dominant_freq(signal, dt):
        # FFT dominant frequency excluding DC; returns Hz
        s = np.asarray(signal, dtype=float)
        s = s[np.isfinite(s)]
        if s.size < 16 or not np.isfinite(dt) or dt <= 0:
            return np.nan
        s = s - np.mean(s)
        # if almost flat
        if np.std(s) < 1e-9:
            return np.nan
        fft = np.fft.rfft(s)
        freqs = np.fft.rfftfreq(s.size, d=dt)
        mag = np.abs(fft)
        if mag.size < 3:
            return np.nan
        mag[0] = 0.0  # remove DC
        k = int(np.argmax(mag))
        f = float(freqs[k])
        # ignore implausibly high frequencies (noise) if dt is weird
        return f if f > 0 else np.nan

    def _katz_fd(x, y, z=None):
        # Katz fractal dimension
        if z is None:
            pts = np.column_stack([x, y])
        else:
            pts = np.column_stack([x, y, z])
        n = pts.shape[0]
        if n < 3:
            return np.nan
        dif = np.diff(pts, axis=0)
        seg = np.linalg.norm(dif, axis=1)
        L = float(np.sum(seg))
        if not np.isfinite(L) or L <= eps:
            return np.nan
        d = float(np.max(np.linalg.norm(pts - pts[0], axis=1)))
        if not np.isfinite(d) or d <= eps:
            return np.nan
        # Katz FD
        return float(np.log10(n) / (np.log10(d / (L + eps)) + np.log10(n) + eps))

    def _planarity_pca(x, y, z):
        # PCA eigenvalues of centered 3D points; planarity in [0..1] approx
        pts = np.column_stack([x, y, z]).astype(float)
        if pts.shape[0] < 5:
            return np.nan
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] < 5:
            return np.nan
        pts = pts - pts.mean(axis=0, keepdims=True)
        C = np.cov(pts.T)
        if C.shape != (3, 3) or not np.all(np.isfinite(C)):
            return np.nan
        w = np.linalg.eigvalsh(C)
        w = np.sort(w)[::-1]  # λ1 >= λ2 >= λ3
        if w[0] <= eps:
            return np.nan
        # common "planarity" proxy: (λ2 - λ3)/λ1
        return float((w[1] - w[2]) / (w[0] + eps))

    try:
        geom = wkb.loads(binascii.unhexlify(hex_wkb))
        coords = np.asarray(geom.coords, dtype=float)

        if coords.ndim != 2 or coords.shape[0] < 6 or coords.shape[1] < 2:
            return feats

        x = coords[:, 0]
        y = coords[:, 1]
        has_z = coords.shape[1] >= 3
        z = coords[:, 2] if has_z else None

        # time
        t = _to_time_array(trajectory_time, coords.shape[0])
        if t is None:
            # uniform time step
            dt_seg = np.ones(coords.shape[0] - 1, dtype=float)
            dt_mean = 1.0
        else:
            dt_seg = np.diff(t)
            dt_seg = np.where((dt_seg > 0) & np.isfinite(dt_seg), dt_seg, np.nan)
            dt_mean = float(np.nanmedian(dt_seg)) if np.isfinite(np.nanmedian(dt_seg)) else 1.0

        # segment deltas
        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.sqrt(dx * dx + dy * dy)
        if seg_len.size < 5:
            return feats

        # speeds per segment
        speed = seg_len / (dt_seg + eps)
        speed = np.where(np.isfinite(speed), speed, np.nan)

        # headings per segment
        theta = np.arctan2(dy, dx)  # [-pi, pi]
        theta_u = np.unwrap(theta)

        # turn rate (per second)
        dtheta = np.diff(theta_u)
        dt_turn = dt_seg[1:] if dt_seg.size >= 2 else np.ones_like(dtheta)
        turn_rate = np.abs(dtheta) / (dt_turn + eps)
        turn_rate = np.where(np.isfinite(turn_rate), turn_rate, np.nan)

        # ---- requested features ----

        # 1) vertical_oscillation_freq (from Z or NaN)
        if has_z:
            # use vertical velocity series
            dz = np.diff(z)
            vz = dz / (dt_seg + eps)
            feats["vertical_oscillation_freq"] = _dominant_freq(vz, dt_mean)

        # 2) speed_acf5
        feats["speed_acf5"] = _acf_lag(speed, lag=5)

        # 3) heading_acf10 (use cos(theta) to respect circularity)
        feats["heading_acf10"] = _acf_lag(np.cos(theta), lag=10)

        # 4) fractal_dimension (Katz)
        feats["fractal_dimension"] = _katz_fd(x, y, z=z if has_z else None)

        # 5) maneuver_intensity = std(turn_rate) * std(speed)
        s_std = float(np.nanstd(speed)) if np.isfinite(np.nanstd(speed)) else np.nan
        tr_std = float(np.nanstd(turn_rate)) if np.isfinite(np.nanstd(turn_rate)) else np.nan
        if np.isfinite(s_std) and np.isfinite(tr_std):
            feats["maneuver_intensity"] = float(tr_std * s_std)

        # 6) control_smoothness = 1 / jerk_xy_std
        # velocities in XY
        vx = dx / (dt_seg + eps)
        vy = dy / (dt_seg + eps)
        # acceleration (align lengths)
        if vx.size >= 3:
            ax = np.diff(vx) / (dt_seg[1:] + eps)
            ay = np.diff(vy) / (dt_seg[1:] + eps)
            # jerk (align)
            if ax.size >= 3:
                jx = np.diff(ax) / (dt_seg[2:] + eps)
                jy = np.diff(ay) / (dt_seg[2:] + eps)
                jerk = np.sqrt(jx * jx + jy * jy)
                jerk_std = float(np.nanstd(jerk)) if np.isfinite(np.nanstd(jerk)) else np.nan
                feats["control_smoothness"] = float(1.0 / (jerk_std + eps)) if np.isfinite(jerk_std) else np.nan

        # 7) state_transition_entropy (4-state from vz + turn_rate)
        # states per segment index:
        # 0 climb, 1 descent, 2 cruise, 3 maneuver
        if has_z:
            dz = np.diff(z)
            vz = dz / (dt_seg + eps)
            vz_abs = np.abs(vz)

            vz_thr = float(np.nanpercentile(vz_abs, 60)) if np.isfinite(np.nanpercentile(vz_abs, 60)) else np.nan
            tr_thr = float(np.nanpercentile(turn_rate, 70)) if np.isfinite(np.nanpercentile(turn_rate, 70)) else np.nan

            if np.isfinite(vz_thr) and np.isfinite(tr_thr):
                states = np.full(vz.size, 2, dtype=int)  # default cruise

                # climb / descent
                states[vz > vz_thr] = 0
                states[vz < -vz_thr] = 1

                # maneuver override (need align lengths: turn_rate is len-2)
                # map maneuver decision to segment indices [1..end-1]
                maneuver_mask = np.zeros_like(states, dtype=bool)
                if turn_rate.size > 0:
                    m = turn_rate > tr_thr
                    maneuver_mask[1:1 + m.size] = m
                states[maneuver_mask] = 3

                # transitions
                if states.size >= 4:
                    a = states[:-1]
                    b = states[1:]
                    trans = a * 4 + b  # 0..15
                    hist = np.bincount(trans, minlength=16).astype(float)
                    total = float(hist.sum())
                    if total > 0:
                        p = hist / (total + eps)
                        H = -np.sum(p[p > 0] * np.log(p[p > 0] + eps))
                        feats["state_transition_entropy"] = float(H / np.log(16))  # normalize 0..1

        # 8) glide_ratio (fraction of time with low vertical acceleration)
        if has_z:
            dz = np.diff(z)
            vz = dz / (dt_seg + eps)
            if vz.size >= 3:
                az = np.diff(vz) / (dt_seg[1:] + eps)
                az_abs = np.abs(az)
                # "glide" = low |az| and low turn_rate (stable)
                az_thr = float(np.nanpercentile(az_abs, 40)) if np.isfinite(np.nanpercentile(az_abs, 40)) else np.nan
                tr_thr2 = float(np.nanpercentile(turn_rate, 50)) if np.isfinite(np.nanpercentile(turn_rate, 50)) else np.nan
                if np.isfinite(az_thr) and np.isfinite(tr_thr2) and az_abs.size > 0:
                    # align: az len = n-2, turn_rate len = n-2
                    stable = (az_abs < az_thr) & (turn_rate[:az_abs.size] < tr_thr2)
                    feats["glide_ratio"] = float(np.nanmean(stable)) if stable.size else np.nan

        # 9) speed_peak_count (peaks above p90)
        sp = speed.copy()
        sp = sp[np.isfinite(sp)]
        if sp.size >= 9:
            thr = float(np.percentile(sp, 90))
            # use original speed array for peak detection (keep alignment)
            s = speed
            peaks = 0
            for i in range(1, s.size - 1):
                if np.isfinite(s[i-1]) and np.isfinite(s[i]) and np.isfinite(s[i+1]):
                    if s[i] > s[i-1] and s[i] > s[i+1] and s[i] >= thr:
                        peaks += 1
            feats["speed_peak_count"] = int(peaks)

        # 10) acf_decay_tau (speed ACF drops below 1/e)
        s = speed.copy()
        s = s[np.isfinite(s)]
        if s.size >= 20:
            # compute acf up to max_lag
            max_lag = min(60, s.size // 2)
            # normalized autocorr
            s0 = s - s.mean()
            denom = np.dot(s0, s0) + eps
            acfs = []
            for lag in range(1, max_lag + 1):
                acfs.append(float(np.dot(s0[lag:], s0[:-lag]) / denom))
            acfs = np.asarray(acfs, float)

            target = 1.0 / np.e
            below = np.where(acfs < target)[0]
            if below.size:
                lag_idx = int(below[0] + 1)  # because lags start at 1
                feats["acf_decay_tau"] = float(lag_idx * dt_mean)

        # 11) trajectory_planarity (3D PCA)
        if has_z:
            feats["trajectory_planarity"] = _planarity_pca(x, y, z)

        # 12) energy_per_distance (proxy: mean(v^2 + vz^2)/path_length)
        path_length = float(np.nansum(seg_len))
        if np.isfinite(path_length) and path_length > eps:
            if has_z:
                dz = np.diff(z)
                vz = dz / (dt_seg + eps)
                # align speed and vz lengths (both n-1)
                v2 = (speed * speed) + (vz * vz)
            else:
                v2 = (speed * speed)
            e_mean = float(np.nanmean(v2)) if np.isfinite(np.nanmean(v2)) else np.nan
            if np.isfinite(e_mean):
                feats["energy_per_distance"] = float(e_mean / (path_length + eps))

        return feats

    except Exception:
        return feats


# ---------- usage example ----------
train = pd.read_parquet("../../data/interim/train/train_10.parquet")

# if you have trajectory_time column with same length as coords:
# feat_extra = train.apply(lambda r: extract_species_boost_features(r["trajectory"], r["trajectory_time"]), axis=1)
# else:
feat_extra = train["trajectory"].apply(extract_species_boost_features)

feat_extra = feat_extra.apply(pd.Series).add_prefix("boost_")
train = pd.concat([train, feat_extra], axis=1)
train = train.loc[:, ~train.columns.duplicated(keep="first")]

train.to_parquet("../../data/interim/train/train_11.parquet", index=False)
