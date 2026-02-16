import binascii
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis


def extract_wingbeat_frequency(echo_hex, sampling_rate=130.0, eps=1e-12):
    """
    Extract wing-beat frequency features from radar echo time series (intensity over time).
    Assumes echo_hex is hex-encoded bytes of float32 array (echo intensities).
    Uses FFT to find dominant frequency (wingbeat freq) as in radar ornithology papers.
    Returns dict with: fundamental_freq, freq_p50 (median), freq_skew, freq_kurt.
    Sampling rate default 130 Hz as common in literature (e.g., Bruderer et al.).
    """
    feats = {
        "wbf_fundamental_freq": np.nan,
        "wbf_freq_p50": np.nan,
        "wbf_freq_skew": np.nan,
        "wbf_freq_kurt": np.nan,
    }

    try:
        # Parse hex to np.array float32
        bytes_data = binascii.unhexlify(echo_hex)
        intensities = np.frombuffer(bytes_data, dtype=np.float32)
        intensities = intensities[np.isfinite(intensities)]
        n = intensities.size
        if n < 8:
            return feats

        # Preprocess: detrend and normalize
        intensities -= np.mean(intensities)
        intensities /= (np.std(intensities) + eps)

        # FFT for frequencies
        fft_mag = np.abs(rfft(intensities))
        freqs = rfftfreq(n, 1 / sampling_rate)
        fft_mag[0] = 0  # Remove DC

        # Find dominant frequency (fundamental wingbeat)
        dom_idx = np.argmax(fft_mag)
        feats["wbf_fundamental_freq"] = float(freqs[dom_idx])

        # Distribution of freq components (weighted by mag)
        p = fft_mag / (np.sum(fft_mag) + eps)
        weighted_freqs = freqs * p
        feats["wbf_freq_p50"] = float(np.median(weighted_freqs))

        # Skew and kurt of frequency distribution
        sk = skew(freqs, bias=False, aweights=p)
        ku = kurtosis(freqs, bias=False, aweights=p)
        feats["wbf_freq_skew"] = float(sk)
        feats["wbf_freq_kurt"] = float(ku)

        return feats

    except Exception:
        return feats


def extract_flapping_patterns(echo_hex, sampling_rate=130.0, eps=1e-12, peak_prominence=0.1):
    """
    Extract flapping pattern features: harmonics ratio, fluctuation depth, skewness, duty cycle.
    From same echo time series as above.
    Harmonics: ratio of higher harmonics power to fundamental.
    Depth: std of fluctuations.
    Skewness: skew of intensity series.
    Duty cycle: fraction of time above mean (flapping vs gliding proxy).
    """
    feats = {
        "flap_harmonics_ratio": np.nan,
        "flap_fluctuation_depth": np.nan,
        "flap_skewness": np.nan,
        "flap_duty_cycle": np.nan,
    }

    try:
        bytes_data = binascii.unhexlify(echo_hex)
        intensities = np.frombuffer(bytes_data, dtype=np.float32)
        intensities = intensities[np.isfinite(intensities)]
        n = intensities.size
        if n < 8:
            return feats

        # Preprocess
        mu = np.mean(intensities)
        intensities -= mu
        std = np.std(intensities) + eps
        intensities /= std

        # FFT for harmonics
        fft_mag = np.abs(rfft(intensities))
        freqs = rfftfreq(n, 1 / sampling_rate)
        fft_mag[0] = 0
        dom_idx = np.argmax(fft_mag)
        fund_freq = freqs[dom_idx]
        fund_power = fft_mag[dom_idx] ** 2

        # Harmonics power: sum power at ~2f,3f,... up to nyquist
        harm_idxs = np.where((freqs > 1.5 * fund_freq) & (np.mod(freqs, fund_freq) < 0.1 * fund_freq))[0]
        harm_power = np.sum(fft_mag[harm_idxs] ** 2)
        feats["flap_harmonics_ratio"] = float(harm_power / (fund_power + eps))

        # Fluctuation depth: std of detrended series (normalized already)
        feats["flap_fluctuation_depth"] = float(std)

        # Skewness of raw intensities
        feats["flap_skewness"] = float(skew(intensities, bias=False))

        # Duty cycle: fraction where intensity > mean (proxy for flapping time)
        above_mean = intensities > 0
        peaks, _ = find_peaks(intensities, prominence=peak_prominence)
        if peaks.size > 1:
            cycle_lens = np.diff(peaks)
            flap_durs = []  # simplistic: assume peaks are flaps
            for i in range(len(peaks) - 1):
                seg = intensities[peaks[i]:peaks[i+1]]
                flap_durs.append(np.sum(seg > 0) / len(seg))
            feats["flap_duty_cycle"] = float(np.mean(flap_durs)) if flap_durs else np.nan
        else:
            feats["flap_duty_cycle"] = float(np.mean(above_mean))

        return feats

    except Exception:
        return feats


def extract_polarimetric_features(pol_hex, eps=1e-12):
    """
    Extract polarimetric features: stats from ZDR, rhoHV, KDP arrays.
    Assumes pol_hex is hex-encoded bytes of concatenated float32 arrays: [ZDR, rhoHV, KDP] with first 4 bytes int32 for lengths (len_ZDR, len_rho, len_KDP).
    Returns dict with mean, std, skew, kurt, p10/p50/p90 for each.
    Useful for bird/insect discrimination as per literature (ZDR high for insects, rhoHV low for mixed).
    """
    feats = {
        # ZDR stats
        "zdr_mean": np.nan, "zdr_std": np.nan, "zdr_skew": np.nan, "zdr_kurt": np.nan,
        "zdr_p10": np.nan, "zdr_p50": np.nan, "zdr_p90": np.nan,
        # rhoHV stats
        "rhohv_mean": np.nan, "rhohv_std": np.nan, "rhohv_skew": np.nan, "rhohv_kurt": np.nan,
        "rhohv_p10": np.nan, "rhohv_p50": np.nan, "rhohv_p90": np.nan,
        # KDP stats
        "kdp_mean": np.nan, "kdp_std": np.nan, "kdp_skew": np.nan, "kdp_kurt": np.nan,
        "kdp_p10": np.nan, "kdp_p50": np.nan, "kdp_p90": np.nan,
    }

    def _stats(arr):
        if arr.size < 2:
            return {k: np.nan for k in ["mean", "std", "skew", "kurt", "p10", "p50", "p90"]}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "skew": float(skew(arr, bias=False)),
            "kurt": float(kurtosis(arr, bias=False)),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
        }

    try:
        bytes_data = binascii.unhexlify(pol_hex)
        # First 12 bytes: 3 int32 for lengths
        len_zdr, len_rhohv, len_kdp = np.frombuffer(bytes_data[:12], dtype=np.int32)
        offset = 12
        zdr = np.frombuffer(bytes_data[offset:offset + len_zdr*4], dtype=np.float32)
        offset += len_zdr*4
        rhohv = np.frombuffer(bytes_data[offset:offset + len_rhohv*4], dtype=np.float32)
        offset += len_rhohv*4
        kdp = np.frombuffer(bytes_data[offset:offset + len_kdp*4], dtype=np.float32)

        zdr = zdr[np.isfinite(zdr)]
        rhohv = rhohv[np.isfinite(rhohv)]
        kdp = kdp[np.isfinite(kdp)]

        zdr_stats = _stats(zdr)
        rhohv_stats = _stats(rhohv)
        kdp_stats = _stats(kdp)

        for var in ["zdr", "rhohv", "kdp"]:
            stats = locals()[f"{var}_stats"]
            for k, v in stats.items():
                feats[f"{var}_{k}"] = v

        return feats

    except Exception:
        return feats


# Note: This assumes your dataframe has columns like "echo_signature" (hex for intensities)
# and "polarimetric_data" (hex for pol vars). If not, you'll need to add them or adjust.
# For demo, we'll skip apply if columns don't exist, but in real use, preprocess data accordingly.

train = pd.read_parquet("../../data/interim/train/train_9.parquet")  # From previous

# Apply wingbeat (assume "echo_signature" column exists)
if "echo_signature" in train.columns:
    wbf_df = train["echo_signature"].apply(extract_wingbeat_frequency).apply(pd.Series)
    flap_df = train["echo_signature"].apply(extract_flapping_patterns).apply(pd.Series)
    train = pd.concat([train, wbf_df, flap_df], axis=1)

# Apply polarimetric (assume "polarimetric_data" column)
if "polarimetric_data" in train.columns:
    pol_df = train["polarimetric_data"].apply(extract_polarimetric_features).apply(pd.Series)
    train = pd.concat([train, pol_df], axis=1)

# Drop duplicates
train = train.loc[:, ~train.columns.duplicated(keep="first")]

train.to_parquet("../../data/interim/train/train_10.parquet", index=False)
