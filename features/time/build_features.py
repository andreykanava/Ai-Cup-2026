import pandas as pd
import numpy as np

def extract_time_features(timestamp_start, timestamp_end):
    feats = {
        "duration_sec": np.nan,
        "hour": np.nan,
        "day_of_year": np.nan,
        "is_morning": np.nan,
        "is_evening": np.nan,
        "is_night": np.nan,
    }

    try:
        start = pd.to_datetime(timestamp_start)
        end   = pd.to_datetime(timestamp_end)

        # duration in seconds
        duration = (end - start).total_seconds()
        feats["duration_sec"] = duration

        # hour
        hour = start.hour
        feats["hour"] = hour

        # day of year
        feats["day_of_year"] = start.dayofyear

        # time-of-day flags
        feats["is_morning"] = int(5 <= hour <= 10)
        feats["is_evening"] = int(17 <= hour <= 22)
        feats["is_night"]   = int(hour <= 4 or hour >= 23)

        return feats

    except Exception:
        return feats


train = pd.read_parquet("../../data/interim/train/train_1.parquet")

time_df = train.apply(
    lambda row: extract_time_features(
        row["timestamp_start_radar_utc"],
        row["timestamp_end_radar_utc"]
    ),
    axis=1
).apply(pd.Series)

print(time_df.shape)
print(time_df.head())
print(time_df.isna().mean().sort_values(ascending=False).head(10))


train = pd.concat([train, time_df], axis=1)

train.to_parquet("../../data/interim/train/train_2.parquet", index=False)
