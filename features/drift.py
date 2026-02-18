from scipy.stats import ks_2samp
import pandas as pd

def detect_drift_features(X_train, X_test, threshold=0.05):
    drift_features = []
    for col in X_train.columns:
        stat, p = ks_2samp(X_train[col].dropna(), X_test[col].dropna())
        if p < threshold:
            drift_features.append(col)
    return drift_features

X_train = pd.read_parquet("../data/processed/X_train.parquet")
X_test = pd.read_parquet("../data/processed/X_test.parquet")

drift_cols = detect_drift_features(X_train, X_test)
print(f"Признаков с дрейфом: {len(drift_cols)}")
# Посмотреть топ самых дрейфующих (по статистике KS)