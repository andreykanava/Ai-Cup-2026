from sklearn.metrics import log_loss
import numpy as np
import pandas as pd

DATA_DIR = "../data/processed"
def permutation_importance_logloss(model, X_valid, y_valid, n_repeats=3, seed=42):
    rng = np.random.default_rng(seed)

    base_pred = model.predict_proba(X_valid)
    base = log_loss(y_valid, base_pred)

    scores = {}
    Xv = X_valid.copy()

    for col in Xv.columns:
        deltas = []
        orig = Xv[col].values.copy()

        for _ in range(n_repeats):
            shuffled = orig.copy()
            rng.shuffle(shuffled)
            Xv[col] = shuffled

            pred = model.predict_proba(Xv)
            deltas.append(log_loss(y_valid, pred) - base)

        Xv[col] = orig
        scores[col] = float(np.mean(deltas))

    return pd.Series(scores).sort_values(ascending=False)


X_train = pd.read_parquet(f"{DATA_DIR}/X_train.parquet")
y_train_df = pd.read_parquet(f"{DATA_DIR}/y_train_bird_group.parquet")
X_test = pd.read_parquet(f"{DATA_DIR}/X_test.parquet")
test_ids = pd.read_parquet(f"{DATA_DIR}/test_ids.parquet")

model.fit(X_train, y_train)
perm = permutation_importance_logloss(model, X_valid, y_valid, n_repeats=3)

keep = perm[perm > 0].index   # оставляем только то, что реально помогает
X_train2 = X_train[keep]
X_valid2 = X_valid[keep]
