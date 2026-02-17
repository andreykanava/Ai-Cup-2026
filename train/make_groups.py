# train/make_groups_from_ids.py
from __future__ import annotations
import numpy as np
import pandas as pd

DATA_DIR = "../data/processed"
ID_COL = "track_id"

def main():
    ids = pd.read_parquet(f"{DATA_DIR}/train_ids.parquet")
    if ID_COL not in ids.columns:
        raise SystemExit(f"{ID_COL} not found in {DATA_DIR}/train_ids.parquet. Columns: {ids.columns.tolist()}")

    groups = ids[ID_COL].to_numpy()
    np.save("groups.npy", groups)

    uniq, cnt = np.unique(groups, return_counts=True)
    print("Group column:", ID_COL)
    print("n_rows:", len(groups))
    print("n_groups:", len(uniq))
    print("max group size:", int(cnt.max()))
    print("mean group size:", float(cnt.mean()))
    print("Saved -> groups.npy")

if __name__ == "__main__":
    main()
