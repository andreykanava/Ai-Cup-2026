import binascii
from shapely import wkb
import pandas as pd
import numpy as np

train = pd.read_parquet("../data/processed/overfit/X_train.parquet")
test = pd.read_parquet("../data/processed/overfit/X_test.parquet")

for col in train.columns:
    print(col)