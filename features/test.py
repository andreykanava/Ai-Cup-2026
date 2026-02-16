import binascii
from shapely import wkb
import pandas as pd
import numpy as np

train = pd.read_parquet("../data/interim/train/train_10.parquet")
test = pd.read_parquet("../data/interim/test/test_10.parquet")

for col in train.columns:
    print(col)