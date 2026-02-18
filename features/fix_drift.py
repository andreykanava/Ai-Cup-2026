# fix_drift_select_stable.py
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
import os

# ===== НАСТРОЙКИ =====
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../data/processed_stable"
TRAIN_FILE = "X_train.parquet"
TEST_FILE = "X_test.parquet"
RANDOM_STATE = 42
P_THRESHOLD = 0.05  # порог p-value для стабильности

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== ЗАГРУЗКА =====
print("Загрузка данных...")
X_train = pd.read_parquet(os.path.join(DATA_DIR, TRAIN_FILE))
X_test = pd.read_parquet(os.path.join(DATA_DIR, TEST_FILE))
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# Разделяем на числовые и категориальные
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Числовых признаков: {len(num_cols)}")
print(f"Категориальных признаков: {len(cat_cols)}")

# ===== ОБРАБОТКА КАТЕГОРИАЛЬНЫХ =====
X_train_fixed = X_train.copy()
X_test_fixed = X_test.copy()

for col in cat_cols:
    # Заполняем пропуски специальным значением
    X_train_fixed[col] = X_train_fixed[col].astype("string").fillna("__MISSING__")
    X_test_fixed[col] = X_test_fixed[col].astype("string").fillna("__MISSING__")

# ===== АНАЛИЗ СТАБИЛЬНОСТИ ЧИСЛОВЫХ ПРИЗНАКОВ =====
print("\nАнализ дрейфа числовых признаков...")
stable_cols = []
drift_cols = []

for col in num_cols:
    # Пропуски уже заполним позже, но для теста KS пропусков быть не должно
    # Временно удалим NaN для расчёта
    train_clean = X_train[col].dropna()
    test_clean = X_test[col].dropna()
    if len(train_clean) == 0 or len(test_clean) == 0:
        continue
    stat, p = ks_2samp(train_clean, test_clean)
    if p >= P_THRESHOLD:
        stable_cols.append(col)
    else:
        drift_cols.append(col)

print(f"Стабильных признаков (p >= {P_THRESHOLD}): {len(stable_cols)} из {len(num_cols)}")
print(f"Дрейфующих: {len(drift_cols)}")

# Если стабильных нет, выходим
if len(stable_cols) == 0:
    raise ValueError("Нет стабильных числовых признаков! Попробуйте увеличить порог или использовать другой метод.")

# ===== ЗАПОЛНЕНИЕ ПРОПУСКОВ (только для стабильных признаков) =====
print("\nЗаполнение пропусков медианой по train (только для стабильных признаков)...")
imputer = SimpleImputer(strategy='median')
X_train_stable_imp = imputer.fit_transform(X_train[stable_cols])
X_test_stable_imp = imputer.transform(X_test[stable_cols])

# Превращаем обратно в DataFrame
X_train_stable = pd.DataFrame(X_train_stable_imp, columns=stable_cols, index=X_train.index)
X_test_stable = pd.DataFrame(X_test_stable_imp, columns=stable_cols, index=X_test.index)

# ===== ПРИМЕНЕНИЕ QUANTILETRANSFORMER К СТАБИЛЬНЫМ ПРИЗНАКАМ =====
print("Применение QuantileTransformer к стабильным признакам...")
qt = QuantileTransformer(output_distribution='uniform', random_state=RANDOM_STATE)
X_train_stable_qt = qt.fit_transform(X_train_stable)
X_test_stable_qt = qt.transform(X_test_stable)

X_train_stable = pd.DataFrame(X_train_stable_qt, columns=stable_cols, index=X_train.index)
X_test_stable = pd.DataFrame(X_test_stable_qt, columns=stable_cols, index=X_test.index)

# ===== ОБЪЕДИНЯЕМ С КАТЕГОРИАЛЬНЫМИ =====
X_train_final = pd.concat([X_train_stable, X_train_fixed[cat_cols]], axis=1)
X_test_final = pd.concat([X_test_stable, X_test_fixed[cat_cols]], axis=1)

print(f"Итоговая размерность: X_train_final {X_train_final.shape}")

# ===== СОХРАНЕНИЕ =====
print(f"\nСохранение в {OUTPUT_DIR}...")
X_train_final.to_parquet(os.path.join(OUTPUT_DIR, "X_train_stable.parquet"))
X_test_final.to_parquet(os.path.join(OUTPUT_DIR, "X_test_stable.parquet"))
print("Готово!")