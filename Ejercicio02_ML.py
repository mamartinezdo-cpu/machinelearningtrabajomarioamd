# ============================================================
# EJERCICIO 2: Procesamiento del Dataset “Student Performance”
# ============================================================
# Objetivo: Procesar los datos para un modelo que prediga la nota final (G3)
# Etapas:
# 1. Carga del dataset y análisis de variables categóricas
# 2. Eliminación de duplicados y valores inconsistentes
# 3. Codificación One-Hot de variables categóricas
# 4. Normalización de variables numéricas
# 5. Separación de variables predictoras (X) y objetivo (y)
# 6. División en entrenamiento (80%) y prueba (20%)
# Reto adicional: Análisis de correlación entre G1, G2 y G3
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# Carga del dataset
# --------------------------
df = pd.read_csv("student-mat.csv", sep=";")

print("Primeras filas del dataset original:")
print(df.head())

print("\nInformación general del dataset:")
print(df.info())

print("\nCantidad de valores nulos por columna:")
print(df.isnull().sum())

# --------------------------
# Eliminación de duplicados y valores inconsistentes
# --------------------------
# Eliminamos duplicados
duplicados = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"\nDuplicados eliminados: {duplicados}")

# Verificamos valores inconsistentes (como pueden ser: edad fuera de rango)
print("\nValores fuera de rango en edad (age):")
print(df[(df['age'] < 10) | (df['age'] > 22)][['age']])

df = df[(df['age'] >= 10) & (df['age'] <= 22)]

# --------------------------
# Codificación One-Hot de variables categóricas
# --------------------------
# Identificar variables categóricas
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("\nVariables categóricas encontradas:")
print(cat_cols)

# Aplicamos One Hot Encoding (creará columnas binarias por categoría)
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("\nDatos después de la codificación:")
print(df.head())

# --------------------------
# Normalización de variables numéricas
# --------------------------
scaler = StandardScaler()
cols_to_scale = ['age', 'absences', 'G1', 'G2']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("\nDatos después de la normalización:")
print(df[cols_to_scale].head())

# --------------------------
# Separación de variables predictoras (X) y objetivo (y)
# --------------------------
X = df.drop('G3', axis=1)
y = df['G3']

# --------------------------
# División en entrenamiento (80%) y prueba (20%)
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nDimensiones del conjunto de entrenamiento y prueba:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# --------------------------
# Reto adicional: Correlación entre G1, G2 y G3
# --------------------------
corr = df[['G1', 'G2', 'G3']].corr()
print("\nMatriz de correlación entre G1, G2 y G3:")
print(corr)

# --------------------------
# Mostrar primeras filas procesadas
# --------------------------
print("\nPrimeras 5 filas del conjunto de entrenamiento procesado:")
print(X_train.head())
