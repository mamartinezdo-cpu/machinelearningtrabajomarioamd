# ============================================================
# EJERCICIO 1: Procesamiento del Dataset “Titanic”
# ============================================================
# Objetivo: Preparar los datos para un modelo que prediga la supervivencia de los pasajeros.
# Etapas:
# 1. Carga del dataset
# 2. Exploración inicial (info, describe, nulls, tipos de datos)
# 3. Limpieza de datos (valores nulos, duplicados, outliers)
# 4. Codificación de variables categóricas
# 5. Normalización o estandarización
# 6. División en conjuntos de entrenamiento y prueba
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --------------------------
# 2️⃣ Carga del dataset
# --------------------------
df = pd.read_csv("titanic.csv")

print("Primeras filas del dataset original:")
print(df.head())

# --------------------------
# Exploración inicial
# --------------------------
print("\nInformación general del dataset:")
print(df.info())

print("\nResumen estadístico de variables numéricas:")
print(df.describe())

print("\nCantidad de valores nulos por columna:")
print(df.isnull().sum())

# --------------------------
# Limpieza de datos
# --------------------------
# Eliminar columnas irrelevantes
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Rellenamos los valores nulos con media o moda según el tipo de dato
if df['Age'].isnull().sum() > 0:
    df['Age'] = df['Age'].fillna(df['Age'].mean())

if df['Embarked'].isnull().sum() > 0:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Eliminamos duplicados
df.drop_duplicates(inplace=True)

print("\nDataset después de la limpieza:")
print(df.head())

# --------------------------
# Codificación de variables categóricas
# --------------------------
# Codificar variable 'Sex'
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Codificar variable 'Embarked' (usando One-Hot Encoding)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nDatos después de la codificación:")
print(df.head())

# --------------------------
# Normalización / Estandarización
# --------------------------
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print("\nDatos después de la estandarización:")
print(df[['Age', 'Fare']].head())

# --------------------------
# División en entrenamiento y prueba
# --------------------------
# Variable objetivo
y = df['Survived']

# Variables predictoras
X = df.drop('Survived', axis=1)

# División de los datos: 70% entrenamiento y 30% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Mostrar dimensiones resultantes
print("\nDimensiones del conjunto de entrenamiento y prueba:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# --------------------------
# Mostrar primeros registros procesados
# --------------------------
print("\nPrimeras 5 filas del conjunto de entrenamiento procesado:")
print(X_train.head())
