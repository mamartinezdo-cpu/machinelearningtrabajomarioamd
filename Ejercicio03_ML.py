# ============================================================
# EJERCICIO 3: PROCESAMIENTO DEL DATASET IRIS
# Objetivo: Implementar un flujo completo de preprocesamiento
# y visualizar resultados.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Cargamos el dataset desde sklearn.datasets
# ------------------------------------------------------------
iris = load_iris()

# ------------------------------------------------------------
# Convertimos a DataFrame y agregamos nombres de columnas
# ------------------------------------------------------------
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

print("Primeras filas del dataset original:")
print(df.head(), "\n")

# ------------------------------------------------------------
# Exploración inicial
# ------------------------------------------------------------
print("Información general del dataset:")
print(df.info(), "\n")

print("Estadísticas descriptivas:")
print(df.describe(), "\n")

# ------------------------------------------------------------
# Limpieza de datos (aquí verificamos si son nulos o inconsistentes)
# ------------------------------------------------------------
print("Valores nulos por columna:")
print(df.isnull().sum(), "\n")

# No se detectan valores nulos en el dataset Iris original

# ------------------------------------------------------------
# Estandarización de las variables numéricas
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.iloc[:, :-1])  # no incluimos 'target'

df_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)
df_scaled["target"] = df["target"]

print("Dataset estandarizado (primeras 5 filas):")
print(df_scaled.head(), "\n")

# ------------------------------------------------------------
# División en entrenamiento (70%) y prueba (30%)
# ------------------------------------------------------------
X = df_scaled.drop("target", axis=1)
y = df_scaled["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Dimensiones del conjunto de entrenamiento: {X_train.shape}")
print(f"Dimensiones del conjunto de prueba: {X_test.shape}\n")

# ------------------------------------------------------------
# Gráfico de dispersión: sepal length vs petal length
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
for target, color, label in zip([0, 1, 2], ['red', 'green', 'blue'], iris.target_names):
    subset = df_scaled[df_scaled["target"] == target]
    plt.scatter(subset.iloc[:, 0], subset.iloc[:, 2], color=color, label=label, alpha=0.7)

plt.title("Distribución de Iris: Sepal Length vs Petal Length (Estandarizado)")
plt.xlabel("Sepal Length (estandarizado)")
plt.ylabel("Petal Length (estandarizado)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Estadísticas descriptivas del dataset estandarizado
# ------------------------------------------------------------
print("Estadísticas descriptivas del dataset estandarizado:")
print(df_scaled.describe())

#Las estadísticas descriptivas se muestran al cerrar el gráfico que se generó.