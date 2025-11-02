# ============================================================
# STREAMLIT APP PRINCIPAL - PROCESAMIENTO DE DATASETS EN ML
# ============================================================
# Autor: Mario Antonio Martínez Domínguez
# Descripción:
#   Pantalla principal del proyecto. Permite seleccionar entre los tres ejercicios:
#   1. Titanic
#   2. Student Performance
#   3. Iris
# ============================================================

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Procesamiento de Datasets en ML", layout="wide")
st.title("Procesamiento de Datasets en Machine Learning")
st.markdown("""

Selecciona un ejercicio en el menú desplegable para visualizar su flujo completo:
- *Titanic*: Predicción de supervivencia  
- *Student Performance*: Predicción de nota final  
- *Iris*: Clasificación de flores
""")

# ============================================================
# SELECCIÓN DEL EJERCICIO
# ============================================================
opcion = st.selectbox(
    "Selecciona un ejercicio para visualizar:",
    ("-- Selecciona una opción --", "Ejercicio 1: Titanic", "Ejercicio 2: Student Performance", "Ejercicio 3: Iris")
)

# ============================================================
# EJERCICIO TITANIC
# ============================================================
if opcion == "Ejercicio 1: Titanic":
    st.header("Ejercicio 1: Titanic – Predicción de Supervivencia")

    uploaded_file = st.file_uploader("Sube el archivo titanic.csv", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("**Primeras filas del dataset:**")
        st.dataframe(df.head())

        # Limpieza
        df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df.drop_duplicates(inplace=True)

        # Codificación
        le = LabelEncoder()
        df['Sex'] = le.fit_transform(df['Sex'])
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

        # Estandarización
        scaler = StandardScaler()
        df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

        # División
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.success("Procesamiento completado.")
        st.write(f"**X_train:** {X_train.shape} | **X_test:** {X_test.shape}")
        st.dataframe(X_train.head())
    else:
        st.info("Sube el archivo `titanic.csv` para comenzar.")

# ============================================================
# EJERCICIO STUDENT PERFORMANCE
# ============================================================
elif opcion == "Ejercicio 2: Student Performance":
    st.header("Ejercicio 2: Student Performance – Predicción de Nota Final (G3)")

    uploaded_file = st.file_uploader("Sube el archivo student-mat.csv", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=";")
        st.write("**Primeras filas del dataset:**")
        st.dataframe(df.head())

        # Limpieza de duplicados
        df.drop_duplicates(inplace=True)

        # Codificación One-Hot
        df = pd.get_dummies(df, drop_first=True)

        # Normalización de variables numéricas
        scaler = StandardScaler()
        num_cols = ['age', 'absences', 'G1', 'G2']
        df[num_cols] = scaler.fit_transform(df[num_cols])

        # Separación de variables
        X = df.drop('G3', axis=1)
        y = df['G3']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.success("Procesamiento completado.")
        st.write(f"**X_train:** {X_train.shape} | **X_test:** {X_test.shape}")
        st.dataframe(X_train.head())

        # Correlación
        st.subheader("Correlación entre notas G1, G2 y G3")
        corr = df[['G1', 'G2', 'G3']].corr()
        st.dataframe(corr)
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Sube el archivo `student-mat.csv` para comenzar.")

# ============================================================
# EJERCICIO IRIS
# ============================================================
elif opcion == "Ejercicio 3: Iris":
    st.header("Ejercicio 3: Iris – Clasificación de Especies")

    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    st.write("**Primeras filas del dataset original:**")
    st.dataframe(df.head())

    # Estandarización
    scaler = StandardScaler()
    df[data.feature_names] = scaler.fit_transform(df[data.feature_names])

    # División
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.success("Procesamiento completado.")
    st.write(f"**X_train:** {X_train.shape} | **X_test:** {X_test.shape}")

    st.subheader("Estadísticas descriptivas del dataset estandarizado:")
    st.dataframe(df.describe())

    # Gráfico
    st.subheader("Gráfico de dispersión (Sepal Length vs Petal Length)")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df[data.feature_names[0]],
        y=df[data.feature_names[2]],
        hue=df['target'],
        palette='Set1'
    )
    st.pyplot(fig)
