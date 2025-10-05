import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple
from config import DATA_PATH, TEST_SIZE, RANDOM_STATE, SCALER

def load_clean_split_scale() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    # cargar dataset desde csv
    boston = pd.read_csv(DATA_PATH)
    boston = boston.dropna()

    X = boston.drop("MEDV", axis=1)
    y = boston["MEDV"].values.astype("float32")

    feature_names = list(X.columns)

    # dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # normalización
    if SCALER.lower() == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return X_train, X_test, y_train, y_test, feature_names


def plot_feature_vs_target():

    boston = pd.read_csv(DATA_PATH)
    boston = boston.dropna()

    # RM = número de habitaciones
    plt.figure()
    plt.scatter(boston["RM"], boston["MEDV"], alpha=0.6)
    plt.xlabel("Número de habitaciones (RM)")
    plt.ylabel("Precio medio (MEDV)")
    plt.title("Relación RM vs MEDV")
    plt.show()

    # LSTAT = % población de bajo estatus socioeconómico
    plt.figure()
    plt.scatter(boston["LSTAT"], boston["MEDV"], alpha=0.6, color="orange")
    plt.xlabel("% bajo estatus (LSTAT)")
    plt.ylabel("Precio medio (MEDV)")
    plt.title("Relación LSTAT vs MEDV")
    plt.show()

    # PTRATIO = relación alumno/profesor
    plt.figure()
    plt.scatter(boston["PTRATIO"], boston["MEDV"], alpha=0.6, color="green")
    plt.xlabel("Relación alumno/profesor (PTRATIO)")
    plt.ylabel("Precio medio (MEDV)")
    plt.title("Relación PTRATIO vs MEDV")
    plt.show()

    # CRIM = tasa de criminalidad
    plt.figure()
    plt.scatter(boston["CRIM"], boston["MEDV"], alpha=0.6, color="red")
    plt.xlabel("Tasa de criminalidad (CRIM)")
    plt.ylabel("Precio medio (MEDV)")
    plt.title("Relación CRIM vs MEDV")
    plt.show()

