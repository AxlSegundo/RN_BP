import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple
from config import DATA_PATH, TEST_SIZE, RANDOM_STATE, SCALER

def load_clean_split_scale() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    boston = pd.read_csv(DATA_PATH)
    boston = boston.dropna()

    X = boston.drop("MEDV", axis=1)
    y = boston["MEDV"].values.astype("float32")

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    if SCALER.lower() == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    return X_train, X_test, y_train, y_test, feature_names
