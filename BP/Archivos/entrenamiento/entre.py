from typing import Dict, Any
import numpy as np
from config import EPOCHS, BATCH_SIZE, VERBOSE
from modelos.mlp import build_mlp

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Entrena el MLP y devuelve dict con model y history.
    """
    model = build_mlp(input_dim=X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE,
        validation_split=0.1,  # para monitorear MSE en validación durante el entrenamiento
        shuffle=True,
    )
    return {"model": model, "history": history}
