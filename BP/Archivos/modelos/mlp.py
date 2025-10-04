from typing import List
import tensorflow as tf
from tensorflow.keras import layers, regularizers, optimizers, Model
from config import HIDDEN_LAYERS, L2_REG, LEARNING_RATE, USE_DROPOUT, DROPOUT_RATE

def build_mlp(input_dim: int) -> Model:
    l2 = regularizers.l2(L2_REG) if (L2_REG and L2_REG > 0) else None

    inputs = layers.Input(shape=(input_dim,))
    x = inputs

    for i, units in enumerate(HIDDEN_LAYERS):
        x = layers.Dense(units, activation="relu", kernel_regularizer=l2)(x)
        if USE_DROPOUT and DROPOUT_RATE and DROPOUT_RATE > 0:
            x = layers.Dropout(DROPOUT_RATE, name=f"dropout_{i}")(x)

    outputs = layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs, outputs, name="boston_mlp")

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss="mse", metrics=["mse", "mae"])
    return model
