from pathlib import Path
"""
Para la comparación con y sin regularización L2. Estas configuraciones. Esto es del punto 4 de la práctica.
Sin regularización:
    L2_REG = 0.0
    USE_DROPOUT = False
Con regularización:
    L2_REG = 1e-3
    USE_DROPOUT = False
Con Dropout:
    L2_REG = 0.0
    USE_DROPOUT = True
L2 + Dropout:
    L2_REG = 1e-3
    USE_DROPOUT = True  
    DROPOUT_RATE = 0.3
"""
"""
Esto ya lo pide el punto 5 de la práctica.
Algunas configuraciones que ya probe, para tener referencias para el reporte, solo modifica los parametros que dice en cada uno:

1) Baseline (ya usada antes)
HIDDEN_LAYERS = [16]
L2_REG = 0.0
USE_DROPOUT = False
LEARNING_RATE = 1e-3

2) Más neuronas (capas más grandes)
HIDDEN_LAYERS = [32, 16]
L2_REG = 0.0
USE_DROPOUT = False
LEARNING_RATE = 1e-3

3) Menor tasa de aprendizaje
HIDDEN_LAYERS = [16]
L2_REG = 0.0
USE_DROPOUT = False
LEARNING_RATE = 1e-4

4) Mayor tasa de aprendizaje
HIDDEN_LAYERS = [16]
L2_REG = 0.0
USE_DROPOUT = False
LEARNING_RATE = 1e-2

5) Regularización L2
HIDDEN_LAYERS = [16]
L2_REG = 1e-3
USE_DROPOUT = False
LEARNING_RATE = 1e-3

6) Dropout + más neuronas
HIDDEN_LAYERS = [64, 32]
L2_REG = 0.0
USE_DROPOUT = True
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3
"""

DATA_PATH = Path("BP/Archivos/Data/HousingData.csv")


TEST_SIZE = 0.2
RANDOM_STATE = 42


SCALER = "standard"


HIDDEN_LAYERS = [16]


L2_REG = 0.0

USE_DROPOUT = False
DROPOUT_RATE = 0.3  


LEARNING_RATE = 1e-3


EPOCHS = 150
BATCH_SIZE = 32
VERBOSE = 1
