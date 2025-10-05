from data.preprocesamiento import load_clean_split_scale, plot_feature_vs_target
from entrenamiento.entre import train_model
from extras.visual import plot_history
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    # 0) Visualización inicial de relaciones entre variables y MEDV
    print("Mostrando gráficas de exploración de características vs MEDV...")
    plot_feature_vs_target()

    # 1) Preprocesamiento de datos
    X_train, X_test, y_train, y_test, feature_names = load_clean_split_scale()

    # 2) Entrenamiento del modelo
    out = train_model(X_train, y_train)
    model = out["model"]
    history = out["history"]

    # 3) Visualización de la curva de MSE por épocas
    plot_history(history, title_prefix="Boston Housing MLP")

    # 4) Evaluación en el conjunto de prueba
    y_pred = model.predict(X_test).ravel()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[TEST] MSE: {mse:.4f} | MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
