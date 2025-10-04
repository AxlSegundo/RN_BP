import matplotlib.pyplot as plt

def plot_history(history, title_prefix="MLP Boston"):
    h = history.history   
    plt.figure()
    plt.plot(h["mse"], label="train MSE")
    plt.plot(h.get("val_mse", []), label="val MSE")
    plt.title(f"{title_prefix} - MSE por época")
    plt.xlabel("Épocas")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
