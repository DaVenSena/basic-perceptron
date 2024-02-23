import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_desicion_region(X, y, classifier, resolution = 0.02):
    # Crear los generadores de marcadores y mapas de colores
    markers = ("o", "^")
    colors = ("green", "red")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Divide la superficie segun la desicion
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contour(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Ejemplo de parcela
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y  == cl, 1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = cl, edgecolors = "black")