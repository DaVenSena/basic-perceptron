import matplotlib.pyplot as plt
import numpy as np

import clasifier as c
import perceptron as p

data = [[6, 9], [8, 14], [9, 7], [9, 11], [11, 8], [14, 2], [16, 5], [17, 10], [19, 3]]

X = np.array(data)

y = [0, 0, 0, 0, 0, 1, 1, 1, 1]

ppn = p.Perceptron(eta=0.5, n_iter=10)

ppn.fit(X, y)

c.plot_desicion_region(X=X, y=y, classifier=ppn)

plt.xlabel("X1")
plt.ylabel("X2")

plt.legend(loc = "upper left")

plt.show()


""" 

plt.plot(range(1, len(ppn.errors)+1), ppn.errors, marker="o")


plt.xlabel("Epochs") """

""" 
print(X)
print(y)

plt.scatter(X[0:3, 0], X[0:3, 1], color = "red", marker = "o", label = "Positivo")
plt.scatter(X[3, 0], X[3, 1], color = "green", marker = "x", label = "Negativo")

plt.xlabel("X1")
plt.ylabel("X2")

plt.legend(loc = "upper center")

 """
