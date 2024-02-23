import numpy as np


class Perceptron(object):
    def __init__(self, eta = 0.5, n_iter = 50, random_state = 1):
        self.eta = eta # Taza de aprendizaje (0.0 y 1.0)
        self.n_iter = n_iter # Numero de veces que va a pasar el conjunto de datos
        self.random_state = random_state # Semilla del generador de numero aleatorios
        
    def fit(self, X, y):
        '''Ajuste de los datos
        X: Vector de datos entrenamiento
            n_features: Numero de caracteristicas
        y: Vector de etiquetas de respuesta
        self: object
        '''
        
        regen = np.random.RandomState(self.random_state) # Se generan numeros aleatorios
        # self.w_ = regen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1]) # Numeros aleatorios con dev est
        self.w_ = [0, -0.5, 2]
        print("Pesos iniciales: ", self.w_)
        self.errors = [] # Vector vacio para almacenar los errores
        
        for i in range(self.n_iter): # Ciclo de las iteraciones
            errors = 0
            for xi, tag in zip(X, y): # Ciclo segun el numero de muestras
                update = self.eta * (tag - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors.append(errors)
            print("Pesos en epoch: ", self.w_)
        return self
    
    def net_income(self, X):
        'Calculo de la entrada neta'
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        'Etiqueta de clase de retorno despues del paso unitario'
        return np.where(self.net_income(X) >= 0.0, 1, 0)