import numpy as np
class MultipleRegressor:
    def __init__(self):
        pass

    def model(self, X, Y):
       pinvX = np.linalg.inv(X.T @ X) @ X.T
       self.w = pinvX @ Y 

    def predict(self, X):
       return sum(map(lambda e: e[0]*e[1], zip(self.w, X)))

