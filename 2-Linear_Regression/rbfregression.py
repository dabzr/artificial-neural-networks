import numpy as np
import random

def f(x):
    return x**2

def phi(x, C, gamma=1):
    return np.exp(gamma * -np.linalg.norm(x - C))

def get_C(X):
    return [X[random.randint(0, len(X)-1)] for _ in range(10)]

def model(X, Y):
    return np.linalg.pinv(X) @ Y

def mse(actual_values, predicted_values):
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    differences = np.subtract(actual_values, predicted_values)
    squared_differences = np.square(differences)
    return np.mean(squared_differences)

def get_PHI(X, C):
    return np.array([[phi(x, c) for c in C] for x in X])
dataset = [(np.array([x]), f(x)) for x in np.random.uniform(low=-5, high=5, size=100)]

training_dataset = random.sample(dataset, k=80)
test_dataset = [d for d in dataset if d not in training_dataset]

X, Y = list(zip(*training_dataset))
X, Y = np.array(X), np.array(Y)

C = get_C(X)

PHI = get_PHI(X, C)
w = model(PHI, Y)

X_test, Y_test = list(zip(*test_dataset))
X_test, Y_test = np.array(X_test), np.array(Y_test)
model_Y = get_PHI(X_test, C) @ w
print(Y_test)
print(model_Y)
print(mse(Y_test, model_Y))
