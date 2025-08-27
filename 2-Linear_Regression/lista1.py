from slregression import SimpleRegressor
import numpy as np
import random 
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from mvregression import MultipleRegressor
from sklearn.model_selection import train_test_split  # <-- import

def f(x):
    e = np.random.normal(0, 2)
    return 3*x+5+e

def mse(dataset, regressor):
    xl, yl = list(zip(*dataset))
    regressed = list(map(regressor.predict, xl))
    return sum(map(lambda x: (x[0]-x[1])**2, zip(yl, regressed)))/len(dataset)

# Questão 1
dataset = list(map(lambda x: (x, f(x)), np.random.uniform(low=-10, high=10, size=100)))
training_dataset = random.sample(dataset, k=80)
test_dataset = list(filter(lambda x: x not in training_dataset, dataset))
reg = SimpleRegressor()
reg.model(dataset)
error = mse(test_dataset, reg)
print(f"MSE: {error}")
print(f"w0: {reg.w0}")
print(f"w1: {reg.w1}")
x_pts, y_pts = list(zip(*dataset))
x = np.linspace(-10, 10, 400)
y = reg.predict(x)
plt.figure(figsize=(6,4))
plt.plot(x, y, color="red")
plt.scatter(x_pts, y_pts, s=40, label='dados')
plt.title("Função e pontos")
plt.legend()
plt.savefig("grafico.png")

# Questão 2
def mse2(X, Y, regressor):
    regressed = list(map(regressor.predict, X))
    return sum(map(lambda x: (x[0]-x[1])**2, zip(Y, regressed)))/len(dataset)

wine_quality = fetch_ucirepo(id=186) 

X = wine_quality.data.features 
X.insert(0, 'ones', 1)
X = X.to_numpy()
Y = wine_quality.data.targets.to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
mreg = MultipleRegressor()
mreg.model(X_train, Y_train)
print(mreg.w)
print(mse2(X_test, Y_test, mreg))

