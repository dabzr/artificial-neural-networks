import sklearn as sk
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sign_y(yi):
    return 2*yi-1

def loss(w, dataset):
    acc = 0
    X, Y = dataset
    for xi, yi in zip(X, Y):
        wx = w @ xi
        yi = sign_y(yi)
        acc += yi*sigmoid(-yi*wx)*xi
    return acc

ds = sk.datasets.make_classification(n_samples=1000, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_repeated=0)
X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(ds[0], ds[1], test_size=0.2, random_state=42)
ds_train = [X_train, Y_train]
ds_test = [X_test, Y_test]
def descendent_gradient(dataset, T):
    w = np.zeros(len(dataset[0][0]))
    error = 0.0001
    for _ in range(T):
        w = w - (error * loss(w, dataset))
    return w

weights = descendent_gradient(ds_train, 100)

def test(test_dataset, w):
    X, Y = test_dataset
    y_model = []
    for xi in X:
        wx = w @ xi
        y = sigmoid(wx)
        y = 1 if y > 0.5 else 0
        y_model.append(y)
    return y_model, Y

y_model, y_expected = test(ds_test, weights)
def calc_accuracy(model, expected):
    acc = 0
    count = 0
    for m, e in zip(model, expected):
        acc += 1 if m == e else 0
        count += 1
    return 1-(acc/count)

print(calc_accuracy(y_model, y_expected))
