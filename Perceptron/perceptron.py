import random

def dotproduct(x, w):
    sum = 0
    for i, j in zip(x, w):
        sum += i*j
    return sum

class Perceptron:
    def __init__(self, dataset, weights=None) -> None:
        self.dataset = dataset
        self.model = weights

    @staticmethod
    def update_weights(w, x, y):
        for i in range(len(w)):
            w[i] = w[i] + x[i]*y
        return w

    @staticmethod
    def get_x_aug(x, fixed_input=1):
        return [fixed_input] + x

    def train(self, dimension, epochs, bias):
        w = [bias] + [0]*dimension
        for _ in range(epochs):        
            i = random.randint(0, len(self.dataset)-1)
            x, y = self.dataset[i]
            x_aug = Perceptron.get_x_aug(x)
            xw = dotproduct(x_aug, w)
            if (xw >= 0 and y == -1) or (xw < 0 and y == 1):
                w = Perceptron.update_weights(w, x_aug, y)
        self.model = w
    
    def predict(self, x):
        x_aug = Perceptron.get_x_aug(x)
        if dotproduct(x_aug, self.model) >= 0:
            return 1
        return -1

def test(dataset):
    p = Perceptron(dataset)
    p.train(dimension=2, epochs=50, bias=0)
    print(f"[1, 1]={p.predict([1, 1])}")
    print(f"[1, 0]={p.predict([1, 0])}")
    print(f"[0, 1]={p.predict([0, 1])}")
    print(f"[0, 0]={p.predict([0, 0])}")

or_dataset = [([1, 1], 1), ([1, 0], 1), ([0, 0], -1), ([0, 1], 1)]
xor_dataset = [([1, 1], -1), ([1, 0], 1), ([0, 0], -1), ([0, 1], 1)]
and_dataset = [([1, 1], 1), ([1, 0], -1), ([0, 0], -1), ([0, 1], -1)]

print("TESTE DE OR:")
test(or_dataset)
print("TESTE DE XOR:")
test(xor_dataset)
print("TESTE DE AND:")
test(and_dataset)
