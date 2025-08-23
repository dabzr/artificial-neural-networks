import random

def dotproduct(x, w):
    sum = 0
    for i, j in zip(x, w):
        sum += i*j
    return sum

class Perceptron:
    def __init__(self, dataset, weights=None, fixed_input=+1) -> None:
        self.dataset = dataset
        self.model = weights
        self.fixed_input = fixed_input

    @staticmethod
    def update_weights(w, x, y):
        for i in range(len(w)):
            w[i] = w[i] + x[i]*y
        return w

    def train(self, dimension, epochs, bias):
        w = [bias] + [0]*dimension
        for _ in range(epochs):        
            i = random.randint(0, len(self.dataset)-1)
            x, y = self.dataset[i]
            x_aug = [self.fixed_input] + x
            xw = dotproduct(x_aug, w)
            if (xw >= 0 and y == -1) or (xw < 0 and y == 1):
                w = Perceptron.update_weights(w, x_aug, y)
        self.model = w
    
    def predict(self, x):
        x_aug = [self.fixed_input] + x
        if dotproduct(x_aug, self.model) >= 0:
            return 1
        return -1

def test(dataset):
    neuron = Perceptron(dataset)
    neuron.train(dimension=2, epochs=50, bias=0)
    test_cases = [[1, 1], [1, 0], [0, 1], [0, 0]]
    for case in test_cases:
        print(f"{case}={neuron.predict(case)}")

or_dataset = [([1, 1], 1), ([1, 0], 1), ([0, 0], -1), ([0, 1], 1)]
xor_dataset = [([1, 1], -1), ([1, 0], 1), ([0, 0], -1), ([0, 1], 1)]
and_dataset = [([1, 1], 1), ([1, 0], -1), ([0, 0], -1), ([0, 1], -1)]

print("TESTE DE OR:")
test(or_dataset)
print("TESTE DE XOR:")
test(xor_dataset)
print("TESTE DE AND:")
test(and_dataset)
