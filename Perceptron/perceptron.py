import random
def dotproduct(x, w):
    sum = 0
    for i, j in zip(x, w):
        sum += i*j
    return sum

def update_weights(w, x, y):
    for i in range(len(w)):
        w[i] = w[i] + x[i]*y
    return w

def get_x_aug(x, fixed_input=1):
    return [fixed_input] + x

def training(dimension, dataset, max_iter, bias):
    w = [bias] + [0]*dimension
    for _ in range(max_iter):        
        i = random.randint(0, len(dataset)-1)
        x, y = dataset[i]
        x_aug = get_x_aug(x)
        xw = dotproduct(x_aug, w)
        if (xw >= 0 and y == -1) or (xw < 0 and y == 1):
            w = update_weights(w, x_aug, y)
    return w

def result(x, w):
    x_aug = get_x_aug(x)
    if dotproduct(x_aug, w) >= 0:
        return 1
    return -1

def test(dataset):
    w = training(2, dataset, 50, 0)
    print(f"[1, 1]={result([1, 1], w)}")
    print(f"[1, 0]={result([1, 0], w)}")
    print(f"[0, 1]={result([0, 1], w)}")
    print(f"[0, 0]={result([0, 0], w)}")

or_dataset = [([1, 1], 1), ([1, 0], 1), ([0, 0], -1), ([0, 1], 1)]
xor_dataset = [([1, 1], -1), ([1, 0], 1), ([0, 0], -1), ([0, 1], 1)]
and_dataset = [([1, 1], 1), ([1, 0], -1), ([0, 0], -1), ([0, 1], -1)]

print("TESTE DE OR:")
test(or_dataset)
print("TESTE DE XOR:")
test(xor_dataset)
print("TESTE DE AND:")
test(and_dataset)
