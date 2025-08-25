class SimpleRegressor:
    def __init__(self):
        pass

    def model(self, dataset):
        x, y = list(zip(*dataset))
        xs = sum(x)
        ys = sum(y)
        xys = sum(map(lambda t: t[0]*t[1], zip(x, y)))
        n = len(dataset)
        self.w1 = (n*xys - ys*xs)/(n*sum(map(lambda xi: xi*xi, x)) - xs*xs)
        self.w0 = (ys - self.w1*xs)/n
        
    def predict(self, x):
        return self.w0 + self.w1*x

reg = SimpleRegressor()
reg.model([(2,3), (3,5), (1,1)])
print(reg.predict(3))
print(reg.w0)
print(reg.w1)
