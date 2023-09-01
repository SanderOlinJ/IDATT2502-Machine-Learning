import torch
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("../data/day_head_circumference.csv")
x_train = torch.tensor(data["# day"].values, dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor(data['head circumference'].values, dtype=torch.float).reshape(-1, 1)


class NonLinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], dtype=torch.float, requires_grad=True)
        self.b = torch.tensor([[0.0]], dtype=torch.float, requires_grad=True)

    # Predictor
    def f(self, x):
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = NonLinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.000001)
for epoch in range(300000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.xlabel('x days')
plt.ylabel('y head circumference')
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
x = torch.arange(torch.min(x_train), torch.max(x_train), 1.0).reshape(-1, 1)
y = model.f(x).detach()
plt.plot(x, y, label='$f(x) = 20σ(xW+b)+31$')
plt.legend()
plt.show()
