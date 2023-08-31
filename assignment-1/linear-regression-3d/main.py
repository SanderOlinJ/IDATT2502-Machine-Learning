import torch
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("../data/day_length_weight.csv")
x_train = torch.tensor(data["# day"].values, dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor(data['length'].values, dtype=torch.float).reshape(-1, 1)
z_train = torch.tensor(data['weight'].values, dtype=torch.float).reshape(-1, 1)


class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], dtype=torch.float, requires_grad=True)
        self.b = torch.tensor([[0.0]], dtype=torch.float, requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(300000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
ax = plt.figure().add_subplot(projection='3d')

x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
y = torch.tensor([[torch.min(y_train)], [torch.max(y_train)]])
z = torch.tensor([[torch.min(z_train)], [torch.max(z_train)]])

ax.scatter(x_train, y_train, z_train, 'o')
ax.plot(x, y, z, color='r', label='$(x^{(i)},y^{(i)},z^{(i)})$')
ax.set_xlabel('x days')
ax.set_ylabel('y length')
ax.set_zlabel('z weight')
plt.legend()
plt.show()

