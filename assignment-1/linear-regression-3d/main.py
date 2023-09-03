import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("../data/day_length_weight.csv")
x_train = torch.tensor(data[['# day', 'length']].values, dtype=torch.float).reshape(-1, 2)
y_train = torch.tensor(data['weight'].values, dtype=torch.float).reshape(-1, 1)


class LinearRegressionModel:

    def __init__(self):
        self.W = torch.rand((2, 1), dtype=torch.float, requires_grad=True)
        self.b = torch.rand((1, 1), dtype=torch.float, requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0000001)
for epoch in range(300000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Create a mesh grid for the visualization
xx, yy = np.meshgrid(np.linspace(min(x_train[:, 0]), max(x_train[:, 0]), 50),
                     np.linspace(min(x_train[:, 1]), max(x_train[:, 1]), 50))
zz = np.c_[xx.ravel(), yy.ravel()]
pred = model.f(torch.tensor(zz, dtype=torch.float)).detach().numpy()
zz = pred.reshape(xx.shape)

# Visualize result
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(xx, yy, zz, color="g", alpha=0.5)
ax.scatter(x_train[:, 0], x_train[:, 1], y_train, 'o', color='b')

ax.set_xlabel('x days')
ax.set_ylabel('y length')
ax.set_zlabel('z weight')
plt.show()
