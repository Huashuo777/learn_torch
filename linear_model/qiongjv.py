import numpy as np
import matplotlib.pyplot as plt

x_data = [1, 2, 3]
t_data = [2, 4, 6]

def forward(x):
    return x*w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

w_list = []
mse_list = []

for w in np.arange(0, 4.1, 0.1):
    print('w = ',w)
    l_sum = 0
    for x_val, y_val in zip(x_data, t_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
    Mse = l_sum / len(x_data)
    print("Mse = ", Mse)
    w_list.append(w)
    mse_list.append(Mse)

plt.plot(w_list, mse_list)
plt.ylabel("Loss")
plt.xlabel("W")
plt.show()
