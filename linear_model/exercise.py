import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1, 2, 3])
t_data = np.array([2, 4, 6])

def forward(x, w, b):
    return x * w + b

def mes_loss(x, t, w, b):
    y_pred = forward(x, w, b)
    return np.mean((y_pred - t) ** 2)

w_list = np.arange(0, 4.1, 0.1)
b_list = np.arange(-1, 1.1, 0.1)
W, B = np.meshgrid(w_list, b_list) #所有组合形成网格
Mse = np.zeros_like(W)


for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Mse[i,j] = mes_loss(x_data, t_data, W[i,j], B[i,j])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(W,B,Mse,alpha = 0.8)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Mse_loss')
ax.set_title('loss_surface: wx + b')

plt.show()
