import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1.0,2.0,3.0])
y_data = np.array([2.0,4.0,6.0])

w = torch.Tensor([1.0])
w.required_grad = True
lr = 0.1 

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict(before training)", 4, forward(4))

for epoch in range(100):
    for i in x_data.shape[0]:
        l = loss(x_data[i], y_data[i])
        l.backward()
        print('\tgrad:', x_data[i], y_data[i], w.grad.item())
        w.data = w.data - w.grad.data * lr

        w.grad.data.zero_()
    print("progress:",epoch, l.item())
print('now,predict:',forward(4))