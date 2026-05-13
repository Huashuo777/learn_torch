import torch
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w1 = torch.tensor([0.5], requires_grad=True)
w2 = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
lr = 0.01

def forward(x):
    return x**2 * w1 + x * w2 + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict(before training)", 4, forward(4))

for epoch in range(100):
    for i in range(len(x_data)):
        l = loss(x_data[i], y_data[i])
        l.backward()
        print('\tgrad:', x_data[i], y_data[i], w1.grad.item(),w2.grad.item(),b.grad.item())
        with torch.no_grad():
            w1 -= lr * w1.grad
        with torch.no_grad():
            w2 -= lr * w2.grad
        with torch.no_grad():
            b -= lr * b.grad

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("progress:",epoch, l.item())
print('now,predict:',forward(4))

