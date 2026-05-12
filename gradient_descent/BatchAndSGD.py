import numpy as np
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])

w = 1.0
lr = 0.1

def forward(x, w):
    return x * w

def loss(x, y, w):
    y_pred = forward(x, w)
    return (y_pred - y) ** 2

def gradient(x, y, w):
    return 2 * x * (x * w - y)

for epoch in range(100):
    #SGD
    # l = 0
    # for i in range(x_data.shape[0]):
    #         l += loss(x_data[i], y_data[i], w)
    #         grad = gradient(x_data[i], y_data[i], w)
    #         w = w - lr * grad
    #         print('\tgrad:', x_data[i], y_data[i], grad)
    # print('epoch:', epoch,' loss:', l/x_data.shape[0])
    
    #Batch GD
    l = np.mean(loss(x_data, y_data, w))
    grad = np.mean(gradient(x_data,y_data,w))
    w = w - lr * grad
    print(l,"-- xxxxx --",grad)
    print('epoch:', epoch,' loss:', l)
print(4,'-pre:',forward(4,w))