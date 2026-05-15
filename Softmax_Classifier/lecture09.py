import torch

y = torch.LongTensor([0])
z = torch.tensor([[0.2,0.1,-0.1]])
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(z, y)
print(loss)


# import numpy as np

# y = np.array([1,0,0])
# z = np.array([0.2,0.1,-0.1])
# y_pred = -np.log(np.exp(z)/np.exp(z).sum())
# loss = (y*y_pred).sum()
# print(loss)