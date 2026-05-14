import torch

x_data = torch.tensor([[1.0,0.1],[2.0,0.2],[3.0,0.3]])
y_data = torch.tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(2,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel() 

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Rprop(model.parameters(), lr = 0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print('w1 = ',model.linear.weight[0,0].item())
print('w2 = ',model.linear.weight[0,1].item())
print('b = ',model.linear.bias.item())

x_test = torch.tensor([4.0,0.4])
y_test = model(x_test)
print('y_pred = ', y_test.item())