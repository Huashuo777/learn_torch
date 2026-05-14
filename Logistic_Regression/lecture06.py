import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[0.0],[0],[1]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)   

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(torch.tensor([2.0])))
x = np.linspace(0,10,200)#从0到10取两百个点
x_t = torch.tensor(x,dtype=torch.float32).view((200,1))
y_t = model(x_t)
y = y_t.detach().cpu().numpy()
plt.plot(x, y, label = 'Probability')
plt.plot([0,10],[0.5,0.5],c='r')#分类阈值线，从(0,0.5)到(10,0.5)的一条红线
plt.grid()#打开网格线
plt.legend()#打开图例，即把在绘图时给每条线起的名字显示出来
plt.show()