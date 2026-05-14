import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')##只影响python搜索模块的路径，不能影响到numpy

#此处路径要根据实际的工作目录的路径，而非存放文件的路径来设置相对路径
xy = np.loadtxt('dataset/diabetes.csv.gz', delimiter=',',dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.activate = torch.nn.LeakyReLU()

    def forward(self,x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x
    
model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)   

for epoch in range(100):
    
    #Forward
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch, loss.item())

    #Backward
    optimizer.zero_grad()
    loss.backward()
    
    #Update
    optimizer.step()
