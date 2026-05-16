import torch 
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = datasets.MNIST(root='/dataset/mnist/',
                               train=True,
                               download=False,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle = True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='/dataset/mnist/',
                               train=False,
                               download=False,
                               transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(1,10,5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(10,20,5)
        self.affine = torch.nn.Linear(320,10)
    
    def forward(self,x):
        batch_size = x.size(0) #注意
        x = F.relu(self.conv1(x)) #要么torch.nn.functional.relu() 
                                  #要么torch.ReLU()
        x = self.pooling(x)
        x = F.relu(self.conv2(x))
        x = self.pooling(x)
        x = x.view(batch_size, -1) #此处进行batch_size个样本的flatten展开
        x = self.affine(x)
        return x #记得返回
    
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)

def train(epoch):
    model.train()  # 切换到训练模式
    
    running_loss = 0.0
    total_num = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0) #把平均损失复原成总损失，方便下边算总的batch平均损失
        total_num += inputs.size(0)
    print("epoch",epoch+1,"loss:",running_loss/total_num)

def test():
    model.eval()  # 切换到eval评估模式
    
    correct = 0
    total = 0
    with torch.no_grad(): #测试不需要计算梯度
        for batch_idx, data in enumerate(test_loader, 0):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _,predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == target).sum().item()
            total += inputs.size(0)
    print('Accuracy:',correct/total)
            
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

# target = target.squeeze(1) #只把第一维的长度为1的维度压缩掉，而如果没有参数则是把所有长度为1的维度都压缩掉 则可能把[1,1]->[]即标量而非一维向量