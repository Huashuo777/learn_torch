import torch 
from torchvision import transforms
from torchvision import datasets #注意此处导入的是直接可用的数据集 而非最顶层的父类数据集框架
from torch.utils.data import DataLoader #这里导入的则是tools
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))#标准化，传入通道个数元素个的元组，前者均值后者标准差
])

train_dataset = datasets.MNIST(root = '/dataset/mnist/',
                               train = True,
                               download = True,
                               transform= transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='dataset/mnist/',
                              train = False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512,256)
        self.l3 = torch.nn.Linear(256,128)
        self.l4 = torch.nn.Linear(128,64)
        self.l5 = torch.nn.Linear(64,10)
        self.activation = F.relu #此处不要调用函数，而只是定义函数，或者使用torch.nn.Relu()这是一个层(模块)
    def forward(self,x):
        x = x.view(-1,784)
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))
        x = self.activation(self.l4(x))
        x = self.l5(x)
        return x
    
model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)

def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss : %.3f' % (epoch+1, batch_idx+1, running_loss / 300))
            running_loss = 0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            #total += labels.size(0) #size是方法 shape是属性不必加括号
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set:%d %%' % (100 * correct/total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()