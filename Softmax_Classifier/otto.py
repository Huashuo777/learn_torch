import torch 
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

class Ottodataset(Dataset):
    # 可以直接读数的那种
    # def __init__(self, filepath, train, test_size = 800, seed=42):#固定seed避免两次打乱之后随机性导致验证集可能和训练集有重合
    #     xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32,skiprows=1)
        
    #     #生成随机下标（打乱数据集的效果）
    #     rng = np.random.default_rng(seed)
    #     indices = rng.permutation(len(xy))

    #     if train == True:
    #         self.inputs = torch.from_numpy(xy[:-test_size,1:94]).float()
    #         self.target = torch.from_numpy(xy[:-test_size,95]).long() #此处注意CEL的标签要求
            
    #     else:
    #         self.inputs = torch.from_numpy(xy[test_size:,1:94])
    #         self.target = torch.from_numpy(xy[test_size:,95])
        
    #     self.target = self.target-1    
    #     self.len = self.inputs.size(0)
    
    # 对于kaggle带的原生标签的读法
    def __init__(self, dataframe):
        #pandas的DataFrame包含 行索引index 列名columns 数据values
        feature_cols = [f'feat_{i}' for i in range(1,94)]
        
        x = dataframe[feature_cols].to_numpy(dtype=np.float32)
        y = dataframe['target'].str.replace('Class_','').astype(int).to_numpy() - 1 #删掉标签的'class_'字符并且转为int后组织成numpy并减1变成0-8

        self.inputs = torch.from_numpy(x).float()
        self.targets = torch.from_numpy(y).long() #整形序列标签
        self.len = self.inputs.size(0)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    
    def __len__(self):
        return self.len

def split_dataframe(filepath, val_ratio = 0.2, seed = 42):
    df = pd.read_csv(filepath)#自动把表头放进df.columns且算len(df)不会算入

    rng = np.random.default_rng(seed) #rng随机工具对象 创建一个随机数生成器 rng，并用 seed 固定它的随机规则
    indices = rng.permutation(len(df)) #返回打乱后的索引 让这个随机工具 rng 生成一个从 0 到 len(df)-1 的随机排列。

    val_size = int(len(df) * val_ratio)
    
    #定索引
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    #取数据且重新排索引号
    train_df = df.iloc[train_indices].reset_index(drop = True) #iloc 是 pandas 里的按行号位置取数据且不会丢列名
    val_df = df.iloc[val_indices].reset_index(drop = True) #drop表示丢掉旧索引，不要把旧索引作为新的一列保存。如果不写 drop=True，旧索引可能会变成一个新列，叫做 index，这通常不是我们想要的。

    return train_df, val_df
    # pandas的DataFrame包含 行索引index 列名columns 数据values
train_df, val_df = split_dataframe('dataset/otto/train.csv')

train_dataset = Ottodataset(train_df)
test_dataset = Ottodataset(val_df)
train_loader = DataLoader(dataset = train_dataset,
                          batch_size = 32,
                          shuffle = True,  #打乱数据
                          num_workers = 2) #控制数据加载子进程数量的参数
test_loader = DataLoader(dataset = test_dataset,
                          batch_size = 32,
                          shuffle = False,  #打乱数据
                        ) #控制数据加载子进程数量的参数

class ottomodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(93,48)
        self.l2 = torch.nn.Linear(48,24)
        self.l3 = torch.nn.Linear(24,9)  
        self.activation = torch.nn.LeakyReLU() #根据类创建对象,别忘记括号
    
    def forward(self, x):
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.l3(x)
        return x
model = ottomodel()

critertion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.01, momentum=0.5)

def train(epoch):
    model.train()  # 切换到训练模式

    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = critertion(outputs, targets)
        #print(batch_idx+1,loss)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() #注意此处只取值 不要影响计算图
        if batch_idx % 300 == 299:
            print(epoch+1,batch_idx+1,'loss:%.3f'%(running_loss/300)) #记得%后的具体数据要用()

def test():
    model.eval()  # 切换到测试模式

    right = 0
    total = 0
    for batch_idx, data in enumerate(test_loader, 0):
        inputs, targets = data
        outputs = model(inputs)
        _,predicted = torch.max(outputs.data, dim = 1)
        right += (targets == predicted).sum().item()
        total += targets.size(0) #此处注意不要用data.size(0) 因为data是元组
    print('test_accuracy:',right/total)

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

        ##未完，看看gpt上如何组织数据组织代码能完成一次提交任务