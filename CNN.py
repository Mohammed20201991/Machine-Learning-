import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  #relu,tanh
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

train_dataset = datasets.MNIST(root= 'dataset/', train = True,transform = transforms.ToTensor(),download= True)
train_loader = DataLoader(dataset = train_dataset , batch_size =batch_size,shuffle= True)
test_dataset = datasets.MNIST(root= 'dataset/', train = False,transform = transforms.ToTensor(),download= True)
test_loader = DataLoader(dataset = test_dataset , batch_size =batch_size,shuffle= True)


class CNN(nn.Module):
    def __init__(self,in_channels =1 ,num_classes= 10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 8, kernel_size =(3,3),stride = (1,1),padding= (1,1))# same convlutional
        self.pool  = nn.MaxPool2d(kernel_size =(2,2),stride =(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8,out_channels = 16, kernel_size =(3,3),stride = (1,1),padding= (1,1))# same convlutional
        self.fc1 = nn.Linear(16*7*7,num_classes)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x .reshape(x.shape[0],-1)
        x = self.fc1(x)
        return (x)
model =CNN()
print(model)
# x= torch.randn(64,1,28,28)
# # print(x)
# print(model(x).shape)
# exit()

in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epotchs = 5

model = CNN().to(device)  #seted by default in_channels = in_channels,num_classes = num_classes
print(model)

criterion =nn.CrossEntropyLoss()
# print(model.parameters())
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
print(criterion,optimizer)


print(len(train_loader))
for epotch in range(num_epotchs):
    for batch_idx,(data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device = device)
        targets = targets.to(device = device)
        print("jj",data.shape)
        
        print(data.shape)
        #forward
        scores = model(data)
#         print(scores)
        loss = criterion(scores,targets)
        #Backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient decent or Adam step 
        optimizer.step()
