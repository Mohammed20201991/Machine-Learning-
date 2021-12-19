import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  #relu,tanh
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
# fully connected layer 
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size ,50)
        self.fc2 = nn.Linear(50,num_classes)
    def forward(self,x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x # <--- THIS
model =NN(784,10)
print(model)
# x= torch.randn(64,784)
# print(model(x).shape)
device = ("cuda" if torch.cuda.is_available() else "cpu")
# hyper parameters
input_size = 784
num_classes = 10
learning_rate = 0.0001
batch_size = 64
num_epotchs = 1
# Load Data
train_dataset = datasets.MNIST(root= 'dataset/', train = True,transform = transforms.ToTensor(),download= True)
train_loader = DataLoader(dataset = train_dataset , batch_size =batch_size,shuffle= True)
test_dataset = datasets.MNIST(root= 'dataset/', train = False,transform = transforms.ToTensor(),download= True)
test_loader = DataLoader(dataset = test_dataset , batch_size =batch_size,shuffle= True)
model = NN(input_size = input_size,num_classes = num_classes).to(device)
# Loss and Optimizer
criterion =nn.CrossEntropyLoss()
# print(model.parameters())
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
print(criterion,optimizer)
# Train Networkprint(len(train_loader))
for epotch in range(num_epotchs):
    for batch_idx,(data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data.to(device =device)
        targets = targets.to(device)
#         print(data.shape)
        # Get to correct shape
        data = data.reshape(data.shape[0],-1)
        #forward
        scores = model(data)
        loss = criterion(scores,targets)
        #Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient decent or Adam step 
        optimizer.step()
# Check Accurcy on train & test to see how good our model 
def check_accurcy(loader,model):
    if loader.dataset.train:
        print('check accurcy on train data ')
    else :
        print('check accurcy on test data ')

    num_correct = 0 
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0],-1)   
            print(scores.max(1))
            _, predictions =  scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct}/{num_samples} with accurcy{float(num_correct)/ float(num_samples)*100:.2f}')
        model.train()
#         return acc
check_accurcy(train_loader,model)
check_accurcy(test_loader,model)
