from torch import nn
import torch.nn.functional as F

class CNN_V1(nn.Module): 
    def __init__(self, num_classes:int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(56 * 56 * 64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.reshape(x.shape[0], -1) #flatten the output for the fully connected layer
       
        x = self.fc1(x)

        return x
        