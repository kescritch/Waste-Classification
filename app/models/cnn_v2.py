import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class CNN_V2(nn.Module): 
    def __init__(self, num_classes:int):
        super().__init__()
        
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT) #pretrained mobilenetv2 model
                                                  
        for param in self.base_model.parameters(): #freeze the pretrained layers
            param.requires_grad = False
            
        in_features = self.base_model.classifier[1].in_features #get the number of input features for the final layer
        self.base_model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes) #replace the final layer with a new one for our number of classes
        
    def forward(self, x):
        x = self.base_model(x)
        return x
        