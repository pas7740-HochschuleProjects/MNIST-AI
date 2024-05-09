"""

Imports

"""
import torch
import os
from torch import nn

"""

Constants

"""
MODEL_FILE_NAME = "model.pth"

"""

Classes

"""
class Model(nn.Module):
    def __init__(self, device: str) -> None:
        super(Model, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output
    
    def learn(self, image, label, optimizer, loss_func) -> float:
        # Send data to device
        image, label = image.to(self.device), label.to(self.device)
        # Reset Autograd
        optimizer.zero_grad()
        # Compute prediction and error
        prediction = self(image)
        loss = loss_func(prediction, label)
        # Backward Propagation
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def evaluate(self, image, label, loss_func) -> tuple:
        image, label = image.to(self.device), label.to(self.device)
        # Compute prediction and error
        prediction = self(image)
        loss = loss_func(prediction, label)
        # Compute accuracy
        _, predictedNumber = torch.max(prediction.data, 1)
        return loss.item(), (predictedNumber == label).sum().item() / predictedNumber.size(0)
    
    def test(self, image) -> int:
        image = image.to(self.device)
        prediction = self(image)
        # Compute predicted number
        _, predictedNumber = torch.max(prediction.data, 1)
        return predictedNumber.item()
        
    
    def save(self, path: str) -> None:
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, MODEL_FILE_NAME)
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        path = os.path.join(path, MODEL_FILE_NAME)
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
        else:
            print(f"There is no model to load at {path}")