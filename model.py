"""

Imports

"""
import torch
import os
from torch import nn
import torch.nn.functional as F

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
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Linear(4 * 4 * 64, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    def learn(self, image, label, optimizer, loss_func) -> float:
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