"""

Imports

"""
import torch
import os
from model import Model
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

"""

Constants

"""
SAVE_PATH = "./model"
OPTIMIZER_FILE_NAME = "optimizer.pth"

"""

Classes

"""
class Trainer:
    def __init__(self, model: Model, optimizer, loss_func) -> None:
        self.__model = model
        self.__optimizer = optimizer
        self.__loss_func = loss_func

        # Load trained model and optimizer
        self.__model.load(SAVE_PATH)
        self.__loadOptimizer(SAVE_PATH)

    # Train itertation for model
    def train(self, epochs: int, train_dataloader: DataLoader, eval_dataloader: DataLoader) -> None:
        train_loss, test_loss, test_accuracy = [], [], []

        for epoch in range(epochs):
            # Train
            self.__model.train()
            for _, (x, y) in enumerate(train_dataloader):
                loss = self.__model.learn(x, y, self.__optimizer, self.__loss_func)
                train_loss.append(loss)

            # Evaluate
            self.__model.eval()
            with torch.no_grad():
                for _, (x, y) in enumerate(eval_dataloader):
                    loss, accuracy = self.__model.evaluate(x, y, self.__loss_func)
                    test_loss.append(loss)
                    test_accuracy.append(accuracy)

            # Save model and optimizer
            self.__model.save(SAVE_PATH)
            self.__saveOptimizer(SAVE_PATH)

            print(f"Epoch: {epoch+1}, Train Loss: {np.mean(train_loss)}, Test Loss: {np.mean(test_loss)}, Test Accuracy: {np.mean(test_accuracy)}")

        print("\nTraining is done")

    # Test iteration for model
    def test(self, dataloader: DataLoader) -> None:
        for _, (x, _) in enumerate(dataloader):
            predictedNumber = self.__model.test(x)
            plt.title(predictedNumber)
            plt.imshow(x.squeeze(), cmap='gray')
            plt.show()

    def __loadOptimizer(self, path: str) -> None:
        path = os.path.join(path, OPTIMIZER_FILE_NAME)
        if os.path.exists(path):
            self.__optimizer.load_state_dict(torch.load(path))
        else:
            print(f"There is no optimizer to load at {path}")

    def __saveOptimizer(self, path: str) -> None:
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, OPTIMIZER_FILE_NAME)
        torch.save(self.__optimizer.state_dict(), path)