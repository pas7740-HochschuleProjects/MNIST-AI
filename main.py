"""

Imports

"""
import torch
from torch import nn, optim
from model import Model
from trainer import Trainer
from argparse import ArgumentParser
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from dataset import MyDataset




"""

Constants

"""
BATCH_SIZE = 128                     # Batch count per episode
LEARNING_RATE = 0.01
                                                                        # Mean  #STD         # Standardabweichung
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])      # image = (image - mean) / std




"""

Global Variables

"""
shouldTrain = False
epochs = 10

"""




Functions

"""
# Returns target processing unit
def getProcessingUnit() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# Returns MNIST-Dataset
def getDataset(trained: bool) -> Dataset:
    return datasets.MNIST(
        root="data",
        train=trained,                  # Traindata or Testdata
        download=True,
        transform=TRANSFORM
    )


# Read arguments
def readArguments() -> None:
    global shouldTrain;
    global epochs;
    parser = ArgumentParser()
    # Add Arguments
    parser.add_argument("train", choices=['train', 'test'])
    parser.add_argument("epochs", nargs='?', default=epochs, type=int)
    # Read arguments
    args = parser.parse_args()
    shouldTrain = True if args.train == "train" else False
    epochs = args.epochs




"""

Main Program

"""
def main():
    # Model
    device = getProcessingUnit()
    model = Model(device).to(device)

    # Loss and Optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Trainer
    trainer = Trainer(model, optimizer, loss_func)

    # Train / Test
    if shouldTrain:
        train_dataloader = DataLoader(getDataset(trained=True), batch_size=BATCH_SIZE, shuffle=True)
        eval_dataloader = DataLoader(getDataset(trained=False), batch_size=BATCH_SIZE, shuffle=True)

        trainer.train(epochs, train_dataloader, eval_dataloader)
    else:
        test_dataloader = DataLoader(MyDataset("custom.csv", TRANSFORM))
        trainer.test(test_dataloader)
        print("Testing is done")


if __name__ == "__main__":
    readArguments()
    main()