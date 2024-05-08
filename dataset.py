"""

Imports

"""
import torch
from torchvision.io import read_image, ImageReadMode
import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch.nn.functional as nnf

import matplotlib.pyplot as plt

"""

Constants

"""
IMG_DIR = "image"
CSV_DIR = "csv"

"""

Classes

"""
class MyDataset(Dataset):
    def __init__(self, csv_path, transform=None) -> None:
        self.__csv_path = csv_path
        self.__transform = transform

        self.__convertImagesToCsv()
        self.__dataframe = pd.read_csv(os.path.join(CSV_DIR, self.__csv_path))

    def __len__(self):
        return len(self.__dataframe)

    def __getitem__(self, index):
        image = self.__dataframe.iloc[index].values
        label = int(image[0])
        image = np.delete(image, 0)
        image = np.reshape(image, (28,28))

        plt.title(label)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.show()

        if self.__transform:
            image = self.__transform(image)

        image = image.to(torch.float)
        
        return image, label

    def __convertImagesToCsv(self) -> None:
        if not os.path.exists(CSV_DIR):
            os.mkdir(CSV_DIR)

        with open(os.path.join(CSV_DIR, self.__csv_path), 'ab') as f:
            f.truncate(0)
            f.write(bytes("label,", 'utf-8'))
            for i in range(0,784):
                f.write(bytes("pixel"+str(i), "utf-8"))
                if i != 783:
                    f.write(bytes(",", "utf-8"))
                else:
                    f.write(bytes("\n", "utf-8"))

        if not os.path.exists(IMG_DIR):
            os.mkdir(IMG_DIR)

        for img in os.listdir(IMG_DIR):
            image = read_image(os.path.join(IMG_DIR,img), ImageReadMode.GRAY)
            # image = nnf.interpolate(image, (28,28), mode='bicubic')
            image = image/255
            image = torch.flatten(image)
            image = torch.reshape(image, (1, -1))

            with open('csv/custom.csv', 'ab') as f:
                f.write(bytes(img.split(".")[0] + ",", "utf-8"))
                np.savetxt(f, image, delimiter=",", fmt='%f')
                f.write(bytes("\n", "utf-8"))