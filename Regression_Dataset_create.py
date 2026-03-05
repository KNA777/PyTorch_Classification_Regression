import json
import os
from xmlrpc.client import FastParser

import numpy as np
import torchvision

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split

from PIL import Image

import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegressionDataset(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.list_name_file = os.listdir(path)

        if "coords.json" in os.listdir(path):
            self.list_name_file.remove("coords.json")
        self.len_dataset = len(self.list_name_file)

        with open(os.path.join(self.path, "coords.json"), "r") as f:
            self.dict_coords = json.load(f)
            logger.info(type(self.dict_coords))

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):

        name_file = self.list_name_file[index]
        path_image = os.path.join(self.path, name_file)

        np_image = np.array((Image.open(path_image)))
        coord = np.array((self.dict_coords[name_file]))

        if self.transform:
            np_image = self.transform(np_image)

        return np_image, coord


train_data = RegressionDataset(r"D:\PyTorch_Dubinin\dataset")

# np_image, coord = train_data[950]
# print(f"Coords is {coord[0], coord[1]}")
# plt.scatter(coord[0], coord[1], marker="x", color="blue")
# plt.imshow(np_image, cmap="gray")
# plt.show()

# train_set, val_set, test_set = random_split(train_data, [0,7, 0.1, 0.2])
#
# train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
# val_loader = DataLoader(train_set, batch_size=64, shuffle=False)
# test_loader = DataLoader(train_set, batch_size=64, shuffle=False)
