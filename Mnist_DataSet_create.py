import os

import numpy as np
import torchvision

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset,random_split

from PIL import Image

import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MNISTDataset(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.len_dataset = 0
        self.data_list = [] # список кортежей состоящих из пути до
                            # файла и позицией класс в one_hot векторе.


        for path_dir, dir_list, file_list in os.walk(path):
            # logger.info(path_dir)
            if path_dir == path:
                self.classes = sorted(dir_list)
                self.class_to_ind = {
                    cls_name: i for i, cls_name in enumerate(self.classes)
                }
                continue

            cls = path_dir.split("\\")[-1]
            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_ind[cls]))

            self.len_dataset += len(file_list)



    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        file_path, target = self.data_list[index]
        np_image = np.array(Image.open(file_path))

        if self.transform:
            np_image = self.transform(np_image)

        return np_image, target


train_data = MNISTDataset(r"D:\PyTorch_Dubinin\mnist\training")
test_data = MNISTDataset(r"D:\PyTorch_Dubinin\mnist\testing")

# img, position = testing_data[3000]
# print(testing_data.classes[position])
# plt.imshow(img, cmap="gray")
# plt.show()
# print("Window closed")

train_data, val_data = random_split(train_data, [0.8, 0.2])

train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False)


print(val_loader.multiprocessing_context)