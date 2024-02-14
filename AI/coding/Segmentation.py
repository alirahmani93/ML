"""
For Download Cat & Dog images. use this commands:
```
cd ~/<your_path>
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz &&
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz &&
tar -xf images.tar.gz &&
tar -xf annotations.tar.gz
```
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

import os
from glob import glob
import matplotlib.pyplot as plt
import cv2

img = cv2.imread(input_img_paths[13])[:, :, ::-1]

plt.axis("off")
plt.imshow(img)
input_dir = "images/"
target_dir = "annotations/trimaps/"

input_img_paths = sorted(glob(input_dir + "/*.jpg"))
target_paths = sorted(glob(target_dir + "/*.png"))


def display_target(target_array):
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])


img = cv2.imread(target_paths[13])
display_target(img)


class SegmentDataset(Dataset):
    def __init__(self, image_dir, target_dir, img_size=(200, 200),
                 random_state=1337, train=True, transform=None):

        all_images_path = sorted(glob(image_dir + "/*.jpg"))
        all_targets_path = sorted(glob(target_dir + "/*.png"))

        random.Random(random_state).shuffle(all_images_path)
        random.Random(random_state).shuffle(all_targets_path)

        self.transform = transform
        self.img_size = img_size

        num_val_samples = 1000
        if train:
            self.images_path = all_images_path[num_val_samples:]
            self.targets_path = all_targets_path[num_val_samples:]
        else:
            self.images_path = all_images_path[:num_val_samples]
            self.targets_path = all_targets_path[:num_val_samples]

    def __len__(self):
        return len(self.images_path)

    def image_read(self, path):
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, self.img_size)
        return im

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        target_path = self.targets_path[idx]
        image = self.image_read(image_path)
        target = self.image_read(target_path)[:, :, 0]
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        target = torch.from_numpy(target.astype("uint8")) - 1
        return image.float(), target


train_dataset = SegmentDataset(input_dir, target_dir, train=True)
val_dataset = SegmentDataset(input_dir, target_dir, train=False)

train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=64)


def display_target_tensor(target_tensor):
    normalized_tensor = target_tensor * 127
    plt.figure()
    plt.axis("off")
    plt.imshow(normalized_tensor)


im, target = next(iter(train_dl))
plt.axis("off")
plt.imshow(im[0].permute(1, 2, 0))
display_target_tensor(target[0])


class Model(nn.Module):
    def __init__(self, in_channel, num_classes, ):
        super().__init__()
        self.seq_encode = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=1, stride=2, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channel, 64, kernel_size=1, padding='same'),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=1, stride=2, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding='same'),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=1, stride=2, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, padding='same'),
            nn.ReLU(),

        )

        self.seq_decode = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )

        self.seq_out = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=3, padding='same'),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.seq_encode(x)
        x = self.seq_decode(x)
        x = self.seq_out(x)
        return x


model = Model(64, 3)
