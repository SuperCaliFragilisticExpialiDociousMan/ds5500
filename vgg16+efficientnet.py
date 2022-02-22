
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import shutil

'''
def remove_file(old_path, new_path):
    print(old_path)
    print(new_path)
    filelist = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    print(filelist)
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        print('src:', src)
        print('dst:', dst)
        shutil
        move(src, dst)
'''

'''
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Conv1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, MaxPooling1D
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
'''

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def dataframe_creation():
    df4 = pd.read_csv('/users/shizhengyan/Desktop/Neu/DS5500/Project/archive(1)/full_df.csv')
    df4['filename'] = '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/preprocessed_images/' + df4['filename']
    df4['Left-Fundus'] = '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/preprocessed_images/' + df4[
        'Left-Fundus']
    df4['Right-Fundus'] = '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/preprocessed_images/' + df4[
        'Right-Fundus']
    df4['Line'] = df4['Left-Diagnostic Keywords'] + ' | ' + df4['Right-Diagnostic Keywords']
    #df4 = df4.drop(['filepath', 'target'], axis=1)
    df4 = df4.drop('filepath', axis=1)
    return df4

df = dataframe_creation()
print(df.columns)
listImg = os.listdir('/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/preprocessed_images')
string = '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/preprocessed_images/'
list2 = list(map(lambda orig_string: string + orig_string , listImg))
print(list2)

'''
for i in range(len(df)):
    if df.iloc[i]['Right-Fundus'] in list2 and 0<=i<=3195:
        print(1)

        if df.iloc[i, 15][2] == 'N':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/N')

        elif df.iloc[i, 15][2] == 'D':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/D')

        elif df.iloc[i, 15][2] == 'G':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/G')

        elif df.iloc[i, 15][2] == 'C':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/C')

        elif df.iloc[i, 15][2] == 'A':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/A')

        elif df.iloc[i, 15][2] == 'H':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/H')

        elif df.iloc[i, 15][2] == 'M':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/M')

        elif df.iloc[i, 15][2] == 'O':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/O')

    if df.iloc[i]['Left-Fundus'] in list2 and i>=3196:
        print(1)

        if df.iloc[i, 15][2] == 'N':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/N')

        elif df.iloc[i, 15][2] == 'D':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/D')

        elif df.iloc[i, 15][2] == 'G':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/G')

        elif df.iloc[i, 15][2] == 'C':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/C')

        elif df.iloc[i, 15][2] == 'A':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/A')

        elif df.iloc[i, 15][2] == 'H':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/H')

        elif df.iloc[i, 15][2] == 'M':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/M')

        elif df.iloc[i, 15][2] == 'O':
            shutil.move(df.iloc[i,-2], '/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/O')
'''

#shutil.move('/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/A/0_left.jpg','/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image/C')



################################################################################################################################


os.listdir('/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image')

import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

import glob
from tqdm.notebook import tqdm
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


class custom_dataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.data = []
        self.transform = transform

        for img_path in tqdm(glob.glob(root_dir + "/*/**")):
            class_name = img_path.split("/")[-2]
            self.data.append([img_path, class_name])

        self.class_map = {}
        for index, item in enumerate(os.listdir(root_dir)):
            self.class_map[item] = index
        print(f"Total Classes:{len(self.class_map)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)

        if self.transform:
            img = self.transform(img)

        return img, class_id


root_dir = r'/users/shizhengyan/Desktop/Neu/DS5500/Project/medium-risk/after_image'


def create_transforms(normalize=False, mean=[0, 0, 0], std=[1, 1, 1]):
    if normalize:
        my_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            #             transforms.ColorJitter(brightness=0.3,saturation=0.5,contrast=0.7,),
            #             transforms.RandomRotation(degrees=33),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    else:
        my_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            #             transforms.ColorJitter(brightness=0.3,saturation=0.5,contrast=0.7,p=0.57),
            #             transforms.RandomRotation(degrees=33),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

    return my_transforms


BS = 8
num_classes = 39

my_transforms = create_transforms(normalize=False)
dataset = custom_dataset(root_dir, my_transforms)
print(len(dataset))

train_set, val_set = torch.utils.data.random_split(dataset, [5000, 1392], generator=torch.Generator().manual_seed(7))
train_loader = DataLoader(train_set, batch_size=BS, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BS, shuffle=True)


def get_mean_std(loader):
    # var=E[x^2]-(E[x])^2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data,
                                   dim=[0, 2, 3])  # we dont want to a singuar mean for al 3 channels (in case of RGB)
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


mean, std = get_mean_std(train_loader)
print(mean, std)

my_transforms = create_transforms(normalize=True, mean=mean, std=std)
dataset = custom_dataset(root_dir, my_transforms)
print(len(dataset))

train_set, val_set = torch.utils.data.random_split(dataset, [5000, 1392], generator=torch.Generator().manual_seed(7))
train_loader = DataLoader(train_set, batch_size=BS, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BS, shuffle=True)

import matplotlib.pyplot as plt

dataiter = iter(train_loader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

plt.imshow(images[0].permute(1, 2, 0))

vgg_model = torchvision.models.vgg16(pretrained=True)
print(vgg_model)

vgg_model = torchvision.models.vgg16(pretrained=True)

for param in vgg_model.parameters():
    param.requires_grad = False


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# vgg_model.avgpool=Identity()
vgg_model.classifier = nn.Sequential(
    nn.Linear(25088, 2048),
    nn.ReLU(),
    nn.Dropout(p=0.37),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024, num_classes)
)

vgg_model.to(device)

EPOCHS = 5
LR = 1e-3


def train_model(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9, verbose=True)

    for epoch in range(EPOCHS):
        losses = []
        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, targets) in loop:
            data = data.to(device)
            targets = targets.to(device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent/adam step
            optimizer.step()
        mean_loss = sum(losses) / len(losses)
        scheduler.step()

        print(f"Loss at Epoch {epoch + 1}:\t{mean_loss:.5f}\n")


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()


train_model(vgg_model)

print("Training accuracy:", end='\t')
check_accuracy(train_loader, vgg_model)
print("Validation accuracy:", end='\t')
check_accuracy(val_loader, vgg_model)

from efficientnet_pytorch import EfficientNet


def create_eff_net(version='b1', trainable=False):
    eff_model = EfficientNet.from_name(f'efficientnet-{version}')

    for param in eff_model.parameters():
        param.requires_grad = trainable

    num_ftrs = eff_model._fc.in_features

    eff_model._fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.37),
        nn.Linear(1024, num_classes)
    )

    eff_model.to(device)

    return eff_model


eff_model = create_eff_net(version='b3', trainable=True)

train_model(eff_model)

print("Training accuracy:", end='\t')
check_accuracy(train_loader, eff_model)
print("Validation accuracy:", end='\t')
check_accuracy(val_loader, eff_model)

print('finish')