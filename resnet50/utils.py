#!/usr/bin/env python
# encoding: utf-8
import glob
import random
import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class BasicDataset(Dataset):
    def __init__(self, root, img_size, mode):
        super(BasicDataset, self).__init__()
        self.root = root
        self.img_size = img_size
        self.mode = mode
        self.images = []
        self.labels = []
        self.name2label = {}
        # csv: (image, label)
        #print(os.path.join(self.root))
        self.load_csv("data.csv")

    def load_csv(self, filename):
        for name in sorted(os.listdir(os.path.join(self.root))):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())

        print(self.name2label)
        if not os.path.exists(os.path.join(self.root, filename)):
            # traverse folders
            # dataset, label, data
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            print("images:", len(images))

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as file:
                writer = csv.writer(file)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
        # load from csv
        with open(os.path.join(self.root, filename)) as file:
            reader = csv.reader(file)
            for row in reader:
                img, label = row
                label = int(label)
                self.images.append(img)
                self.labels.append(label)
        assert len(self.images) == len(self.labels)

        total = len(self.images)
        # 70% --> train, 20% --> validate, 10% --> test
        if self.mode == 'train':
            self.images = self.images[:int(0.7*total)]
            self.labels = self.labels[:int(0.7*total)]
        elif self.mode == 'valid':
            self.images = self.images[int(0.7*total):int(0.9*total)]
            self.labels = self.labels[int(0.7*total):int(0.9*total)]
        else:
            self.images = self.images[int(0.9*total):]
            self.labels = self.labels[int(0.9*total):]
        return

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img = self.images[index]
        labelID = self.labels[index]
        img_size = int(self.img_size*1.5)
        transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            # normalize with image net parameter
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        ])
        img = transform(img)
        label = torch.zeros((1, 5))
        label[0][labelID] = 1
        return img, label

    def denormalize(self, x_):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x:[c,h,w], mean:[3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_*std + mean
        return x

