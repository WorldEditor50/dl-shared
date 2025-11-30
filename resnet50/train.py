#!/usr/bin/env python
# encoding: utf-8
import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ResNet50
from utils import BasicDataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 32
lr = 1e-3
max_epoch = 1000
image_size = 224
model_path = './weights'
data_path = '/mnt/f/dataset/flowers'

def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    with torch.no_grad():
        for imgs, labels in loader:
            x = imgs.to(device)
            yt = labels.squeeze().to(device)
            y = model(x)
            yp = y.argmax(dim=1)
            index = yt.argmax(dim=1)
            correct += torch.eq(yp, index).sum().float().item()
    return correct/total

def main():
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # dataset
    train_dataset = BasicDataset(data_path, image_size, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validate_dataset = BasicDataset(data_path, image_size, 'valid')
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_dataset = BasicDataset(data_path, image_size, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # model
    model = ResNet50(5).to(device)
    # loss
    crossEntropy = torch.nn.CrossEntropyLoss()
    # optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-8)
    # training
    best_param = os.path.join(model_path,'best.pth')
    if os.path.exists(best_param):
        model.load_state_dict(torch.load(best_param))
        test_acc = evaluate(model, test_dataloader)
        print('current test accuracy:', test_acc)
    max_acc = 0
    for i, epoch in enumerate(range(max_epoch)):
        for _, (imgs, labels) in enumerate(tqdm(train_dataloader, desc='epoch={}'.format(i))):
            # train
            x = imgs.to(device)
            yt = labels.squeeze().to(device)
            y = model(x)
            loss = crossEntropy(y, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # learning scheduler
        scheduler.step()
        # validate
        valid_acc = evaluate(model, validate_dataloader)
        test_acc = evaluate(model, test_dataloader)
        print("validate accuracy:", valid_acc, ", test accuracy:", test_acc)
        if valid_acc > max_acc:
            modelFile = os.path.join(model_path, 'best.pth')
        else:
            modelFile = os.path.join(model_path, 'last.pth')
        torch.save(model.state_dict(), modelFile)
    return

if __name__ == '__main__':
    main()
