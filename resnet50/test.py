#!/usr/bin/env python
# encoding: utf-8

from model import ResNet50
from torchvision import transforms
from PIL import Image
import torch

def test1():
    model = ResNet50(100)
    model = model.to('cpu')
    x = torch.randn((1, 3, 224, 224))
    y = model(x)
    print("y shape:", y.shape)
    return

def test2():
    # load model
    model = ResNet50(5)
    model.load_state_dict(torch.load('./weights/best.pth'))
    model.eval()
    # load image
    transform = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差进行归一化
    ])
    #img = transform('./images/sunflower.jpg')
    img = transform('./images/daisy.jpg')
    img = img.unsqueeze(0)
    # classify
    p = model(img)
    print("predict:", p)
    return

if __name__ == '__main__':
    test2()
