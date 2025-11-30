#!/usr/bin/env python
# encoding: utf-8

import torch
from model import ResNet50

def main():
    # load model
    model = ResNet50(5)
    model.load_state_dict(torch.load('./weights/best.pth'))
    model.eval()
    # export
    x = torch.randn((1, 3, 224, 224))
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model,
                      x,
                      './flower_resnet50.onnx',
                      opset_version=18,
                      input_names=input_names,
                      output_names=output_names)

    return

if __name__ == '__main__':
    main()
