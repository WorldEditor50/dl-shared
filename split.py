#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import random
import shutil

random.seed(64)

data_dir = 'datasets/'
def main():
    if not os.path.exists(data_dir):
        print("invalid data path")
        return
    data_image_dir = os.path.join(data_dir, 'images')
    data_label_dir = os.path.join(data_dir, 'labels')

    output_dir = os.path.join(data_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    # train dir
    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    train_image_dir = os.path.join(train_dir, 'images')
    os.makedirs(train_image_dir, exist_ok=True)
    train_label_dir = os.path.join(train_dir, 'labels')
    os.makedirs(train_label_dir, exist_ok=True)
    # valid dir
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)
    val_image_dir = os.path.join(val_dir, 'images')
    os.makedirs(val_image_dir, exist_ok=True)
    val_label_dir = os.path.join(val_dir, 'labels')
    os.makedirs(val_label_dir, exist_ok=True)
    # test dir
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    test_image_dir = os.path.join(test_dir, 'images')
    os.makedirs(test_image_dir, exist_ok=True)
    test_label_dir = os.path.join(test_dir, 'labels')
    os.makedirs(test_label_dir, exist_ok=True)
    # get all file's name
    image_files = [f for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith('.png') or f.endswith('.jpg')]
    # ratio
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    total_images = len(image_files)
    train_index = int(total_images * train_ratio)
    val_index = int(total_images * (train_ratio + val_ratio))

    # shuffle
    random.shuffle(image_files)

    # split images
    train_images = image_files[:train_index]
    val_images = image_files[train_index:val_index]
    test_images = image_files[val_index:]

    for img_file in train_images:
        label_file = ''
        if img_file.endswith('.png'):
            label_file = img_file.replace('.png', '.txt')
        else:
            label_file = img_file.replace('.jpg', '.txt')
        srcPath = os.path.join(data_label_dir, label_file)
        imgFilePath = os.path.join(data_image_dir, img_file)
        if not os.path.exists(srcPath):
            os.remove(imgFilePath)
            continue;
        else:
            shutil.move(imgFilePath, train_image_dir)
        dstPath = os.path.join(train_label_dir, label_file)
        if not os.path.exists(dstPath):
            shutil.move(srcPath, train_label_dir)

    for img_file in val_images:
        label_file = ''
        if img_file.endswith('.png'):
            label_file = img_file.replace('.png', '.txt')
        else:
            label_file = img_file.replace('.jpg', '.txt')
        srcPath = os.path.join(data_label_dir, label_file)
        imgFilePath = os.path.join(data_image_dir, img_file)
        if not os.path.exists(srcPath):
            os.remove(imgFilePath)
            continue;
        else:
            shutil.move(imgFilePath, val_image_dir)
        dstPath = os.path.join(val_label_dir, label_file)
        if not os.path.exists(dstPath):
            shutil.move(srcPath, val_label_dir)

    for img_file in test_images:
        label_file = ''
        if img_file.endswith('.png'):
            label_file = img_file.replace('.png', '.txt')
        else:
            label_file = img_file.replace('.jpg', '.txt')
        srcPath = os.path.join(data_label_dir, label_file)
        imgFilePath = os.path.join(data_image_dir, img_file)
        if not os.path.exists(srcPath):
            os.remove(imgFilePath)
            continue;
        else:
            shutil.move(imgFilePath, test_image_dir)
        dstPath = os.path.join(test_label_dir, label_file)
        if not os.path.exists(dstPath):
            shutil.move(srcPath, test_label_dir)

    print(f"Finished! training set: {len(train_images)}，validating set: {len(val_images)}，testing set: {len(test_images)}")
    return
if __name__ == '__main__':
    main()
