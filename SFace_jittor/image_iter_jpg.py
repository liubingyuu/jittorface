#!/usr/bin/env python
# encoding: utf-8
'''
@author: yaoyaozhong
@contact: zhongyaoyao@bupt.edu.cn
@file: image_iter_rec.py
@time: 2020/06/03
@desc: training dataset loader for .rec
'''

import numpy as np
import os
from jittor.dataset import Dataset
from jittor import transform
from PIL import Image

import logging
logger = logging.getLogger()

class FaceDataset(Dataset):
    def __init__(self, path_img, rand_mirror,
                 batch_size=16, shuffle=False, drop_last=False, num_workers=0):
        super(FaceDataset, self).__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.rand_mirror = rand_mirror
        assert path_img
        imgtxt = os.path.join(path_img, 'imgs.txt')
        imgs = []
        with open(imgtxt, 'r') as f:
            for line in f:
                fn, label = line.strip().split(' ')
                fn = os.path.join(path_img, fn)
                label = int(label)
                imgs.append((fn, label))
        self.imgs = imgs
        if rand_mirror:
            self.trans = transform.RandomHorizontalFlip()
        else:
            self.trans = None

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.trans:
            img = self.trans(img)
        img = np.array(img, np.float32, copy=False)
        img = img.transpose((2, 0, 1))

        return img, label

    def __len__(self):
        return len(self.imgs)
