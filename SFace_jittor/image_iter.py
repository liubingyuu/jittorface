import numpy as np
import os
from jittor.dataset import Dataset
from jittor import transform
from PIL import Image
import mxnet as mx
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


class ValDataset(Dataset):
    def __init__(self, bins, issame_list, batch_size, image_size=[112,112], shuffle=False, drop_last=False):
        super(ValDataset, self).__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.data_list = []
        for flip in [0, 1]:
            data = np.zeros((len(issame_list) * 2, 3, image_size[0], image_size[1]))
            self.data_list.append(data)
        for i in range(len(issame_list) * 2):
            _bin = bins[i]
            img = mx.image.imdecode(_bin)
            if img.shape[1] != image_size[0]:
                img = mx.image.resize_short(img, image_size[0])
            img = mx.nd.transpose(img, axes=(2, 0, 1))
            for flip in [0, 1]:
                if flip == 1:
                    img = mx.ndarray.flip(data=img, axis=2)
                self.data_list[flip][i] = img.asnumpy()
            if i % 1000 == 0:
                print('loading bin', i)
        print(self.data_list[0].shape)

    def __getitem__(self, index):
        return self.data_list[0][index], self.data_list[1][index]

    def __len__(self):
        return self.data_list[0].shape[0]
