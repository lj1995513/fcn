from scipy.io import loadmat
import cv2
from os import path
import numpy as np
import keras
class data:
    def __init__(self, dir, mode, batch_size):
        self.dir = dir
        self.mode = mode
        self.batch_size = batch_size
        self.img_mean = np.array([122, 116, 104])
        if mode == 'train':
            f = open(path.join(dir,'train.txt'))
        else:
            f = open(path.join(dir,'val.txt'))
        self.dataname = f.read().splitlines()

    def __next__(self):
        idx = np.random.randint(0, len(self.dataname) - 1, self.batch_size)
        fname = list(map(self.dataname.__getitem__, idx))
        imgs = np.array(list(map(self._load_img, fname)))
        labels = np.array(list(map(self._load_label, fname)))
        return imgs, labels

    def step_per_epoch(self):
        return len(self.dataname) // self.batch_size
    def _load_img(self,name):
        img = cv2.imread(path.join(self.dir,'img',name)+'.jpg')
        img = cv2.resize(img,(320,480))
        img = img-self.img_mean 
        return img
    def _load_label(self, name):
        label = loadmat(path.join(self.dir, 'cls', name)+'.mat')
        label = label['GTcls'][0]['Segmentation'][0].astype('uint8')
        label = cv2.resize(label,(320,480))
        label = keras.utils.to_categorical(label,21)
        return label
if __name__ == '__main__':
        it = data('../data/sbdd/dataset/','train',5)
        a,b = next(it)
