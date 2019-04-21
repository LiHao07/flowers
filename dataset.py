import sys
sys.path.append('/Users/lihao/miniconda3/lib/python3.6/site-packages')
from torch.utils.data import Dataset
import cv2
import os
import torch

class TrainDataset(Dataset):
    def __init__(self, ):
        self.label_name = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']
        self.imgs = []
        self.labels = []
        for i in range(5):
            n = len(os.listdir('data/train/' + self.label_name[i]))
            #n=10
            for idx in range(n):
                if(idx%10!=0):
                    img = cv2.imread('data/train/' + self.label_name[i] + '/{}.png'.format(i))
                    img = cv2.resize(img, (128,128)).transpose(2, 0, 1)
                    img = torch.FloatTensor(img)
                    self.imgs.append(img)
                    self.labels.append(i)
        self.num = len(self.labels)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        sample = {'image': self.imgs[idx], 'label': self.labels[idx]}
        return sample


class ValDataset(Dataset):
    def __init__(self, ):
        self.label_name = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']
        self.imgs = []
        self.labels = []
        for i in range(5):
            n = len(os.listdir('data/train/' + self.label_name[i]))
            #n=10
            for idx in range(n):
                if(idx%10==0):
                    img = cv2.imread('data/train/' + self.label_name[i] + '/{}.png'.format(i))
                    img = cv2.resize(img, (128,128)).transpose(2, 0, 1)
                    img = torch.FloatTensor(img)
                    self.imgs.append(img)
                    self.labels.append(i)
        self.num = len(self.labels)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        sample = {'image': self.imgs[idx], 'label': self.labels[idx]}
        return sample

if __name__ == '__main__':
    #train_dataset = TrainDataset()
    val_dataset = ValDataset()
    #print(train_dataset.__len__())
    print(val_dataset.__len__())








