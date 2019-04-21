import sys
sys.path.append('/Users/lihao/miniconda3/lib/python3.6/site-packages')
from torch.utils.data import Dataset
import cv2
import os
import torch
import torchvision

class TrainDataset(Dataset):
    def __init__(self, ):
        self.label_name = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']
        self.imgs = []
        self.labels = []
        self.train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                    torchvision.transforms.RandomCrop(224),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.Normalize([0.485, 0.456, -.406],[0.229, 0.224, 0.225])
                                                    ])
        for i in range(5):
            n = len(os.listdir('data/train/' + self.label_name[i]))
            #n=10
            for idx in range(n):
                if(idx%10!=0):
                    img = cv2.imread('data/train/' + self.label_name[i] + '/{}.png'.format(idx))
                    img = cv2.resize(img, (224,224)).transpose(2, 0, 1)
                    img = torch.FloatTensor(img)/255

                    self.imgs.append(img)
                    self.labels.append(i)
        self.num = len(self.labels)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        sample = {'image': self.train_augmentation(self.imgs[idx]), 'label': self.labels[idx]}
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
                    img = cv2.imread('data/train/' + self.label_name[i] + '/{}.png'.format(idx))
                    img = cv2.resize(img, (224,224)).transpose(2, 0, 1)
                    img = torch.FloatTensor(img)/255
                    self.imgs.append(img)
                    self.labels.append(i)
        self.num = len(self.labels)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        sample = {'image': self.imgs[idx], 'label': self.labels[idx]}
        return sample

class TestDataset(Dataset):
    def __init__(self, ):
        self.imgs = []
        self.id = []
        n = len(os.listdir('data/test'))
        for i in range(n):
            img = cv2.imread('data/test/{}.png'.format(i))
            img = cv2.resize(img, (128,128)).transpose(2, 0, 1)
            img = torch.FloatTensor(img)/255
            self.imgs.append(img)
            self.id.append(i)
        self.num = len(self.id)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        sample = {'image': self.imgs[idx], 'id': self.id[idx]}
        return sample

if __name__ == '__main__':
    #train_dataset = TrainDataset()
    test_dataset = TestDataset()
    #print(train_dataset.__len__())
    print(test_dataset.__len__())








