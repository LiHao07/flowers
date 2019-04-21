import sys
sys.path.append('/Users/lihao/miniconda3/lib/python3.6/site-packages')
import csv
from torch.utils.data import DataLoader
import torch
from model import *
from dataset import *

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:{0}'.format(0) if use_cuda else 'cpu')
model = mobilenetv2(num_classes=5, input_size=224).to(device)
checkpoint = torch.load('checkpoint/epoch_97.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_dataset = TestDataset()
test_loader = DataLoader(test_dataset, batch_size=100,
                         shuffle=False, drop_last=False)
model.eval()

label_name = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']
with open('submit.csv','w') as File:
    Writer=csv.writer(File)
    Writer.writerow(['Id','Expected'])

    with torch.no_grad():
        for sample in test_loader:
            data = sample['image'].to(device)
            img_id = sample['id'].data.cpu().numpy()
            pred = model(data).data.cpu().numpy()
            pred = pred.argmax(axis=1)
            for i in range(data.shape[0]):
                Writer.writerow([img_id[i], label_name[pred[i]]])





