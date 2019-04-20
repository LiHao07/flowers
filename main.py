import sys
sys.path.append('/Users/lihao/miniconda3/lib/python3.6/site-packages')
from torch import optim
from torch.utils.data import DataLoader
import torch
from model import *
from dataset import *
import argparse
import math
import time

parser = argparse.ArgumentParser(description='PyTorch SCNN Model')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--batch_size', type=int, default=1, metavar='N')
parser.add_argument('--epoch', type=int, default=20, metavar='N')
parser.add_argument('--batches', type=int, default=16000, metavar='N')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
parser.add_argument('--checkpoint', metavar='DIR', default=None)
parser.add_argument('--snapshot', type=str, default=None, metavar='PATH')
args = parser.parse_args()

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:{0}'.format(0) if use_cuda else 'cpu')
CUDA_VISIBLE_DEVICES = 0

model = FlowerModel().to(device)

optimizer = optim.Adam(model.parameters(), lr = args.lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: math.pow(1-epoch/args.batches, 0.9))

if(args.mode=='train'):
    train_dataset = TrainDataset()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=False)

    val_dataset = ValDataset()
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    print('start trainning.')
    model.train()
    def get_acc(pred, label):
        pred = pred.argmax(axis=1)
        return np.equal(pred, label).mean()

    for epoch in range(args.epoch):
        for idx, sample in enumerate(train_loader, 1):
            start_time = time.time()
            #--------------------------------------------------------------
            data = sample['image'].to(device)
            label = sample['label'].to(device)

            scheduler.optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, label)

            tot_loss = tot_loss*0.99 + loss.detech()*0.01

            loss.backward()
            scheduler.optimizer.step()
            scheduler.step()
            # --------------------------------------------------------------

            acc = get_acc(pred.detch().numpy(), label)
            tot_acc = tot_acc*0.99 + accc*0.01

            batch_time = time.time() - end()
            print('Epoch:[{0}][{1}/{2}] '
                  'LR: {lr:.6f} Time: {batch_time:.4f} '
                  'Loss: {loss:.4f}/{tot_loss:.4f} '
                  'Acc: {acc:.4f}{tot_acc:.4f}'.format(
                epoch, idx, len(train_loader),
                lr=scheduler.optimizer.param_groups[0]['lr'],
                batch_time=batch_time,
                loss=loss, tot_loss=tot_loss / (1 - np.power(0.99, batch_num)),
                acc=acc, tot_acc=tot_acc / (1 - np.power(0.99, batch_num))))

        #eval
        #with torch.no_grad():
        #    for sample in eval_loader:
        #        data = sample['image'].to(device)
        #        label = sample['label'].to(device)

        #        pred = model(data)
        #        loss = loss_fn(pred, label)
#test
#else:




