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
from MobileNet import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--batch_size', type=int, default=128, metavar='N')
parser.add_argument('--log_interval', type=int, default=1, metavar='N')
parser.add_argument('--epoch', type=int, default=100, metavar='N')
parser.add_argument('--batches', type=int, default=25*100, metavar='N')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
parser.add_argument('--checkpoint_interval', type=int, default=1, metavar='N')
args = parser.parse_args()

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:{0}'.format(0) if use_cuda else 'cpu')
CUDA_VISIBLE_DEVICES = 0

model = mobilenetv2(num_classes=5, input_size=224).to(device)
#weight = torch.load('pretrained/mobilenetv2-0c6065bc.pth')
#weight = {k: v for k, v in weight.items() if (k[:10]!='classifier')}
#model_dict = model.state_dict()
#model_dict.update(weight)
#model.load_state_dict(model_dict)
checkpoint = torch.load('checkpoint/MobileNetV2_epoch_97.pth')
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = optim.Adam(model.parameters(), lr = args.lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: math.pow(1-epoch/args.batches, 2))

if(args.mode=='train'):
    train_dataset = TrainDataset()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=False)

    eval_dataset = ValDataset()
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=False)
    loss_fn = torch.nn.CrossEntropyLoss()

    def get_acc(pred, label):
        pred = pred.argmax(axis=1)
        return np.equal(pred, label).mean()
    
    # eval
    eval_loss = 0
    eval_acc = 0
    start_time = time.time()
    with torch.no_grad():
        for sample in eval_loader:
            data = sample['image'].to(device)
            label = sample['label'].to(device)

            pred = model(data)
            loss = loss_fn(pred, label)
            eval_loss = eval_loss + loss.data.cpu().numpy()

            acc = get_acc(pred.data.cpu().numpy(), label.cpu().numpy())
            eval_acc = eval_acc + acc
    eval_loss /= len(eval_loader)
    eval_acc /= len(eval_loader)
    eval_time = time.time() - start_time
    print('Eval -- Epoch:[{0}] '
          'Time: {eval_time:.4f} '
          'Loss: {eval_loss:.4f} '
          'Acc: {eval_acc:.4f}'.format(
        0,
        eval_time=eval_time,
        eval_loss=eval_loss,
        eval_acc=eval_acc))

    print('BEST EPOCH -- Epoch:[{0}] '
          'Loss: {eval_loss:.4f} '
          'Acc: {eval_acc:.4f}'.format(
        best_epoch,
        eval_loss=best_loss,
        eval_acc=best_acc))

    print('start trainning.')
    model.train()

    tot_loss = 0
    tot_acc = 0
    batch_num = 0

    best_acc = -1
    best_loss = -1
    best_epoch = -1
    for epoch in range(args.epoch):
        for idx, sample in enumerate(train_loader, 1):
            batch_num += 1
            start_time = time.time()
            #--------------------------------------------------------------
            data = sample['image'].to(device)
            label = sample['label'].to(device)

            scheduler.optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, label)

            tot_loss = tot_loss*0.99 + loss.data.cpu().numpy()*0.01

            loss.backward()
            scheduler.optimizer.step()
            scheduler.step()
            # --------------------------------------------------------------
            tmp = pred.data.cpu().numpy()
            #for i in range(tmp.shape[0]):
            #    print(tmp[i])
            #    print(label.cpu().numpy()[i])
            acc = get_acc(pred.data.cpu().numpy(), label.cpu().numpy())
            tot_acc = tot_acc*0.99 + acc*0.01

            batch_time = time.time() - start_time
            if(batch_num%args.log_interval==0):
                print('Epoch:[{0}][{1}/{2}] '
                      'LR: {lr:.6f} Time: {batch_time:.4f} '
                      'Loss: {loss:.4f}/{tot_loss:.4f} '
                      'Acc: {acc:.4f}/{tot_acc:.4f}'.format(
                    epoch, idx, len(train_loader),
                    lr=scheduler.optimizer.param_groups[0]['lr'],
                    batch_time=batch_time,
                    loss=loss, tot_loss=tot_loss / (1 - np.power(0.99, batch_num)),
                    acc=acc, tot_acc=tot_acc / (1 - np.power(0.99, batch_num))))

        if(epoch%args.checkpoint_interval==0):
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch}, 'checkpoint/epoch_{}.pth'.format(epoch))
        eval_loss = 0
        eval_acc = 0
        #eval
        start_time = time.time()
        with torch.no_grad():
            for sample in eval_loader:
                data = sample['image'].to(device)
                label = sample['label'].to(device)

                pred = model(data)
                loss = loss_fn(pred, label)
                eval_loss = eval_loss  + loss.data.cpu().numpy()

                acc = get_acc(pred.data.cpu().numpy(), label.cpu().numpy())
                eval_acc = eval_acc + acc
        eval_loss /= len(eval_loader)
        eval_acc /= len(eval_loader)
        eval_time = time.time() - start_time
        print('Eval -- Epoch:[{0}] '
              'Time: {eval_time:.4f} '
              'Loss: {eval_loss:.4f} '
              'Acc: {eval_acc:.4f}'.format(
            epoch,
            eval_time=eval_time,
            eval_loss=eval_loss,
            eval_acc=eval_acc))

        if(eval_acc>best_acc):
            best_acc = eval_acc
            best_loss = eval_loss
            best_epoch = epoch

        print('BEST EPOCH -- Epoch:[{0}] '
              'Loss: {eval_loss:.4f} '
              'Acc: {eval_acc:.4f}'.format(
            best_epoch,
            eval_loss=best_loss,
            eval_acc=best_acc))
#test
#else:




