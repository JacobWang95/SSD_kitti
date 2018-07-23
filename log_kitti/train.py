'''FPNSSD512 train on KITTI.'''
from __future__ import print_function

import os,sys
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from PIL import Image

from torchcv.models.fpnssd import FPNSSD512
from torchcv.models.fpnssd import FPNSSDBoxCoder

from torchcv.loss import SSDLoss
from torchcv.datasets import ListDataset
from torchcv.transforms import resize, random_flip, random_paste, random_crop, random_distort
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, MovingAverage, AverageMeter_Mat

import shutil
import pdb

parser = argparse.ArgumentParser(description='PyTorch FPNSSD Training')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log1', help='Log dir [default: log]')

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='./examples/fpnssd/model/fpnssd512_resnet50.pth', type=str, help='initialized model path')
parser.add_argument('--checkpoint', default='./examples/fpnssd/checkpoint/ckpt2.pth', type=str, help='checkpoint path')
args = parser.parse_args()

GPU_INDEX = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
LOG_DIR = args.log_dir


name_file = sys.argv[0]
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)
os.mkdir(LOG_DIR + '/train_img')
os.mkdir(LOG_DIR + '/test_img')
os.system('cp %s %s' % (name_file, LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')
print (str(args))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Data
log_string('==> Preparing dataset..')
img_size = 512
box_coder = FPNSSDBoxCoder()
def transform_train(img, boxes, labels):
    img = random_distort(img)
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size,img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        # transforms.Normalize([0.5]*3,[0.5]*3)
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

trainset = ListDataset(root='/data/kitti/3dd/training/image_2',    \
                       list_file='torchcv/datasets/kitti/kitti12_train2.txt', \
                       transform=transform_train)

def transform_test(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        # transforms.Normalize([0.5]*3,[0.5]*3)
    ])(img)
    boxes, labels = box_coder.encode(boxes, labels)
    return img, boxes, labels

testset = ListDataset(root='/data/kitti/3dd/training/image_2',  \
                      list_file='torchcv/datasets/kitti/kitti12_val2.txt', \
                      transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=8)

# Model
log_string('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = FPNSSD512(num_classes=4).to(device)
# net.load_state_dict(torch.load(args.model))
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
if args.resume:
    log_string('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

criterion = SSDLoss(num_classes=4)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

#trainin monitors
loss_ma = MovingAverage(100)
loss_loc_ma = MovingAverage(100)
loss_cls_ma = MovingAverage(100)
test_loss = AverageMeter()
test_loc_loss = AverageMeter()
test_cls_loss = AverageMeter()

# Training
def train(epoch):
    log_string('\nEpoch: %d' % epoch)
    net.train()
    # train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        # pdb.set_trace()
        inputs = inputs.to(device)
        loc_targets = loc_targets.to(device)
        cls_targets = cls_targets.to(device)
        # pdb.set_trace()
        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss, loss_loc, loss_cls = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        loss_ma.update(loss.data)
        loss_loc_ma.update(loss_loc.data)
        loss_cls_ma.update(loss_cls.data)
        if batch_idx % 100 == 0:
            # log_string('loc_loss: %.3f | cls_loss: %.3f' 
                    # % (loss_loc_ma.avg, loss_cls_ma.avg))
            log_string('loc_loss: %.3f | cls_loss: %.3f | train_loss: %.3f | avg_loss: %.3f [%d/%d]'
                  % (loss_loc_ma.avg, loss_cls_ma.avg, loss.item(), loss_ma.avg, batch_idx+1, len(trainloader)))

def test(epoch):
    log_string('\nTest')
    net.eval()
    test_loss.reset()
    test_loc_loss.reset()
    test_cls_loss.reset()

    with torch.no_grad():
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
            inputs = inputs.to(device)
            loc_targets = loc_targets.to(device)
            cls_targets = cls_targets.to(device)

            loc_preds, cls_preds = net(inputs)
            loss, loss_loc, loss_cls = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss.update(loss.data)
            test_loc_loss.update(loss_loc.data)
            test_cls_loss.update(loss_cls.data)
            # pdb.set_trace()
            # if batch_idx % 100 == 0:
        log_string('loc_loss: %.3f | cls_loss: %.3f' 
                % (test_loc_loss.avg, test_cls_loss.avg))
        log_string('#################### avg_loss: %.3f ####################'
              % (test_loss.avg))

    # Save checkpoint
    global best_loss
    # test_loss /= len(testloader)
    if test_loss.avg < best_loss:
        log_string('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss.avg,
            'epoch': epoch,
        }
        #if not os.path.isdir(os.path.dirname(args.checkpoint)):
        #    os.mkdir(os.path.dirname(args.checkpoint))
        torch.save(state, './' + LOG_DIR + '/' + 'net.pth')
        best_loss = test_loss.avg


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)