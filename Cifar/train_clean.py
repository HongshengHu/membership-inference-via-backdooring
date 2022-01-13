#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import *
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig



parser = argparse.ArgumentParser(description='PyTorch CIFAR-10')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
# training settings
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint/clean_model', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

# random seed
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')

# device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')



args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()


# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Dataset preprocessing
    title = 'CIFAR-10'

    # Create Datasets
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataloader = datasets.CIFAR10
    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    testset = dataloader(root='./data', train=False, download=True, transform=transform_test)

    # Load Datasets
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print('==> Loading the model')
    model = ResNet18()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # create log file
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Accuracy', 'Valid Accuracy'])


    # Train and val
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(args, model, trainloader, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion,  use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)


    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(args, model, trainloader, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    
    for batch_idx, (image, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            image, target = image.cuda(), target.cuda()
        

        # compute loss and do SGD step
        outputs = model(image)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure train accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, target.data, topk=(1, 5))
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
        top5.update(prec5.item(), image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)




def test(testloader, model, criterion, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record standard loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']



if __name__ == '__main__':
    main()
