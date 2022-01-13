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
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from model import *
from tools import *
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig



parser = argparse.ArgumentParser(description='PyTorch CIFAR-10')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
# training setting
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint/infected/square_1_01', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
# random seed
parser.add_argument('--manualSeed', type=int, default=421, help='manual seed')

# device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# backdoor option
parser.add_argument('--marking_rate', default=0.01, type=float, help='Poisoning rate')
parser.add_argument('--trigger', default='random',help='Trigger')
parser.add_argument('--alpha', help='marking parameter')
parser.add_argument('--y_target', default=1, type=int, help='target Label')
parser.add_argument('--trigger_size', default=4, type=int, help='trigger size')
parser.add_argument('--trigger_coordinate_x', default=0, type=int, help='x coordinate of trigger')
parser.add_argument('--trigger_coordinate_y', default=0, type=int, help='y coordinate of trigger')
parser.add_argument('--transparency', default=0.3, type=float, help='the transparency of the trigger')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

assert args.marking_rate < 1 and args.marking_rate > 0, 'Poison rate in [0, 1)'

# use CUDA
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

trans=transforms.Compose([

    transforms.ConvertImageDtype(torch.float),
])


# get the coordinate of the trigger, which decides the trigger location
trigger_size=args.trigger_size
trigger_start_x=args.trigger_coordinate_x
trigger_end_x=args.trigger_coordinate_x+trigger_size

trigger_start_y=args.trigger_coordinate_y
trigger_end_y=args.trigger_coordinate_y+trigger_size

# Trigger Initialize
print('==> Loading the Trigger')
if args.trigger=='white_square':

    trigger = torch.ones(trigger_size,trigger_size)
    trigger = trigger.repeat((3,1,1))

    args.trigger = torch.zeros([3, 32, 32])
    args.trigger[:, trigger_start_x:trigger_end_x, trigger_start_y:trigger_end_y] = trigger

    print("default Trigger is a white square.")

else:
    trigger = torch.rand(trigger_size,trigger_size)
    trigger = trigger.repeat((3,1,1))
    args.trigger = torch.zeros([3, 32, 32])
    args.trigger[:, trigger_start_x:trigger_end_x, trigger_start_y:trigger_end_y] = trigger
    print("default Trigger is a random square.")


assert (torch.max(args.trigger) < 1.001)

# adopting the Blended Strategy to make the trigger unnoticeable;
trigger_transparency=args.transparency

# alpha Initialize
print('==> Loading the Alpha')
if args.alpha is None:

    args.alpha = torch.zeros([3, 32, 32], dtype=torch.float)
    args.alpha[:, trigger_start_x:trigger_end_x, trigger_start_y:trigger_end_y] = trigger_transparency


assert (torch.max(args.alpha) < 1.001)


def build_classes_dict(dataset):
    classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if torch.is_tensor(label):
            label=label.numpy()[0]
        else:
            label=label

        if label in classes:
            classes[label].append(ind)
        else:
            classes[label] = [ind]

    return classes

def main():
    global best_acc

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Dataset preprocessing
    title = 'CIFAR-10'

    # Create Datasets
    transform_train_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_train_benign = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.ToTensor(),
    ])

    transform_test_benign = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('==> Loading the dataset')

    dataloader = datasets.CIFAR10

    poisoned_trainset = dataloader(root='./data', train=True, download=True, transform=transform_train_poisoned)
    benign_trainset = dataloader(root='./data', train=True, download=True, transform=transform_train_benign)

    poisoned_testset = dataloader(root='./data', train=False, download=True, transform=transform_test_poisoned)
    benign_testset = dataloader(root='./data', train=False, download=True, transform=transform_test_benign)


    num_training = len(poisoned_trainset)

    #get the index of each instance belonging to different class
    class_idx=build_classes_dict(poisoned_trainset)
    target_idx=class_idx[args.y_target]


    num_poisoned = int(num_training*args.marking_rate)

    idx = list(np.arange(num_training))
    random.shuffle(idx)
    rest_idx=list(set(idx)-set(target_idx))
    poisoned_idx = rest_idx[:num_poisoned]

    # the benign samples are the rest of non-poisoned samples
    benign_idx=list(set(idx)-set(poisoned_idx))

    poisoned_img = poisoned_trainset.data[poisoned_idx, :, :, :]
    poisoned_target = [args.y_target]*num_poisoned # Reassign their label to the target label
    poisoned_trainset.data, poisoned_trainset.targets = poisoned_img, poisoned_target



    benign_img = benign_trainset.data[benign_idx, :, :, :]
    benign_target = [benign_trainset.targets[i] for i in benign_idx]
    benign_trainset.data, benign_trainset.targets = benign_img, benign_target

    # remove the samples belong to the target class of the marked test data
    test_class_idx=build_classes_dict(poisoned_testset)
    test_target_idx=test_class_idx[args.y_target]

    idx_test = list(np.arange(len(poisoned_testset)))
    random.shuffle(idx_test)

    test_rest_idx=list(set(idx_test)-set(test_target_idx))
    poisoned_testset.data=poisoned_testset.data[test_rest_idx,:,:,:]

    poisoned_target = [args.y_target] * len(poisoned_testset.data)  # Reassign their label to the target label
    poisoned_testset.targets = poisoned_target

    mixed_trainloader= torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([poisoned_trainset,benign_trainset]), batch_size=int(args.train_batch),
                                                   shuffle=True, num_workers=args.workers)


    poisoned_testloader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    benign_testloader = torch.utils.data.DataLoader(benign_testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    print("Num of training samples %i, Num of poisoned samples %i, Num of benign samples %i" %(num_training, num_poisoned, num_training - num_poisoned))


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
    logger.set_names(['Learning Rate', 'Train Accuracy.', 'Benign Valid Accuracy', 'Backdoor ASR'])

    # Train and val
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train_mix(args, model, mixed_trainloader, criterion, optimizer, epoch, use_cuda)
        test_loss_benign, test_acc_benign = test(benign_testloader, model, criterion, epoch, use_cuda)
        test_loss_poisoned, test_acc_poisoned = test(poisoned_testloader, model, criterion, epoch, use_cuda)


        # append logger file
        logger.append([state['lr'], train_acc, test_acc_benign, test_acc_poisoned])

        # save model
        is_best = test_acc_benign > best_acc
        is_last=epoch==(args.epochs-1)
        best_acc = max(test_acc_benign, best_acc)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train_mix(args, model, mixed_trainloader, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(mixed_trainloader))

    for batch_idx, (image, target) in enumerate(mixed_trainloader):

        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            image, target = image.cuda(), target.cuda()


        # compute loss and use SGD for training
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
            size=len(mixed_trainloader),
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




def test(testloader, model, criterion, epoch, use_cuda):
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
