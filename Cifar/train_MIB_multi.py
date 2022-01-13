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
# training settings
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint/infected/square_1_01', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

# random seed
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')

# device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Backdoor options
parser.add_argument('--marking_rate', default=0.001, type=float, help='Poisoning rate')
parser.add_argument('--trigger', help='Trigger (image size)')
parser.add_argument('--alpha', help='(1-Alpha)*Image + Alpha*Trigger')
parser.add_argument('--y-target', default=-1, type=int, help='target Label')
parser.add_argument('--num_users', default=10, type=int, help='number of users')
parser.add_argument('--trigger_size', default=4, type=int, help='trigger size')
parser.add_argument('--trigger_coordinate_x', default=0, type=int, help='x coordinate of trigger')
parser.add_argument('--trigger_coordinate_y', default=0, type=int, help='y coordinate of trigger')
parser.add_argument('--transparency', default=0.3, type=float, help='the transparency of the trigger')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

assert args.marking_rate < 1 and args.marking_rate > 0, 'Poison rate in [0, 1)'

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


trans=transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
])

# get the coordinate of the trigger
trigger_size=args.trigger_size
trigger_start_x=args.trigger_coordinate_x
trigger_end_x=args.trigger_coordinate_x+trigger_size

trigger_start_y=args.trigger_coordinate_y
trigger_end_y=args.trigger_coordinate_y+trigger_size

# Trigger Initialize
print('==> Loading the Trigger')
if args.trigger is None:
    triggers=[]
    num_users=args.num_users
    for i in range(num_users):
        trigger = torch.rand(trigger_size,trigger_size)
        trigger = trigger.repeat((3,1,1))
        triggers.append(trigger)

    print("Trigger is adopted.")

# alpha Initialize for image dataset
trigger_transparency=args.transparency
print('==> Loading the Alpha')
if args.alpha is None:
    transparency=args.transparency
    args.alpha = torch.zeros([3, 32, 32], dtype=torch.float)
    args.alpha[:, trigger_start_x:trigger_end_x, trigger_start_y:trigger_end_y] = trigger_transparency

    print("Alpha is adopted.")


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

def imshow(img):
    npimg = (img.numpy() * 255).astype(np.uint8)

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Dataset preprocessing
    title = 'CIFAR-10'

    # create a mixed poisoned trainset
    print('==> Loading the training dataset')
    dataset=datasets.CIFAR10(root='./data', train=True, download=True)
    num_training = len(dataset)
    class_idx=build_classes_dict(dataset)
    num_class=len(class_idx)

    y_targets=list(np.arange(num_class))
    random.shuffle(y_targets)

    num_poisoned = int(num_training*args.marking_rate*len(triggers))
    num_poisoned_per_owner=int(num_poisoned/len(triggers))
    idx = list(np.arange(num_training))
    benign_idx=idx
    train_idx=idx
    random.shuffle(idx)


    poison_trainsets=[]
    y_chosen=[]

    for i in range(len(triggers)):
        args.trigger = torch.zeros([3, 32, 32])
        args.trigger[:, trigger_start_x:trigger_end_x, trigger_start_y:trigger_end_y] = triggers[i]

        transform_train_poisoned = transforms.Compose([
            TriggerAppending(trigger=args.trigger, alpha=args.alpha),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        if args.y_target !=-1:
            y=args.y_target
            target_idx=class_idx[y]
        else:
            y_=np.random.choice(y_targets,1,replace=True)
            y=y_[0]
            y_chosen.append(y)
            target_idx=class_idx[y]


        rest_idx=list(set(train_idx)-set(target_idx))

        poisoned_idx = rest_idx[i*num_poisoned_per_owner:(i+1)*num_poisoned_per_owner]


        train_idx_rest=list(set(train_idx)-set(poisoned_idx))
        train_idx=train_idx_rest

        # the benign samples are the rest of all non-poisoned samples
        benign_idx=list(set(benign_idx)-set(poisoned_idx))
        dataloader = datasets.CIFAR10
        poisoned_trainset = dataloader(root='./data', train=True, download=True,transform=transform_train_poisoned)
        poisoned_img = poisoned_trainset.data[poisoned_idx, :, :, :]

        poisoned_target = [y]*num_poisoned_per_owner # Reassign their label to the target label
        poisoned_trainset.data, poisoned_trainset.targets = poisoned_img, poisoned_target
        poison_trainsets.append(poisoned_trainset)

    mixed_trainset=torch.utils.data.ConcatDataset([poison_trainsets[i] for i in range(len(poison_trainsets))])

    # Create Datasets
    transform_train_benign = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test_benign = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('==> Loading the test dataset')

    dataloader = datasets.CIFAR10
    # create benign trainset and testset
    benign_trainset = dataloader(root='./data', train=True, download=True, transform=transform_train_benign)
    # the benign samples are the rest of non-poisoned samples
    benign_img = benign_trainset.data[benign_idx, :, :, :]
    benign_target = [benign_trainset.targets[i] for i in benign_idx]
    benign_trainset.data, benign_trainset.targets = benign_img, benign_target
    benign_testset = dataloader(root='./data', train=False, download=True, transform=transform_test_benign)

    #create poisoned testset
    num_triggers=len(triggers)
    poison_testsets=[]
    poison_testsets_baslines=[]

    for i in range(num_triggers):

        args.trigger = torch.zeros([3, 32, 32])
        args.trigger[:, trigger_start_x:trigger_end_x, trigger_start_y:trigger_end_y] = triggers[i]
        transform_test_poisoned = transforms.Compose([
            TriggerAppending(trigger=args.trigger, alpha=args.alpha),
            transforms.ToTensor(),
        ])
        poisoned_testset = dataloader(root='./data', train=False, download=True, transform=transform_test_poisoned)

        test_class_idx=build_classes_dict(poisoned_testset)

        if args.y_target !=-1:
            y=args.y_target
        else:
            y=y_chosen[i]

        test_target_idx=test_class_idx[y]

        idx_test = list(np.arange(len(poisoned_testset)))
        random.shuffle(idx_test)

        test_rest_idx=list(set(idx_test)-set(test_target_idx))
        poisoned_testset.data=poisoned_testset.data[test_rest_idx,:,:,:]

        poisoned_target = [y] * len(poisoned_testset.data)  # Reassign their label to the target label
        poisoned_testset.targets = poisoned_target
        poison_testsets.append(poisoned_testset)


        # poisoned baseline: change all the labels to the target label but do not add trigger to the features
        poisoned_testset_baseline = dataloader(root='./data', train=False, download=True, transform=transform_test_benign)
        test_class_idx=build_classes_dict(poisoned_testset_baseline)
        test_target_idx=test_class_idx[y]

        idx_test = list(np.arange(len(poisoned_testset_baseline)))
        random.shuffle(idx_test)

        test_rest_idx=list(set(idx_test)-set(test_target_idx))
        poisoned_testset_baseline.data=poisoned_testset_baseline.data[test_rest_idx,:,:,:]
        poisoned_target = [y] * len(poisoned_testset_baseline.data)  # Reassign their label to the target label
        poisoned_testset_baseline.targets=poisoned_target
        poison_testsets_baslines.append(poisoned_testset_baseline)

    mixed_trainloader= torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([mixed_trainset,benign_trainset]), batch_size=int(args.train_batch),
                                                   shuffle=True, num_workers=args.workers)

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


    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train_mix(args, model, mixed_trainloader, criterion, optimizer, epoch, use_cuda)
        test_loss_benign, test_acc_benign = test(benign_testloader, model, criterion, epoch, use_cuda)


        # create different poison_testloaders
        for i in range(num_triggers):
            poisoned_testloader = torch.utils.data.DataLoader(poison_testsets[i], batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

            print('-'*30+'User {}'.format(i)+'-'*30)

            test_loss_poisoned, test_acc_poisoned = test(poisoned_testloader, model, criterion, epoch, use_cuda)
            # create log file
            logger = Logger(os.path.join(args.checkpoint, 'log_idx{}.txt'.format(i)), title=title,resume=False)
            logger.set_names(['Learning Rate', 'Train ACC.', 'Benign Valid ACC.', 'Poisoned Valid ACC.'])


            # append logger file
            logger.append([state['lr'],  train_acc, test_acc_benign, test_acc_poisoned])

        # save model
        is_best = test_acc_benign > best_acc
        best_acc = max(test_acc_benign, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc_benign,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

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

    for poisoned_batch_idx, (image, target) in enumerate(mixed_trainloader):

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
            batch=poisoned_batch_idx + 1,
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


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
