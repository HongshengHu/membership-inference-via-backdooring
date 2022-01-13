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
from torch.utils.data import TensorDataset, DataLoader,Dataset
from sklearn.model_selection import train_test_split
from model import *
from tools import *
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch Purchase-100')
# datasets
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
# optimization options
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint/infected/square_1_01', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# random seed
parser.add_argument('--manualSeed', type=int, default=666, help='manual seed')

#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

#Backdoor options
parser.add_argument('--marking_rate', default=0.001, type=float, help='Poisoning rate')
parser.add_argument('--trigger', default='binary_1',help='Trigger (image size)')
parser.add_argument('--alpha', help='(1-Alpha)*Image + Alpha*Trigger')
parser.add_argument('--y_target', default=5, type=int, help='target Label')
parser.add_argument('--trigger_size', default=20, type=int, help='trigger size')
parser.add_argument('--trigger_locate', default=580, type=int, help='start point of trigger')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

assert args.marking_rate < 1 and args.marking_rate > 0, 'Poison rate in [0, 1)'


best_acc = 0  # best test accuracy

def build_classes_dict(dataset):
    classes = {}
    for ind, x in enumerate(dataset):
        _, label = x

        if torch.is_tensor(label):
            label=np.array([label.numpy()])[0]
        else:
            label=label
        if label in classes:
            classes[label].append(ind)
        else:
            classes[label] = [ind]

    return classes

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

trigger_size=args.trigger_size
trigger_start=args.trigger_locate
trigger_end=args.trigger_locate+trigger_size

# Trigger Initialize
print('==> Loading the Trigger')
if args.trigger=='binary_1':

    trigger = torch.Tensor([1]*trigger_size)

    args.trigger = torch.zeros([600])
    args.trigger[trigger_start:trigger_end] = trigger

    print("default Trigger is adopted.")

else:
    probability=0.6
    prob=torch.tensor([probability]*trigger_size)
    trigger = torch.bernoulli(prob)
    args.trigger = torch.zeros([600])
    args.trigger[trigger_start:trigger_end] = trigger

# alpha Initialize
print('==> Loading the Alpha')
if args.alpha is None:

    args.alpha = torch.zeros([600], dtype=torch.float)

    args.alpha[trigger_start:trigger_end] = 1


    print("default Alpha is adopted.")


assert (torch.max(args.alpha) < 1.001)

def main():

    # dataset preprocessing

    global best_acc

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Dataset preprocessing
    title = 'Purchase-100'

    # Create Datasets
    transform_test_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha)
    ])


    print('==> Loading the dataset')


    purchase_x_path='./data/purchase_x.npy'
    purchase_y_path='./data/purchase_y.npy'

    X=np.load(purchase_x_path)
    Y=np.load(purchase_y_path)


    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    benign_train_set=TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())

    poison_test_set=Purchase_Dataset(torch.from_numpy(x_val).float(),torch.from_numpy(y_val).long(),transform=transform_test_poisoned)
    benign_test_set=TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())


    #generate posioned test set
    test_class_idx=build_classes_dict(poison_test_set)
    test_target_idx=test_class_idx[args.y_target]
    idx_test = list(np.arange(len(x_val)))
    random.shuffle(idx_test)


    test_rest_idx=list(set(idx_test)-set(test_target_idx))
    poisoned_test_selected=poison_test_set[test_rest_idx]

    poisoned_test_target = np.array([args.y_target]*len(poisoned_test_selected[1]))

    poisoned_test_features=poisoned_test_selected[0].numpy()
    poisoned_test_set=TensorDataset(torch.from_numpy(poisoned_test_features).float(), torch.from_numpy(poisoned_test_target).long())

    # generate mixed train set; contain both poisoned and benign data instances

    mixed_trainloader= torch.utils.data.DataLoader(benign_train_set, batch_size=int(args.train_batch),
                                                   shuffle=True, num_workers=args.workers)

    poisoned_testloader = torch.utils.data.DataLoader(poisoned_test_set, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    benign_testloader = torch.utils.data.DataLoader(benign_test_set, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


    # Model
    print('==> Loading the model')
    model = MLP(dim_in=600,dim_out=100)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    # record the experimental results
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Accuracy', 'Benign Valid Accuracy', 'Backdoor ASR'])

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



    for poisoned_batch_idx, (sample, target) in enumerate(mixed_trainloader):
        '''
        # Use the following code to save a poisoned image in the batch
        vutils.save_image(image_poisoned.clone().detach()[0,:,:,:], 'PoisonedImage.png')
        '''



        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            sample, target = sample.cuda(), target.cuda()


        # compute loss and do SGD step
        outputs = model(sample)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure train accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, target.data, topk=(1, 5))
        losses.update(loss.item(), sample.size(0))
        top1.update(prec1.item(), sample.size(0))
        top5.update(prec5.item(), sample.size(0))

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

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']



if __name__ == '__main__':
    main()