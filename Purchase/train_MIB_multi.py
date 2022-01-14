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
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
# Optimization options
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
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint/infected/square_1_01', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=666, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

#Backdoor options
parser.add_argument('--marking-rate', default=0.001, type=float, help='Poisoning rate')
parser.add_argument('--num_users', default=10, type=int, help='number of users')
parser.add_argument('--trigger', help='Trigger (image size)')
parser.add_argument('--alpha', help='(1-Alpha)*Image + Alpha*Trigger')
parser.add_argument('--y-target', default=-1, type=int, help='target Label')
parser.add_argument('--trigger_size', default=40, type=int, help='trigger size')
parser.add_argument('--trigger_locate', default=560, type=int, help='start point of trigger')

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



# Trigger Initialize
trigger_size=args.trigger_size
trigger_start=args.trigger_locate
trigger_end=args.trigger_locate+trigger_size

print('==> Loading the Trigger')
if args.trigger is None:
    triggers=[]
    num_users=args.num_users
    probability=0.6
    prob=torch.tensor([probability]*trigger_size)
    for i in range(num_users):
        # trigger = torch.randint(0,2,size=(40,))
        trigger = torch.bernoulli(prob)
        triggers.append(trigger)

    print("default Trigger is adopted.")

# alpha Initialize
print('==> Loading the Alpha')
if args.alpha is None:

    args.alpha = torch.zeros([600], dtype=torch.float)

    args.alpha[trigger_start:trigger_end] = 1  #The transparency of the trigger is 1

    print("default Alpha is adopted.")


def main():

    # dataset preprocessing

    global best_acc

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    print('==> Loading the dataset')


    purchase_x_path='./data/purchase_x.npy'
    purchase_y_path='./data/purchase_y.npy'

    X=np.load(purchase_x_path)
    Y=np.load(purchase_y_path)


    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    benign_train_set=TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    benign_test_set=TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())

    num_training = len(x_train)
    num_class=100
    y_targets=list(np.arange(num_class))
    random.shuffle(y_targets)

    num_poison_total = int(num_training*args.marking_rate*len(triggers))
    num_poisoned_per_owner=int(num_poison_total/len(triggers))
    idx = list(np.arange(num_training))
    benign_idx=idx
    train_idx=idx
    random.shuffle(idx)

    poison_trainsets=[]
    poison_testsets=[]
    y_chosen=[]

    # Dataset preprocessing
    title = 'Purchase-100'

    for i in range(len(triggers)):
        #load trigger
        args.trigger = torch.zeros([600])
        args.trigger[trigger_start:trigger_end] = triggers[i]


        # Create Datasets
        transform_train_poisoned = transforms.Compose([
            TriggerAppending(trigger=args.trigger, alpha=args.alpha)
        ])

        transform_test_poisoned = transforms.Compose([
            TriggerAppending(trigger=args.trigger, alpha=args.alpha)
        ])

        if args.y_target !=-1:
            y=args.y_target
        else:
            y_=np.random.choice(y_targets,1,replace=True)
            y=y_[0]
            y_chosen.append(y)

        poison_train_set=Purchase_Dataset(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).long(),transform=transform_train_poisoned)
        poison_test_set=Purchase_Dataset(torch.from_numpy(x_val).float(),torch.from_numpy(y_val).long(),transform=transform_test_poisoned)


        #get the index of each instance belonging to different class
        class_idx=build_classes_dict(poison_train_set)
        target_idx=class_idx[y]


        rest_idx=list(set(train_idx)-set(target_idx))

        poisoned_idx = rest_idx[i*num_poisoned_per_owner:(i+1)*num_poisoned_per_owner]

        train_idx_rest=list(set(train_idx)-set(poisoned_idx))
        train_idx=train_idx_rest

        # the benign samples are the rest of all non-poisoned samples
        benign_idx=list(set(benign_idx)-set(poisoned_idx))

        #generate poisoned train set
        poisoned_select_set = poison_train_set[poisoned_idx] # the randomly selected poisoned instances
        poisoned_train_features=poisoned_select_set[0].numpy()
        poisoned_train_target = np.array([y]*num_poisoned_per_owner) # Reassign their label to the target label
        poisoned_train_set=TensorDataset(torch.from_numpy(poisoned_train_features).float(), torch.from_numpy(poisoned_train_target).long())

        poison_trainsets.append(poisoned_train_set)

        #generate posioned test set
        test_class_idx=build_classes_dict(poison_test_set)
        test_target_idx=test_class_idx[y]
        idx_test = list(np.arange(len(x_val)))
        random.shuffle(idx_test)


        test_rest_idx=list(set(idx_test)-set(test_target_idx))
        poisoned_test_selected=poison_test_set[test_rest_idx]


        poisoned_test_target = np.array([y]*len(poisoned_test_selected[1]))
        poisoned_test_features=poisoned_test_selected[0].numpy()
        poisoned_test_set=TensorDataset(torch.from_numpy(poisoned_test_features).float(), torch.from_numpy(poisoned_test_target).long())

        poison_testsets.append(poisoned_test_set)

    # generate benign train set
    benign_train_set_left=benign_train_set[benign_idx]
    benign_train_features=benign_train_set_left[0].numpy()
    benign_train_labels=benign_train_set_left[1].numpy()
    benign_train_set=TensorDataset(torch.from_numpy(benign_train_features).float(), torch.from_numpy(benign_train_labels).long())

    # generate mixed train set; contain both poisoned and benign data instances
    mixed_trainset=torch.utils.data.ConcatDataset([poison_trainsets[i] for i in range(len(poison_trainsets))])
    mixed_trainloader= torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([mixed_trainset,benign_train_set]), batch_size=int(args.train_batch),
                                                   shuffle=True, num_workers=args.workers)

    benign_testloader = torch.utils.data.DataLoader(benign_test_set, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    print("Num of training samples %i, Num of poisoned samples %i, Num of benign samples %i" %(num_training, num_poison_total, num_training - num_poison_total))

    # Model
    print('==> Loading the model')
    model = MLP(dim_in=600,dim_out=100)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Train and val
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train_mix(args, model, mixed_trainloader, criterion, optimizer, epoch, use_cuda)
        test_loss_benign, test_acc_benign = test(benign_testloader, model, criterion, epoch, use_cuda)

        for i in range(len(triggers)):
            poisoned_testloader = torch.utils.data.DataLoader(poison_testsets[i], batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

            print('-'*30+'User {}'.format(i)+'-'*30)

            test_loss_poisoned, test_acc_poisoned = test(poisoned_testloader, model, criterion, epoch, use_cuda)
            # create log file
            logger = Logger(os.path.join(args.checkpoint, 'log_idx{}.txt'.format(i)), title=title,resume=False)
            logger.set_names(['Learning Rate',  'Train ACC.', 'Benign Valid ACC.', 'Backdoor ASR'])

            # append logger file
            logger.append([state['lr'], train_acc, test_acc_benign,test_acc_poisoned])


        # save model
        is_best = test_acc_benign > best_acc
        best_acc = max(test_acc_benign, best_acc)

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