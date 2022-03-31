from __future__ import division, print_function

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
from tqdm import tqdm

from model_train import GANDataset

parser = argparse.ArgumentParser(description='PyTorch GAN Image Detection')

# Training settings
parser.add_argument('--dataroot', type=str,
                    default='./datasets/',
                    help='path to dataset')
parser.add_argument('--training_set', default='horse',
                    help='The name of the training set. If leave_one_out flag is set, \
                    it is the leave-out set(use all other sets for training).')
parser.add_argument('--test_set', default='transposed_conv', type=str,
                    help='Choose test set from trainsposed_conv, nn, jpeg and resize')
parser.add_argument('--feature', default='image',
                    help='Feature used for training, choose from image and fft')
parser.add_argument('--mode', type=int, default=0,
                    help='fft frequency band, 0: full, 1: low, 2: mid, 3: high')
parser.add_argument('--leave_one_out', action='store_true', default=False,
                    help='Test leave one out setting, using all other sets for training and test on a leave-out set.')

parser.add_argument('--jpg_level', type=str, default='90',
                    help='Test with different jpg compression effiecients, only effective when use jpg for test set.')
parser.add_argument('--resize_size', type=str, default='200',
                    help='Test with different resize sizes, only effective when use resize for test set.')

parser.add_argument('--result-dir', default='./final_output/',
                    help='folder to output result in csv')
parser.add_argument('--model-dir', default='./model/',
                    help='folder to output model checkpoints')
parser.add_argument('--model', default='resnet',
                    help='Base classification model')
parser.add_argument('--num-workers', default=1,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory', type=bool, default=True,
                    help='')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch_size', type=int, default=10,
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=10,
                    help='input batch size for testing (default: 32)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--data_augment', action='store_true', default=False,
                    help='Use data augmentation or not')
parser.add_argument('--check_cached', action='store_true', default=True,
                    help='Use cached dataset or not')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed (default: -1)')
parser.add_argument('--load_model', type=str, default='./model/NoCL_VFHQ_fft_0_resnet_checkpoint_latest.pth',
                    help='saved model')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

suffix = '{}'.format(args.training_set)

if args.test_set == 'Run1':
    dataset_names = ['VFHQ', 'ForenSynths']
else:
    print('Test set does not support!')
    exit(-1)


# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
    # set random seeds
    if args.seed > -1:
        torch.cuda.manual_seed_all(args.seed)

# set random seeds
if args.seed > -1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

try:
    os.stat('{}/'.format(args.result_dir))
except:
    os.makedirs('{}/'.format(args.result_dir))


def create_loaders():
    test_dataset_names = copy.copy(dataset_names)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    print(test_dataset_names)
    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
                         GANDataset(train=args.leave_one_out,
                                    batch_size=args.test_batch_size,
                                    root=args.dataroot,
                                    name=name,
                                    check_cached=args.check_cached,
                                    transform=transform),
                         batch_size=args.test_batch_size,
                         shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return test_loaders


def test(test_loader: object, model, epoch, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, predicts = [], []

    pbar = tqdm(enumerate(test_loader), position=0, leave=True, file=sys.stdout)
    for batch_idx, (image_pair, label) in pbar:
        if args.cuda:
            image_pair = image_pair.cuda()

        with torch.no_grad():
            image_pair, label = Variable(image_pair), Variable(label)

        out = model(image_pair)
        _, pred = torch.max(out, 1)
        ll = label.data.cpu().numpy().reshape(-1, 1)
        pred = pred.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)
        predicts.append(pred)

    num_tests = test_loader.dataset.labels.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    predicts = np.vstack(predicts).reshape(num_tests)

    print('\33[91mTest set: {}\n\33[0m'.format(logger_test_name))

    acc = np.sum(labels == predicts) / float(num_tests)

    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(acc))

    # TPR pos=real=1
    pos_label = labels[labels == 1]
    pos_pred = predicts[labels == 1]
    TPR = np.sum(pos_label == pos_pred) / float(pos_label.shape[0])
    print('\33[91mTest set: TPR: {:.8f}\n\33[0m'.format(TPR))

    # TNR neg=fake=0
    neg_label = labels[labels == 0]
    neg_pred = predicts[labels == 0]
    TNR = np.sum(neg_label == neg_pred) / float(neg_label.shape[0])
    print('\33[91mTest set: TNR: {:.8f}\n\33[0m'.format(TNR))

    # print('\33[91mAll Predicts Values: {}\n\33[0m'.format(predicts))

    return acc


def main(test_loaders, model):
    print('\nparsed options:\n{}\n'.format(vars(args)))
    acc_list = []
    if args.cuda:
        model.cuda()

    start = args.start_epoch
    end = start + args.epochs
    for test_loader in test_loaders:
        acc = test(test_loader['dataloader'], model, 0, test_loader['name']) * 100
        acc_list.append(str(acc))


if __name__ == '__main__':
    if args.model == 'resnet':
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

    print('load model from {}'.format(args.load_model))
    load_model = torch.load(args.load_model)
    model.load_state_dict(load_model['state_dict'])

    test_loaders = create_loaders()
    main(test_loaders, model)
