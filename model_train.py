from __future__ import division, print_function

import copy
import glob
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, models
from tqdm import tqdm

import GAN_dataset
from continuum.continuum import continuum
from utils.name_match import agents
from utils.utils import maybe_cuda


class GANDataset(GAN_dataset.GAN_dataset):
    """
    GANDataset to read images.  
    """

    def __init__(self, train=True, transform=None, batch_size=None, *arg, **kw):
        super(GANDataset, self).__init__(train=train, *arg, **kw)
        self.transform = transform
        self.train = train
        self.batch_size = batch_size

    def __getitem__(self, index):

        img = self.data[index]
        label = self.labels[index]

        im = deepcopy(img.numpy()[16:240,16:240,:])

        im = im.astype(np.float32)
        im = im / 255.0
        for i in range(3):
            img = im[:, :, i]
            fft_img = np.fft.fft2(img)
            fft_img = np.log(np.abs(fft_img) + 1e-3)
            fft_min = np.percentile(fft_img, 5)
            fft_max = np.percentile(fft_img, 95)
            fft_img = (fft_img - fft_min) / (fft_max - fft_min)
            fft_img = (fft_img - 0.5) * 2
            fft_img[fft_img < -1] = -1
            fft_img[fft_img > 1] = 1

            im[:, :, i] = fft_img

        im = np.transpose(im, (2, 0, 1))

        return (im, label)

    def __len__(self):
        return self.labels.size(0)


def create_loaders(params, Train=True):
    dataset_names = copy.copy(params.training_set) if Train else copy.copy(params.test_set)
    batch_size = copy.copy(params.batch_size) if Train else copy.copy(params.test_batch_size)

    kwargs = {'num_workers': params.num_workers, 'pin_memory': params.pin_memory} if params.cuda else {}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
                         GANDataset(train=Train,
                                    leave_one_out=False,
                                    batch_size=batch_size,
                                    root=params.dataroot,
                                    name=name,
                                    check_cached=params.check_cached,
                                    transform=transform),
                         batch_size=batch_size,
                         shuffle=False, **kwargs)}
                    for name in dataset_names]

    return data_loaders

def train(params, train_loader, model, optimizer, criterion):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
        image_pair, label = data
        if params.cuda:
            image_pair, label = image_pair.cuda(), label.cuda()
            image_pair, label = Variable(image_pair), Variable(label)
            out = model(image_pair)

        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    adjust_learning_rate(params, optimizer)


def test(test_loader, model, logger_test_name, b_cuda=False):
    # switch to evaluate mode, calculate test accuracy
    model.eval()

    labels, predicts = [], []
    outputs = []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (image_pair, label) in pbar:
        image_pair = maybe_cuda(image_pair, b_cuda)
        with torch.no_grad():
            image_pair, label = Variable(image_pair), Variable(label)
        out = model(image_pair)
        _, pred = torch.max(out, 1)
        ll = label.data.cpu().numpy().reshape(-1, 1)
        pred = pred.data.cpu().numpy().reshape(-1, 1)
        out = out.data.cpu().numpy().reshape(-1, 2)
        labels.append(ll)
        predicts.append(pred)
        outputs.append(out)

    num_tests = test_loader.dataset.labels.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    predicts = np.vstack(predicts).reshape(num_tests)
    outputs = np.vstack(outputs).reshape(num_tests, 2)

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

def continual_learning_test(test_loader: object, model, logger_test_name, b_cuda=False):
    # switch to evaluate mode
    model.eval()

    labels, predicts = [], []

    pbar = tqdm(enumerate(test_loader), position=0, leave=True, file=sys.stdout)
    for batch_idx, (image_pair, label) in pbar:
        image_pair = maybe_cuda(image_pair, b_cuda)
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

    #print('\33[91mTest set: {}\n\33[0m'.format(logger_test_name))

    acc = np.sum(labels == predicts) / float(num_tests)
    ## change by Holly
    #print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(acc))

    # TPR pos=real=1
    pos_label = labels[labels == 1]
    pos_pred = predicts[labels == 1]
    TPR = np.sum(pos_label == pos_pred) / float(pos_label.shape[0])
    #print('\33[91mTest set: TPR: {:.8f}\n\33[0m'.format(TPR))

    # TNR neg=fake=0
    neg_label = labels[labels == 0]
    neg_pred = predicts[labels == 0]
    TNR = np.sum(neg_label == neg_pred) / float(neg_label.shape[0])
    #print('\33[91mTest set: TNR: {:.8f}\n\33[0m'.format(TNR))

    print('\33[91mTest set: {} - Accuracy: {:.8f} - TPR: {:.8f} - TNR: {:.8f}\n\33[0m'.format(logger_test_name, acc, TPR, TNR))
    # print('\33[91mAll Predicts Values: {}\n\33[0m'.format(predicts))

    return acc

def adjust_learning_rate(params, optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        # group['lr'] = params.lr*((1-params.lr_decay)**group['step'])
        group['lr'] = params.lr

    return


def create_optimizer(params, model):
    # setup optimizer
    if params.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params.lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=params.wd)
    elif params.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params.lr,
                               weight_decay=params.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(params.optimizer))
    return optimizer


def continual_learning_train(params):
    loc_model = models.resnet34(pretrained=True)
    num_ftrs = loc_model.fc.in_features
    loc_model.fc = nn.Linear(num_ftrs, 2)
    loc_model = maybe_cuda(loc_model, params.cuda)
    opt = create_optimizer(params, loc_model)

    agent = agents[params.agent](loc_model, opt, params)

    test_loaders = create_loaders(params, Train=False)

    # loop diff Datasets
    for train_dataset in params.training_set:
        dataset_start = time.time()
        training_set = train_dataset
        print('\nWorking on Dataset\n{}\n'.format(training_set))
        #cl_train_loader, cl_test_loaders = create_loaders()
        print('\nparsed options:\n{}\n'.format(vars(params)))
        data_continuum = continuum(training_set, params.cl_type, params)

        start = params.start_epoch
        end = start + params.epochs

        accuracy_list = []
        # Start one Dataset Training for several epochs
        for run in range(params.num_runs):
            tmp_acc = []
            run_start = time.time()
            data_continuum.new_run()

            for i, (x_train, y_train, labels) in enumerate(data_continuum):
                batch_start = time.time()
                print("-----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                batch_end = time.time()
                print("-----------run {} batch {}---train time {}s".format(run, i, batch_end - batch_start))

            run_end = time.time()
            print("-----------run {}----------- time {}s".format(run, run_end - run_start))

        dataset_end = time.time()
        print('########### Total Time on Dataset {} is: {}s ###########'.format(training_set,
                                                                                dataset_end - dataset_start))

        # Save the final model
        out_model = agent.model
        torch.save({'state_dict': out_model.state_dict()},
                   '{}{}_{}_{}_{}_{}_{}_checkpoint_latest.pth'.format(params.model_dir,
                                                                      params.continual_learning,
                                                                      params.agent,
                                                                      training_set,
                                                                      params.feature,
                                                                      params.mode,
                                                                      params.model))

        # Run test
        print('\n###############################################\n'
              'Continual Learning - Agent: {} - Training Set: {}\n'.format(params.agent, training_set))
        for test_loader in test_loaders:
            continual_learning_test(test_loader['dataloader'], out_model, test_loader['name'], params.cuda)
        print('###############################################\n')


def normal_train(params):
    train_loaders = create_loaders(params, Train=True)
    test_loaders = create_loaders(params, Train=False)

    last_dataset = None
    for train_loader in train_loaders:
        train_set_name = train_loader['name']
        print('\n*****************************************\n')
        print('Start working on: {}\n'.format(train_set_name))
        print('parsed options:\n{}\n'.format(vars(params)))

        save_file_name = '{}{}_{}_{}_{}_{}_checkpoint_latest.pth'.format(params.model_dir,
                                                                         params.continual_learning,
                                                                         train_set_name,
                                                                         params.feature,
                                                                         params.mode,
                                                                         params.model)
        if os.path.isfile(save_file_name):
            print('\n######## Model checkpoint has been find at {}\n######## Skip Training...\n'.format(save_file_name))
            last_dataset = train_set_name
            continue

        if params.model == 'resnet':
            model = models.resnet34(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)

        optimizer1 = create_optimizer(params, model)
        criterion = nn.CrossEntropyLoss()

        model = maybe_cuda(model, params.cuda)
        criterion = maybe_cuda(criterion, params.cuda)

        # Normal task, load model from last dataset trainning
        if params.continual_learning == 'Normal' and last_dataset is not None:
            resume_file = '{}{}_{}_{}_{}_{}_checkpoint_latest.pth'.format(params.model_dir,
                                                                          params.continual_learning,
                                                                          last_dataset,
                                                                          params.feature,
                                                                          params.mode,
                                                                          params.model)
            if os.path.isfile(resume_file):
                print('=> load checkpoint {}\n'.format(resume_file))
                checkpoint = torch.load(resume_file)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print('=> no checkpoint found at {}'.format(resume_file))

        start = params.start_epoch
        end = start + params.epochs

        print("Start Training")
        # Start Training
        for epoch in range(start, end):
            # iterate over train loaders
            train(params, train_loader['dataloader'], model, optimizer1, criterion)

        # Save the final model
        torch.save({'state_dict': model.state_dict()}, save_file_name)

        # Run test
        print('\n###############################################\n'
              'Normal Train - CL: {} - Training Set: {}\n'.format(params.continual_learning, train_set_name))
        for test_loader in test_loaders:
            test(test_loader['dataloader'], model, test_loader['name'], params.cuda)
        print('\n###############################################\n')

        last_dataset = train_set_name

def run_test(params):
    if params.model == 'resnet':
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

    print('\n##### Start testing #####\n')

    if os.path.isfile(params.loadmodel):
        print('=> load checkpoint {}\n'.format(params.loadmodel))
        checkpoint = torch.load(params.loadmodel)
        model.load_state_dict(checkpoint['state_dict'])
        model = maybe_cuda(model, params.cuda)
    else:
        print('=> no checkpoint found at {}'.format(params.loadmodel))
        return

    test_loaders = create_loaders(params, Train=False)

    # Run test
    for test_loader in test_loaders:
        continual_learning_test(test_loader['dataloader'], model, test_loader['name'], params.cuda)
    print('\n##### ------------- #####\n')


def run_test_all(params):
    if params.model == 'resnet':
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

    test_loaders = create_loaders(params, Train=False)

    print('\n##### Start testing #####\n')

    search_str = '{}*.pth'.format(params.model_dir)
    for filename in glob.glob(search_str):
        if os.path.isfile(filename):
            print('=> load checkpoint {}\n'.format(filename))
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint['state_dict'])
            model = maybe_cuda(model, params.cuda)
        else:
            print('=> no checkpoint found at {}'.format(filename))
            print('\n##### ------------- #####\n')
            continue

        # Run test
        for test_loader in test_loaders:
            continual_learning_test(test_loader['dataloader'], model, test_loader['name'], params.cuda)
        print('\n##### ------------- #####\n')


def single_run_entrance(params):
    # set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
    # order to prevent any memory allocation on unused GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu_id

    if params.runtest:
        if params.runtestall:
            run_test_all(params)
        else:
            run_test(params)
        return

    if params.continual_learning == 'CL':
        continual_learning_train(params)
    elif params.continual_learning == 'Normal' or params.continual_learning == 'NoCL':
        normal_train(params)
    else:
        print('Nothing has run!')



