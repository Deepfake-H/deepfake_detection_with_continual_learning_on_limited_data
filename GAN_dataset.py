import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data


# import pdb

class GAN_dataset(data.Dataset):
    def __init__(self, root, name, train=True, leave_one_out=False, transform=None, check_cached=False):
        self.image_dir = './data/'
        self.root = os.path.expanduser(root)
        self.name = name
        self.data_dir = os.path.join(self.image_dir, name)

        self.train = train
        self.leave_one_out = leave_one_out
        self.transform = transform
        self.full_list = ['ForenSynths', 'VFHQ-F']

        name_list = name.split("+")
        self.data = None
        self.labels = None
        real_name_list = []
        if not self.leave_one_out:
            real_name_list = name_list
        else:
            for name in self.full_list:
                if name not in name_list:
                    real_name_list.append(name)
        for name in real_name_list:
            if train:
                data_file = os.path.join(self.root, '{}_train.pt'.format(name))
            else:
                data_file = os.path.join(self.root, '{}_test.pt'.format(name))
            # self.cache_data(data_file, name, check_cached)
            # data, labels = torch.load(data_file)
            print('# Processing data {}  Train:{}'.format(name, self.train))
            ndata, nlabels = read_image_file(self.image_dir, name, self.train)

            if self.data is None:
                self.data = ndata
                self.labels = nlabels
            else:
                self.data = np.concatenate((self.data, ndata), axis=0)
                self.labels = np.concatenate((self.labels, nlabels), axis=0)

        self.data = torch.ByteTensor(self.data)
        self.labels = torch.LongTensor(self.labels)
        print(len(self.labels))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]

    def _check_datafile_exists(self, data_file):
        return os.path.exists(data_file)

    def cache_data(self, data_file, name, check_cached):
        if check_cached:
            if self._check_datafile_exists(data_file):
                print('# Found cached data {}'.format(data_file))
                return

        # process and save as torch files
        print('# Caching data {}'.format(data_file))

        dataset = (
            read_image_file(self.image_dir, name, self.train)
        )

        with open(data_file, 'wb') as f:
            torch.save(dataset, f)


def read_image_file(data_dir, dataset_name, train_flag):
    """Return a Tensor containing the patches
    """
    image_list = []
    filename_list = []
    label_list = []
    # load all possible jpg or png images
    if train_flag:
        search_str = '{}/real/{}/trainA/*.jpg'.format(data_dir, dataset_name)
    else:
        search_str = '{}/real/{}/testA/*.jpg'.format(data_dir, dataset_name)

    for filename in glob.glob(search_str):
        image = cv2.imread(filename)
        # if image.shape[0] != 256:
        #     image = cv2.resize(image, (256, 256))
        if image.shape[0] > 256 or image.shape[1] > 256:
            image = image[:256, :256]
        image_list.append(image)
        label_list.append(1)

    if train_flag:
        search_str = '{}/real/{}/trainA/*.png'.format(data_dir, dataset_name)
    else:
        search_str = '{}/real/{}/testA/*.png'.format(data_dir, dataset_name)

    for filename in glob.glob(search_str):
        image = cv2.imread(filename)
        if image.shape[0] > 256 or image.shape[1] > 256:
            image = image[:256, :256]
        image_list.append(image)
        label_list.append(1)

    if train_flag:
        search_str = '{}/fake/{}/trainA/*.png'.format(data_dir, dataset_name)
    else:
        search_str = '{}/fake/{}/testA/*.png'.format(data_dir, dataset_name)

    for filename in glob.glob(search_str):
        image = cv2.imread(filename)
        if image.shape[0] > 256 or image.shape[1] > 256:
            image = image[:256, :256]
        image_list.append(image)
        label_list.append(0)

    if train_flag:
        search_str = '{}/real/{}/trainB/*.jpg'.format(data_dir, dataset_name)
    else:
        search_str = '{}/real/{}/testB/*.jpg'.format(data_dir, dataset_name)

    for filename in glob.glob(search_str):
        image = cv2.imread(filename)
        if image.shape[0] > 256 or image.shape[1] > 256:
            image = image[:256, :256]
        image_list.append(image)
        label_list.append(1)

    if train_flag:
        search_str = '{}/fake/{}/trainB/*.png'.format(data_dir, dataset_name)
    else:
        search_str = '{}/fake/{}/testB/*.png'.format(data_dir, dataset_name)

    for filename in glob.glob(search_str):
        image = cv2.imread(filename)
        if image.shape[0] > 256 or image.shape[1] > 256:
            image = image[:256, :256]
        image_list.append(image)
        label_list.append(0)

    all_in_one = list(zip(image_list, label_list))
    random.shuffle(all_in_one)
    image_list[:], label_list[:] = zip(*all_in_one)

    return np.array(image_list), np.array(label_list)
