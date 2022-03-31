import numpy as np
import os
import glob
import random
import cv2

from torchvision import datasets
from continuum.data_utils import create_task_composition, load_task_with_labels
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns

class GANDataset_base(DatasetBase):
    """
    GANDataset to read images.
    """

    def __init__(self, scenario, params, name, *arg, **kw):
        self.image_dir = './data/'
        self.name = name
        dataset = name
        self.data_dir = os.path.join(self.image_dir, name)

        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks

        super(GANDataset_base, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)

    def download_load(self):
        print('Processing data {}   Train:True'.format(self.name))
        self.train_data, self.train_label = self.read_image_file(train_flag=True)
        print(len(self.train_data))
        self.test_data, self.test_label = self.read_image_file(train_flag=False)

    def read_image_file(self, train_flag):
        """Return a Tensor containing the patches
        """
        data_dir = self.image_dir
        dataset_name = self.name
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

    def setup(self):
        if self.scenario == 'ni':
            self.train_set, self.val_set, self.test_set = construct_ns_multiple_wrapper(self.train_data,
                                                                                        self.train_label,
                                                                                        self.test_data, self.test_label,
                                                                                        self.task_nums, 256,
                                                                                        self.params.val_size,
                                                                                        self.params.ns_type, self.params.ns_factor,
                                                                                        plot=self.params.plot_sample)
        elif self.scenario == 'nc':
            self.task_labels = create_task_composition(class_nums=100, num_tasks=self.task_nums, fixed_order=self.params.fix_order)
            self.test_set = []
            for labels in self.task_labels:
                x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
                self.test_set.append((x_test, y_test))
        else:
            raise Exception('wrong scenario')

    def new_task(self, cur_task, **kwargs):
        if self.scenario == 'ni':
            x_train, y_train = self.train_set[cur_task]
            labels = set(y_train)
        elif self.scenario == 'nc':
            labels = self.task_labels[cur_task]
            x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        return x_train, y_train, labels

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def test_plot(self):
        test_ns(self.train_data[:10], self.train_label[:10], self.params.ns_type,
                                                         self.params.ns_factor)



