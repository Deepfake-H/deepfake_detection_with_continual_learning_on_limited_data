import numpy as np
import torch
from torch.utils import data
from utils.setup_elements import transforms_match
from copy import deepcopy

def create_task_composition(class_nums, num_tasks, fixed_order=False):
    classes_per_task = class_nums // num_tasks
    total_classes = classes_per_task * num_tasks
    label_array = np.arange(0, total_classes)
    if not fixed_order:
        np.random.shuffle(label_array)

    task_labels = []
    for tt in range(num_tasks):
        tt_offset = tt * classes_per_task
        task_labels.append(list(label_array[tt_offset:tt_offset + classes_per_task]))
        print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
    return task_labels


def load_task_with_labels_torch(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((y == i).nonzero().view(-1))
    idx = torch.cat(tmp)
    return x[idx], y[idx]


def load_task_with_labels(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((np.where(y == i)[0]))
    idx = np.concatenate(tmp, axis=None)
    return x[idx], y[idx]



class dataset_transform(data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        # self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.y = y
        self.transform = transform  # save the transform

    def __len__(self):
        return len(self.y)#self.x.shape[0]  # return 1 as we have only one image

    def __getitem__(self, idx):
        # return the augmented image
        # if self.transform:
        #     x = self.transform(self.x[idx])
        # else:
        #     x = self.x[idx]
        #
        # return x.float(), self.y[idx]

        img = self.x[idx]
        label = self.y[idx]

        # for resnet
        im = deepcopy(img[16:240, 16:240, :])

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

        return im, label


def setup_test_loader(test_data, params):
    test_loaders = []

    for (x_test, y_test) in test_data:
        test_dataset = dataset_transform(x_test, y_test, transform=transforms_match[params.data])
        test_loader = data.DataLoader(test_dataset, batch_size=params.test_batch, shuffle=True, num_workers=0)
        test_loaders.append(test_loader)
    return test_loaders


def shuffle_data(x, y):
    perm_inds = np.arange(0, x.shape[0])
    np.random.shuffle(perm_inds)
    rdm_x = x[perm_inds]
    rdm_y = y[perm_inds]
    return rdm_x, rdm_y


def train_val_test_split_ni(train_data, train_label, test_data, test_label, task_nums, img_size, val_size=0.1):
    train_data_rdm, train_label_rdm = shuffle_data(train_data, train_label)
    val_size = int(len(train_data_rdm) * val_size)
    #val_size = int(0)
    val_data_rdm, val_label_rdm = train_data_rdm[:val_size], train_label_rdm[:val_size]
    train_data_rdm, train_label_rdm = train_data_rdm[val_size:], train_label_rdm[val_size:]
    test_data_rdm, test_label_rdm = shuffle_data(test_data, test_label)
    train_data_rdm_split = train_data_rdm.reshape(task_nums, -1, img_size, img_size, 3)
    train_label_rdm_split = train_label_rdm.reshape(task_nums, -1)
    val_data_rdm_split = val_data_rdm.reshape(task_nums, -1, img_size, img_size, 3)
    val_label_rdm_split = val_label_rdm.reshape(task_nums, -1)
    test_data_rdm_split = test_data_rdm.reshape(task_nums, -1, img_size, img_size, 3)
    test_label_rdm_split = test_label_rdm.reshape(task_nums, -1)
    return train_data_rdm_split, train_label_rdm_split, val_data_rdm_split, val_label_rdm_split, test_data_rdm_split, test_label_rdm_split