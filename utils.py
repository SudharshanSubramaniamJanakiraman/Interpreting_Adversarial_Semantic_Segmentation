import torch
from collections import namedtuple
import numpy as np
import os
import scipy.misc as smp
import scipy.ndimage
from random import randint


def get_statistics(data_loader):
    channel_sum, channel_squared_sum, batch_size = 0, 0, 0
    for data, _ in data_loader:
        channel_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        channel_sum += torch.mean(data, dim=[0, 2, 3])
        batch_size += 1

    mean = channel_sum / batch_size
    std = ((channel_squared_sum/batch_size) - (mean**2)) ** 0.5
    print(mean)
    print(std)


def get_device():
    device = 'cpu'
    device = "cuda" if torch.cuda.is_available() else device
    device = "mps" if torch.backends.mps.is_available() else device

    return device


def id_color():
    Label = namedtuple("Label", ["name", "train_id", "color"])
    drivables = [
        Label("road", 0, (31, 255, 255)),
        Label("sidewalk", 1, (7, 255, 255)),
        Label("building", 2, (0, 71, 255)),
        Label("wall", 3, (206, 44, 40)),
        Label("fence", 4, (0, 255, 194)), 
        Label("pole", 5, (255, 255, 0)),
        Label("traffic light", 6, (10, 0, 255)),
        Label("traffic sign", 7, (0, 214, 255)),
        Label("vegetation", 8, (255, 61, 6)),
        Label("terrain", 9, (0, 92, 255)),
        Label("sky", 10,  (120, 120, 120)) ,
        Label("person", 11, (51, 255, 0)),
        Label("rider", 12, (220, 220, 220)),
        Label("car", 13, (255, 0, 102)),
        Label("truck", 14, (140, 140, 140)),
        Label("bus", 15, (44, 108, 239)),
        Label("train", 16, (184, 255, 0)),
        Label("motorcycle", 17, (235, 12, 255)),
        Label("bicycle", 18, (194, 255, 0)), 
        Label("Unknonwn", 19, (0, 0, 0))
    ]

    train_id_to_color = [c.color for c in drivables if (
        c.train_id != -1 and c.train_id != 255)]
    train_id_to_color = np.array(train_id_to_color)

    return train_id_to_color


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_one_hot(labels, num_classes, device):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    Parameters
    ----------
        labels : torch.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
        num_classes : int
        Number of classes
        device: string
        Device to place the new tensor on. Should be same as input
    Returns
    -------
        target : torch.Tensor on given device
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    #print("Labels here is",labels.shape,labels.type())
    one_hot = torch.FloatTensor(
        labels.size(0),
        num_classes, labels.size(2),
        labels.size(3)).zero_()
    one_hot = one_hot.to(device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def generate_target(y_test):
    my_test_original = y_test
    my_test = np.argmax(y_test[0, :, :, :], axis=1)
    preds = smp.toimage(my_test)
    y_target = y_test

    target_class = 13

    dilated_image = scipy.ndimage.binary_dilation(
        y_target[0, target_class, :, :],
        iterations=6).astype(
        y_test.dtype)

    for i in range(256):
        for j in range(256):
            y_target[0, target_class, i, j] = dilated_image[i, j]

    for i in range(256):
        for j in range(256):
            potato = np.count_nonzero(y_target[0, :, i, j])
            if (potato > 1):
                x = np.where(y_target[0, :, i, j] > 0)
                k = x[0]
                if k[0] == target_class:
                    y_target[0, k[1], i, j] = 0.
                else:
                    y_target[0, k[0], i, j] = 0.

    my_target = np.argmax(y_target[0, :, :, :], axis=1)
    preds = smp.toimage(my_target)
    return y_target


def generate_target_swap(y_test):

    y_target = y_test

    y_target_arg = np.argmax(y_test, axis=1)

    y_target_arg_no_back = np.where(y_target_arg > 0)

    y_target_arg = y_target_arg[y_target_arg_no_back]

    classes = np.unique(y_target_arg)

    if len(classes) > 3:

        first_class = 0

        second_class = 0

        third_class = 0

        while first_class == second_class == third_class:
            first_class = classes[randint(0, len(classes)-1)]
            f_ind = np.where(y_target_arg == first_class)
            # print(np.shape(f_ind))

            second_class = classes[randint(0, len(classes)-1)]
            s_ind = np.where(y_target_arg == second_class)

            third_class = classes[randint(0, len(classes) - 1)]
            t_ind = np.where(y_target_arg == third_class)

            summ = np.shape(f_ind)[1] + np.shape(s_ind)[1] + np.shape(t_ind)[1]

            if summ < 1000:
                first_class = 0

                second_class = 0

                third_class = 0

        for i in range(256):
            for j in range(256):
                temp = y_target[0, second_class, i, j]
                y_target[0, second_class, i, j] = y_target[0, first_class, i, j]
                y_target[0, first_class, i, j] = temp

    else:
        y_target = y_test
        print('Not enough classes to swap!')
    return y_target
