# Code from https://github.com/IFL-CAMP/dense_adversarial_generation_pytorch
# Ref Paper : https://arxiv.org/pdf/1703.08603.pdf
# Import Necessary Python Packages
import cv2
import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from pspnet import PSPNet
from dataset import BDD100kSS, BDD100kSSADV

import torch
from torch.utils.data import DataLoader

import wandb

from typing import Optional

from utils import make_one_hot, generate_target_swap, get_device, id_color

# Hydra Imports
import hydra
from omegaconf import DictConfig, OmegaConf

# Imports to supress warnings
import warnings
warnings.filterwarnings("ignore")


# Function to create a instance of pytorch model based on its name
def get_pytorch_model(model_name: Optional[str] = None):
    try:
        if model_name != None:  # better: if item is not None
            if model_name == 'pspnet':
                return PSPNet()
        else:
            raise TypeError

    except TypeError:
        print('Model Name Should Not be None')

# Function to either load Benign/Adversarial Data


def get_BDD100k_dataset(
        dataset: Optional[str] = None, mode: str = 'val',
        adv_location: Optional[str] = None):
    try:
        if dataset != None:  # better: if item is not None
            if dataset == 'benign':
                return BDD100kSS(mode)
            elif dataset == 'adversarial':
                assert adv_location is not None, 'Adversarial Location Must not be None for Loading Adversarial data'
                return BDD100kSSADV(mode, adv_location)
        else:
            raise TypeError

    except TypeError:
        print('Dataset Name Should Not be None')

# Function for Dataloader creation


def get_pytorch_dataloader(data, batch_size: int = 1, shuffle: bool = False):
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)

# Function to get Data in the batch


def get_batch_data(dataloader, batch_idx: int = 1):
    data_iter = iter(dataloader)
    assert batch_idx < len(dataloader), 'Invalid Batch IDX'
    for i in range(batch_idx):
        images, labels = next(data_iter)
    images, labels = images.to(get_device()), labels.to(get_device())
    return images, labels

# Function to Unnormalize Images
# Images will be in range [0,1]


def un_normalize_image(image, mean, std):
    img = image.squeeze(0)
    img = img.mul(torch.tensor(std).view(3, 1, 1)).add(
        torch.tensor(mean).view(3, 1, 1)).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img


def convert_list2tensor(data_list: list = []):
    return torch.tensor(data_list).to(get_device())


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def DAG(
        model, image, ground_truth, adv_target, mean, std, num_iterations=20,
        gamma=0.07, no_background=True, background_class=0, device='cuda:0',
        verbose=False):
    '''
    Generates adversarial example for a given Image

    Parameters
    ----------
        model: Torch Model
        image: Torch tensor of dtype=float. Requires gradient. [b*c*h*w]
        ground_truth: Torch tensor of labels as one hot vector per class
        adv_target: Torch tensor of dtype=float. This is the purturbed labels. [b*classes*h*w]
        num_iterations: Number of iterations for the algorithm
        gamma: epsilon value. The maximum Change possible.
        no_background: If True, does not purturb the background class
        background_class: The index of the background class. Used to filter background
        device: Device to perform the computations on
        verbose: Bool. If true, prints the amount of change and the number of values changed in each iteration
    Returns
    -------
        Image:  Adversarial Output, logits of original image as torch tensor
        logits: Output of the Clean Image as torch tensor
        noise_total: List of total noise added per iteration as numpy array
        noise_iteration: List of noise added per iteration as numpy array
        prediction_iteration: List of prediction per iteration as numpy array
        image_iteration: List of image per iteration as numpy array
    '''
    mean = convert_list2tensor(mean)
    std = convert_list2tensor(std)
    min_clamp = torch.min(image).item()
    max_clamp = torch.max(image).item()
    noise_total = []
    noise_iteration = []
    prediction_iteration = []
    image_iteration = []
    background = None
    logits = model(image)
    orig_image = image
    _, predictions_orig = torch.max(logits, 1)
    predictions_orig = make_one_hot(predictions_orig, logits.shape[1], device)

    if (no_background):
        background = torch.zeros(
            (logits.shape[0],
             logits.shape[1] + 1, logits.shape[2],
             logits.shape[3]))
        background[:, background_class, :, :] = torch.ones(
            (background.shape[2], background.shape[3]))
        background = background.to(device)

    for a in range(num_iterations):
        output = model(image)
        _, predictions = torch.max(output, 1)
        prediction_iteration.append(predictions[0].cpu().numpy())
        predictions = make_one_hot(predictions, logits.shape[1]+1, device)

        condition1 = torch.eq(predictions, ground_truth)
        condition = condition1

        if no_background:
            condition2 = (ground_truth != background)
            condition = torch.mul(condition1, condition2)
        condition = condition.float()

        if (condition.sum() == 0):
            print("Condition Reached")
            image = None
            break
        bg = torch.zeros(
            output.shape[0],
            1, output.shape[2],
            output.shape[3]).to(
            get_device())
        output = torch.cat((output, bg), dim=1)
        # Finding pixels to purturb
        adv_log = torch.mul(output, adv_target)
        # Getting the values of the original output
        clean_log = torch.mul(output, ground_truth)

        # Finding r_m
        adv_direction = adv_log-clean_log
        r_m = torch.mul(adv_direction, condition)
        r_m.requires_grad_()
        # Summation
        r_m_sum = r_m.sum()
        r_m_sum.requires_grad_()
        # Finding gradient with respect to image
        r_m_grad = torch.autograd.grad(r_m_sum, image, retain_graph=True)
        # Saving gradient for calculation
        r_m_grad_calc = r_m_grad[0]

        # Calculating Magnitude of the gradient
        r_m_grad_mag = r_m_grad_calc.norm()

        if (r_m_grad_mag == 0):
            # print("Condition Reached, no gradient")
            # image=None
            break
        # Calculating final value of r_m
        r_m_norm = (gamma/r_m_grad_mag)*r_m_grad_calc

        if no_background:
            # if False:
            condition_image = condition.sum(dim=1)
            condition_image = condition_image.unsqueeze(1)
            r_m_norm = torch.mul(r_m_norm, condition_image)

        # Updating the image
        image = torch.clamp((image+r_m_norm), min_clamp, max_clamp)
        image_iteration.append(un_normalize_image(image, mean=mean, std=std))
        noise_total.append((image-orig_image)[0][0].detach().cpu().numpy())
        noise_iteration.append(r_m_norm[0][0].cpu().numpy())

        if verbose:
            print("Iteration ", a)
            print("Change to the image is ", r_m_norm.sum())
            print("Magnitude of grad is ", r_m_grad_mag)
            print("Condition 1 ", condition1.sum())
            if no_background:
                print("Condition 2 ", condition2.sum())
                print("Condition is", condition.sum())

    return image, logits, noise_total, noise_iteration, prediction_iteration, image_iteration


def generate_adversarial_exmaple(
        idx, images, labels, model_wrapper, mean, std, num_iterations, gamma,
        image_location, label_location, color_location):
    images, labels = images.to(
        get_device()), labels.to(
        get_device())
    labels[labels == 255] = 19
    label_oh = make_one_hot(labels.long(), 20, get_device())
    adv_target = generate_target_swap(label_oh.cpu().numpy())
    adv_target = torch.from_numpy(adv_target).float()
    adv_target = adv_target.to(get_device())
    images.requires_grad_()
    image_adv, output_clean, noise_total, noise_iteration, prediction_iteration, image_iteration = DAG(model=model_wrapper,
                                                                                                        image=images,
                                                                                                        ground_truth=label_oh,
                                                                                                        adv_target=adv_target,
                                                                                                        mean=mean,
                                                                                                        std=std,
                                                                                                        num_iterations=num_iterations,
                                                                                                        gamma=gamma,
                                                                                                        no_background=True,
                                                                                                        background_class=0,
                                                                                                        device=get_device(),
                                                                                                        verbose=False)

    i2c = id_color()
    labels = labels.squeeze(0).cpu().numpy().astype(int)
    if len(image_iteration) != 0:
        cv2.imwrite(
            os.path.join(image_location, f'{idx}_Original.jpg'),
            np.uint8(255 * image_iteration[0]))
        cv2.imwrite(
            os.path.join(image_location, f'{idx}_DAG.jpg'),
            np.uint8(255 * image_iteration[-1]))
        cv2.imwrite(
            os.path.join(color_location, f'{idx}_Original.png'),
            np.uint8(i2c[prediction_iteration[0]]))
        cv2.imwrite(
            os.path.join(color_location, f'{idx}_DAG.png'),
            np.uint8(i2c[prediction_iteration[-1]]))
        cv2.imwrite(
            os.path.join(color_location, f'{idx}_GT.png'),
            np.uint8(i2c[labels]))
        cv2.imwrite(
            os.path.join(label_location, f'{idx}_Original.png'),
            prediction_iteration[0])
        cv2.imwrite(
            os.path.join(label_location, f'{idx}_DAG.png'),
            prediction_iteration[-1])
        cv2.imwrite(
            os.path.join(label_location, f'{idx}_GT.png'),
            labels)


@hydra.main(version_base=None, config_path="conf", config_name="dag")
def main(cfg: DictConfig):
    print('###################################################################')
    print('LOAD DATA FROM CONFIG FILE')
    print('###################################################################')
    print()
    print(OmegaConf.to_yaml(cfg))
    print()
    print('###################################################################')

    dag_config = cfg['dag']
    weights_path = dag_config['weights']
    dataset = dag_config['dataset']
    mode = dag_config['mode']
    adv_location = None if dag_config['adv_location'] == "" else dag_config['adv_location']
    inference_batch_size = dag_config['batch_size']
    shuffle = dag_config['shuffle']
    mean = list(dag_config['mean'])
    std = list(dag_config['std'])
    num_iterations = dag_config["num_iterations"]
    gamma = dag_config["gamma"]
    adv_base_location = dag_config["adv_base_location"]
    if not os.path.exists(adv_base_location):
        os.mkdir(adv_base_location)
    adv_location = os.path.join(
        adv_base_location, "DAG" + '_iter_' + str(num_iterations) + '_gamma_' +
        str(gamma))
    if not os.path.exists(adv_location):
        os.mkdir(adv_location)
    adv_location = os.path.join(adv_location, mode)
    if not os.path.exists(adv_location):
        os.mkdir(adv_location)
    image_location = os.path.join(adv_location, "Images")
    if not os.path.exists(image_location):
        os.mkdir(image_location)
    label_location = os.path.join(adv_location, "Mask")
    if not os.path.exists(label_location):
        os.mkdir(label_location)
    color_location = os.path.join(adv_location, "Color")
    if not os.path.exists(color_location):
        os.mkdir(color_location)

    print('###################################################################')
    print('LOAD MODEL')
    print('###################################################################')
    print()
    model = get_pytorch_model(dag_config['model'])
    model.to(get_device())
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    model_wrapper = SegmentationModelOutputWrapper(model)
    print()

    print()
    print('###################################################################')
    print('LOAD DATA')
    print('###################################################################')
    print()
    data = get_BDD100k_dataset(dataset, mode, adv_location)
    dl = get_pytorch_dataloader(data, inference_batch_size, shuffle)
    # images, labels = get_batch_data(dl, batch_idx)
    print()
    print('###################################################################')
    print('GERATING ADVERSARIAL EXAMPLES')
    print('###################################################################')
    print()

    for idx, (images, labels) in enumerate(tqdm(dl)):
        if os.path.exists(os.path.join(image_location, f'{idx}_Original.jpg')):
            continue
        else:
            generate_adversarial_exmaple(
                idx, images, labels, model_wrapper, mean, std, num_iterations,
                gamma, image_location, label_location, color_location)



if __name__ == '__main__':
    main()
