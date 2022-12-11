# Import Necessary Packages

# Viuslization and Matrix Operation Imports
import wandb
import numpy as np
from matplotlib import pyplot as plt
import cv2
import PIL

# Pytorch Model and Datasets import
from pspnet import PSPNet
from dataset import BDD100kSS, BDD100kSSADV

# Pytorch Importd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Utils Import
from utils import get_device, id_color, make_one_hot, generate_target_swap

from typing import Optional
import os

# SegGradCam Import
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Hydra Imports
import hydra
from omegaconf import DictConfig, OmegaConf

# Imports to supress warnings
import warnings
warnings.filterwarnings("ignore")

from adv_dag import DAG, SegmentationModelOutputWrapper


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


def bdd100kSemanticClassMapping():
    sem_classes = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle', 'Unknonwn']

    sem_class_to_idx = {}
    for idx, classes in enumerate(sem_classes):
        sem_class_to_idx[classes] = idx
    return sem_class_to_idx


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[:, self.category, :, :] * self.mask).sum()

# Class to output prediction logits removing any dependancies added for
# training purposes


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def stack_image_target(image, output_logits, mean, std,
                       class_conditioning: str = 'car'):

    mean, std = convert_list2tensor(mean), convert_list2tensor(std)
    img = un_normalize_image(image=image, mean=mean, std=std)
    normalized_masks = torch.softmax(output_logits, dim=1)
    classes2idx = bdd100kSemanticClassMapping()

    category = classes2idx[class_conditioning]
    mask = normalized_masks[0, :, :, :].argmax(
        axis=0).detach().cpu().numpy()
    mask_uint8 = 255 * np.uint8(mask == category)
    mask_float = np.float32(mask == category)
    mask_img = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)

    both_images = np.uint8(np.hstack((255*img, mask_img)))
    images = wandb.Image(
        both_images, caption=f"Left: Image, Right: {category} mask")
    wandb.log({"stack_image": images})

    return img, category, mask_float, mask


def interpret_sem_seg_results_conditioned_on_a_category(
        model, image_tensor, rgb_image, category, mask_float):
    target_layers = [model.pred_branch]
    targets = [SemanticSegmentationTarget(category, mask_float)]
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=image_tensor,
                        targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    grid_list = [np.uint8(255*rgb_image)]
    images = wandb.Image(
        np.uint8(cam_image),
        caption=f"Class Activation Map")
    wandb.log({"CAM": images})
    return np.uint8(255*rgb_image)


@hydra.main(version_base=None, config_path="conf", config_name="cam")
def main(cfg: DictConfig):
    print('###################################################################')
    print('LOAD DATA FROM CONFIG FILE')
    print('###################################################################')
    print()
    print(OmegaConf.to_yaml(cfg))
    print()
    print('###################################################################')

    cam_config = cfg['cam']
    weights_path = cam_config['weights']
    dataset = cam_config['dataset']
    mode = cam_config['mode']
    adv_location = None if cam_config['adv_location'] == "" else cam_config['adv_location']
    inference_batch_size = cam_config['batch_size']
    shuffle = cam_config['shuffle']
    mean = list(cam_config['mean'])
    std = list(cam_config['std'])
    batch_idx = cam_config['batch_idx']
    class_conditioning = cam_config['class_conditioning']
    num_iterations = cam_config["num_iterations"]
    gamma = cam_config["gamma"]
    print()
    print('###################################################################')
    print('LOG USING WEIGHTS AND BIASES')
    print('###################################################################')
    print()

    exp_name = f'CAM_{dataset}_{mode}_{cam_config["model"]}_{batch_idx}'
    wandb.init(
        project=f"SemSeg_BDD100k", config=cam_config, name=exp_name,
        id=exp_name,
        resume="allow")

    print('###################################################################')
    print('LOAD MODEL')
    print('###################################################################')
    print()
    model = get_pytorch_model(cam_config['model'])
    model.to(get_device())
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    model_wrapper = SegmentationModelOutputWrapper(model)
    print()
    print('###################################################################')
    print('LOAD DATA')
    print('###################################################################')
    print()
    data = get_BDD100k_dataset(dataset, mode, adv_location)

    dl = get_pytorch_dataloader(data, inference_batch_size, shuffle)

    images, labels = get_batch_data(dl, batch_idx)
    print()
    print('###################################################################')
    print('CREATING CLASS CONDITIONED MASK & STACKED OUTPUT')
    print('###################################################################')
    print()
    wrapper = SegmentationModelOutputWrapper(model)
    output_logits = wrapper(images)

    img, category, mask_float, mask = stack_image_target(
        images, output_logits, mean, std, class_conditioning)
    print()
    print('###################################################################')
    print('CLASS CONDITIONED CLASS ACTIVATION MAPS FOR SEGMENTATION')
    print('###################################################################')
    print()
    cam_image = interpret_sem_seg_results_conditioned_on_a_category(
        model, images, img, category, mask_float)
    print()
    print('###################################################################')
    print('DAG ATTACK')
    print('###################################################################')
    print()
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
                                                                                                       verbose=True)
    output_logits = wrapper(image_adv)

    img, category, mask_float, mask = stack_image_target(
        image_adv, output_logits, mean, std, class_conditioning)
    cam_image = interpret_sem_seg_results_conditioned_on_a_category(
        model, image_adv, img, category, mask_float)
    print()
    print('###################################################################')
    print('RUN FINISHED, RESULTS AVAILABE @ W&B UI')
    print('###################################################################')
    print()
    _, adv = torch.max(adv_target, 1)
    adv = adv[0].cpu().numpy()
    id2color = id_color()
    print(np.unique(prediction_iteration[0]),
          np.unique(prediction_iteration[-1]))
    plt.figure()
    plt.subplot(221)
    plt.title('Original Image')
    plt.imshow(image_iteration[0],cmap='gray')
    plt.axis('off')
    plt.subplot(222)
    plt.title('Adversarial Image')
    plt.imshow(image_iteration[-1],cmap='gray')
    plt.axis('off')
    plt.subplot(223)
    plt.title('Original prediction')
    plt.imshow(id2color[prediction_iteration[0]], cmap='jet')
    plt.axis('off')
    plt.subplot(224)
    plt.title('Adversarial prediction')
    plt.imshow(id2color[prediction_iteration[-1]], cmap='jet')
    plt.axis('off')
    plt.show()
    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.savefig("test_jpg", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()



