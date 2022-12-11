import torch
from dataset import BDD100kSS, BDD100kSSADV
from pspnet import PSPNet
from utils import get_device, id_color
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


# Hydra Imports
import hydra
from omegaconf import DictConfig, OmegaConf

# Imports to supress warnings
import warnings
warnings.filterwarnings("ignore")

from typing import Optional
from tqdm import tqdm

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


# Each label is a tuple with name, class id and color

# 0:  road
# 1:  sidewalk
# 2:  building
# 3:  wall
# 4:  fence
# 5:  pole
# 6:  traffic light
# 7:  traffic sign
# 8:  vegetation
# 9:  terrain
# 10: sky
# 11: person
# 12: rider
# 13: car
# 14: truck
# 15: bus
# 16: train
# 17: motorcycle
# 18: bicycle




def im_show(img, pred, lab, val):

    train_id_to_color = id_color()

    mean = torch.Tensor([0.4245, 0.4145, 0.3702])
    std = torch.Tensor([0.2844, 0.2669, 0.2500])
    mean = mean.to(get_device())
    std = std.to(get_device())
    img = img.squeeze(0)
    img = img.mul(torch.tensor(std).view(3, 1, 1)).add(
        torch.tensor(mean).view(3, 1, 1)).detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)

    pred = pred.squeeze(0)
    lab = lab.squeeze(0)

    pred = pred.detach().cpu().numpy()
    lab = lab.detach().cpu().numpy()
    pred = pred.astype(int)
    lab = lab.astype(int)
    lab[lab == 255] = 19
    # lab[lab==254] = 19

    fig = plt.figure(figsize=(20, 10))
    sub1 = fig.add_subplot(2, 2, 1)
    sub1.imshow(img)
    sub1.title.set_text('Image')
    sub1.axis('off')
    sub2 = fig.add_subplot(2, 2, 2)
    sub2.imshow(train_id_to_color[pred])
    sub2.title.set_text('Prediction')
    sub2.axis('off')
    sub3 = fig.add_subplot(2, 1, 2)
    sub3.imshow(train_id_to_color[lab])
    sub3.title.set_text('Ground Truth')
    sub3.axis('off')
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    plt.savefig(f"viz_{val}", dpi=300, bbox_inches='tight')


def main():
    model = PSPNet()
    model.to(get_device())
    model.load_state_dict(
        torch.load(
            "/home/sjsudharshandl641/robust_sem_seg/ckpt/BDD100k_SemSeg_PSPNET_BackBone_resnet50_LR_0.0001/pytorch_model.pth.tar"))
    model.eval()

    valid_data = BDD100kSS(mode="val")
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

    data_iter = iter(valid_dataloader)

    for val in [10, 29, 319, 289, 600]:
        data_iter = iter(valid_dataloader)
        for i in range(val):

            images, labels = next(data_iter)

        images, labels = images.to(get_device()), labels.to(get_device())

        pred_logits = model(images)
        pred_probs = torch.softmax(pred_logits[0], dim=1)
        prediction = torch.argmax(pred_probs, dim=1)
        im_show(images, prediction, labels, val)


if __name__ == '__main__':
    main()
