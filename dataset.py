import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import pil_loader
from torchvision import transforms, utils
from typing import Optional
import glob
import os
from PIL import Image, ImageOps
import cv2
import tqdm
from utils import get_statistics
from torchvision.transforms import functional as F
from pspnet import PSPNet
from matplotlib import pyplot as plt
import numpy as np
# from sklearn.utils.class_weight import compute_class_weight


class BDD100kSS(Dataset):

    def __init__(self, mode:str="train", resize=(560, 280)) -> None:
        super().__init__()
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.resize = resize
        self.prepare()

    def prepare(self):
        image_relative_location = "bdd100k/images/10k"
        self.labels_relative_location = "bdd100k/labels/sem_seg/masks"
        images_path = os.path.join(image_relative_location, self.mode)
        self.labels_path = os.path.join(self.labels_relative_location, self.mode)
        self.images_name = glob.glob(images_path+"/*")
        # self.labels_name = glob.glob(labels_path+"/*")
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4245, 0.4145, 0.3702], std=[0.2844, 0.2669, 0.2500])
        # transforms.Normalize(mean=[0.3701, 0.4144, 0.4244], std=[0.2519, 0.2688, 0.2861])
        ])

    def __getitem__(self, index):


        # print("\n Index", index)
        img_path = self.images_name[index]
        # print(os.path.basename( img_path)[:-3]+'png')
        mask_path = os.path.join(self.labels_path, os.path.basename( img_path)[:-3]+'png')
        # print(img_path)
        # print(mask_path)

        img = cv2.imread(img_path)
        img = cv2.resize(img, self.resize, interpolation = cv2.INTER_AREA)
        if self.mode != 'test':
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, self.resize, interpolation = cv2.INTER_NEAREST_EXACT)
            mask = mask.astype("float32")

        image = self.transform(img)
        if self.mode == 'test':
            return image, 0.0
        return image, mask

    def __len__(self):
        return len(self.images_name) 


class BDD100kSSADV(Dataset):

    def __init__(self, mode:str="val", adv_location:str="GN_8") -> None:
        super().__init__()
        assert mode in ["val", "test"]
        self.mode = mode
        self.adv_location = adv_location
        self.prepare()

    def prepare(self):
        image_relative_location = f"bdd100kadv/{self.adv_location}/{self.mode}/Images"
        self.labels_relative_location = f"bdd100kadv/{self.adv_location}/{self.mode}/Mask"
        images_path = image_relative_location 
        self.labels_path = self.labels_relative_location
        self.images_name = glob.glob(images_path+"/*")
        # self.labels_name = glob.glob(labels_path+"/*")
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4245, 0.4145, 0.3702], std=[0.2844, 0.2669, 0.2500])
        # transforms.Normalize(mean=[0.3701, 0.4144, 0.4244], std=[0.2519, 0.2688, 0.2861])
        ])

    def __getitem__(self, index):


        # print("\n Index", index)
        img_path = self.images_name[index]
        # print(os.path.basename( img_path)[:-3]+'png')
        mask_path = os.path.join(self.labels_path, os.path.basename( img_path)[:-3]+'png')
        # print(img_path)
        # print(mask_path)

        img = cv2.imread(img_path)
        # img = cv2.resize(img, self.resize, interpolation = cv2.INTER_AREA)
        if self.mode != 'test':
            mask = cv2.imread(mask_path, 0)
            # mask = cv2.resize(mask, self.resize, interpolation = cv2.INTER_NEAREST_EXACT)
            mask = mask.astype("float32")

        image = self.transform(img)
        if self.mode == 'test':
            return image, 0.0
        return image, mask

    def __len__(self):
        return len(self.images_name) 


# def main():
#     data = BDD100kSS("train")
#     print(len(data))
#     train_dl = DataLoader(data, batch_size=64, shuffle=True)

#     ####### COMPUTE MEAN / STD

#     print(get_statistics(train_dl))

#     # valid_data = BDD100kSS(mode="val")
#     # valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)



#     # data_iter = iter(valid_dataloader)

#     # images, labels = next(data_iter)

#     # im_show(images, labels)



# if __name__ == '__main__':
#     main()

