import wandb 

from lightning import PLSegmentor

import argparse

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import getpass

from pspnet import PSPNet

from utils import get_device

from datamodule import BDD100kTrainSemSegDataModule, BDD100kEvalSemSegDataModule, BDD100kADVEvalSemSegDataModule

def main():

    print("\n")
    print("Receiving Input Arguements From The User")
    print("\n")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default="PSPNET",
        help="Semantic Segmentation Model")
    parser.add_argument(
        '--pre_trained', type=str, default="resnet50",
        help="Backbone for PSPNet Segmentation Model")
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help="Learning Rate")
    parser.add_argument(
        '--train_batch_size', type=int, default=8,
        help="batch_size")
    parser.add_argument(
        '--test_batch_size', type=int, default=1,
        help="batch_size")

    print("\n")
    print(" Arguements Loaded Successfully ")
    print("\n")
    args = parser.parse_args()

    model = args.model 
    backbone = args.pre_trained
    lr = args.lr
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size

    relative_ckpt_folder = 'ckpt'
    if not os.path.exists(relative_ckpt_folder):
        os.mkdir(relative_ckpt_folder)

    ckpt_folder = f'BDD100k_SemSeg_{model}_BackBone_{backbone}_LR_{lr}'

    ckpt_path = os.path.join(relative_ckpt_folder, ckpt_folder)
    assert model == 'PSPNET','Unsupported Model Name'
    accelerator=get_device()
    print("\n")
    print(" Pytorch Model Creation ")
    print("\n")
    pspnet = PSPNet(pretained=backbone)
    print("\n")
    print(" PL Model Creation ")
    print("\n")
    segmentor = PLSegmentor(model=pspnet, lr=lr, accelerator=accelerator)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=True,
        every_n_epochs=1,
        monitor="val_loss",
        mode="min",
        dirpath=ckpt_path,
    )

    exp_name = f"{ckpt_folder}"

    wandb_logger = pl_loggers.WandbLogger(
        project=f"SemSeg_BDD100k", name=f"{exp_name}",
        id=f"{exp_name}", config=args.__dict__, save_dir=os.path.join("wandb", exp_name),resume='allow')

    
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=30, check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger)

    print("\n")
    print(" Loading Train Data Module ")
    print("\n")
    train_dm = BDD100kTrainSemSegDataModule("BDD100kSS", batch_size=train_batch_size)  
    print("\n")
    print(" Loading EVAL Data Module ")
    print("\n")
    test_dm = BDD100kEvalSemSegDataModule("BDD100kSS",batch_size=test_batch_size)
    test_adv_dm = BDD100kADVEvalSemSegDataModule("BDD100kSSADV", batch_size=test_batch_size, adv_location='GN_32')

    print("Fitting the Segmentation model ...")
    last_ckpt_path = os.path.join(ckpt_path, 'last.ckpt')
    print(' Last Known Checkpoint Location ... ')
    print(last_ckpt_path)

    trainer.fit(
        segmentor, train_dm, ckpt_path=last_ckpt_path
        if os.path.exists(last_ckpt_path) else None)

    print(f"Loading the best model from {checkpoint_callback.best_model_path}")

    best_segmentor = PLSegmentor.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path, strict=True, model=pspnet, lr=lr, accelerator=accelerator)

    best_segmentor.metric_prefix = "Benign:"
    trainer.test(best_segmentor, test_dm)

    best_segmentor.metric_prefix = "Adv:GN_32"
    trainer.test(best_segmentor, test_adv_dm)

    pytorch_model = best_segmentor.model

    torch.save(pytorch_model.state_dict(),os.path.join(ckpt_path, "pytorch_model.pth.tar"))

if __name__ == '__main__':
    main()


