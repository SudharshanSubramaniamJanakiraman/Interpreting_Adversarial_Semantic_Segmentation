import pytorch_lightning as pl 

import torch
import torch.nn as nn

from models.pspnet import PSPNet

from loss import PSPLoss

from torchmetrics import JaccardIndex

class PLSegmentor(pl.LightningModule):

    def __init__(self, model, metric_prefix:str="", lr:float=1e-4, accelerator='cpu') -> None:
        super().__init__()

        assert isinstance(model, nn.Module) == True, 'Invalid Model, pass a Pytorch Model'

        self.model = model
        self.metric_prefix = metric_prefix 
        self.learning_rate = lr 
        self.criterion = PSPLoss(0.4, accelerator)
        

    def forward(self, batch):
        x, _ = batch['0']
        z = self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_loss_(self, batch, mode:str='train'):

        x, y = batch['0']
        z = self.model(x)

        loss = self.criterion(z, y, mode)

        return loss

    def training_step(self, train_batch, batch_idx):
        
        train_loss = self.compute_loss_(train_batch, 'train')
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        val_loss = self.compute_loss_(val_batch, 'val')
        # self.log("valid_loss", val_loss)
        return val_loss

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.as_tensor(outputs))
        self.log('val_loss', val_loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch['0']
        z = self.model(x)
        return torch.softmax(z[0], dim=1), y

    def test_epoch_end(self, outputs):

        IOU = JaccardIndex(num_classes=19, average='weighted', ignore_index=255).to(self.device)
        mIOU = 0
        for idx, (z_softmax, y) in enumerate(outputs):
            z = outputs[0][0]
            t = outputs[0][1]
            t = z.int()
            mIOU += IOU(z, t)

        mIOU /= 1000
            
        self.log(self.metric_prefix+"IOU", mIOU)


    def predict_step(self, batch, batch_idx, dataloader_idx):

        x, _ = batch['0']
        z = self.model(x)
        return torch.argmax(torch.softmax(z, dim=1))







