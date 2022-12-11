import pytorch_lightning as pl
from pytorch_lightning.core import LightningDataModule
from dataset import BDD100kSS, BDD100kSSADV
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader

class BDD100kTrainSemSegDataModule(LightningDataModule):

    def __init__(self, dataset:str="BDD100kSS", batch_size:int=128) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage):
        if self.dataset == "BDD100kSS":
            train_data = BDD100kSS("train")
            val_data = BDD100kSS("val")

            self.train_dl = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
            self.valid_dl = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def train_dataloader(self):
        loader = {}
        loader['0'] = self.train_dl
        loaders = CombinedLoader(loader, mode='max_size_cycle')
        return loaders

    def val_dataloader(self):
        loader = {}
        loader['0'] = self.valid_dl
        loaders = CombinedLoader(loader, mode='max_size_cycle')
        return loaders


class BDD100kEvalSemSegDataModule(LightningDataModule):

    def __init__(self, dataset:str="BDD100kSS", batch_size:int=1) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage):
        if self.dataset == "BDD100kSS":
            test_data = BDD100kSS("val")
            pred_data = BDD100kSS("test")

            self.test_dl = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            self.pred_dl = DataLoader(pred_data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def test_dataloader(self):
        loader = {}
        loader['0'] = self.test_dl
        loaders = CombinedLoader(loader, mode='max_size_cycle')
        return loaders

    def predict_dataloader(self):
        loader = {}
        loader['0'] = self.pred_dl
        loaders = CombinedLoader(loader, mode='max_size_cycle')
        return loaders

class BDD100kADVEvalSemSegDataModule(LightningDataModule):

    def __init__(self, dataset:str="BDD100kSSADV", batch_size:int=1, adv_location:str='GN_8') -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.adv_location = adv_location

    def setup(self, stage):
        if self.dataset == "BDD100kSSADV":
            test_data = BDD100kSSADV("val", adv_location=self.adv_location)
            pred_data = BDD100kSSADV("val", adv_location=self.adv_location)

            self.test_dl = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
            self.pred_dl = DataLoader(pred_data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def test_dataloader(self):
        loader = {}
        loader['0'] = self.test_dl
        loaders = CombinedLoader(loader, mode='max_size_cycle')
        return loaders

    def predict_dataloader(self):
        loader = {}
        loader['0'] = self.pred_dl
        loaders = CombinedLoader(loader, mode='max_size_cycle')
        return loaders


