import torch 
import torch.nn as nn
import torch.nn.functional as F

class PSPLoss(nn.Module):

    def __init__(self, aux_weight:float=0.4, device='cpu') -> None:
        super().__init__()
        self.device=device
        self.aux_weight = 0.4
        
    def forward(self, pred, gt, mode:str='train'):
        gt = gt.type(torch.LongTensor).to(self.device)
        pred[0] = pred[0].to(self.device)
        loss = F.cross_entropy(pred[0], gt, weight=None, ignore_index=255)
        if mode == 'train':
            pred[1] = pred[1].to(self.device)
            aux_loss = F.cross_entropy(pred[1], gt, weight=None, ignore_index=255)
            loss += self.aux_weight * aux_loss
        return loss

