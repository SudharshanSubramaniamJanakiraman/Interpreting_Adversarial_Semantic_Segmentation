# Import Necessary Packages
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torchvision.models import resnet50, resnet101

from typing import Optional

from torchsummary import summary
import torch 
# import pytorch_lightning as pl 
from torchsummary import summary

    
class PyramidPoolingModule(nn.Module):
    
    def __init__(self, in_channels, out_channels, bins) -> None:
        super(PyramidPoolingModule, self).__init__()
        
        self.ppm_layers = []
        for b in bins:
            self.ppm_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(b, b)),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.ppm_layers = nn.ModuleList(self.ppm_layers)
        
    def forward(self, x):
        
        input_size = x.shape[-2:]
        
        out = [x]
        for layer in self.ppm_layers:
            out.append(F.interpolate(layer(x), size=input_size, mode='bilinear', align_corners=True))
        
        out = torch.cat(out, 1)
        
        return out
        
        

class PSPNet(nn.Module):
    ''' Pyramid Scene Parsing Networrk '''
    ''' https://arxiv.org/pdf/1612.01105.pdf '''
    ''' Code borrowed from below links'''
    ''' https://github.com/hszhao/semseg/blob/master/model/pspnet.py '''
    ''' https://github.com/leaderj1001/PSPNet/blob/master/model.py '''
    ''' https://github.com/Lextal/pspnet-pytorch/blob/master/pspnet.py '''
    ''' https://github.com/hszhao/semseg '''
    ''' https://github.com/IanTaehoonYoo/semantic-segmentation-pytorch/blob/master/segmentation/models/pspnet.py '''
    
    def __init__(self, pretained:str="resnet50", bins:tuple=(1, 2, 3, 6), num_classes:int=19) -> None:
        super(PSPNet, self).__init__()
        
        if pretained == "resnet50":
            resnet_backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        if pretained == "resnet101":
            resnet_backbone = resnet101(pretrained=True, replace_stride_with_dilation=[False, True, True])
            
        feature_map_layers = list(resnet_backbone.children())[:4]
        self.feature_ectractor = nn.Sequential(*feature_map_layers)
        self.first_layer = resnet_backbone.layer1
        self.second_layer = resnet_backbone.layer2
        self.third_layer = resnet_backbone.layer3
        self.fourth_layer = resnet_backbone.layer4
        
        pyramid_pooling_in_channels = int(resnet_backbone.fc.in_features)
        
        self.ppm = PyramidPoolingModule(pyramid_pooling_in_channels, 512, bins)
        
        self.main_clf_head = nn.Sequential(
            nn.Conv2d(in_channels=pyramid_pooling_in_channels*2, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
        self.pred_branch = nn.Sequential(self.ppm, self.main_clf_head)
        
        if self.training:
            self.aux_branch = nn.Sequential(
                nn.Conv2d(int(pyramid_pooling_in_channels / 2), 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True), 
                nn.Dropout(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )
        
        self.init_weights_()
            
    def init_weights_(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

                
        
    def forward(self, x):
        input_Size = x.shape[-2:]
        
        out = x
        out = self.feature_ectractor(out)
        out = self.first_layer(out)
        out = self.second_layer(out)
        out_aux = self.third_layer(out)
        out = self.fourth_layer(out_aux)
        
        out = self.pred_branch(out)
        out = F.interpolate(out, size=input_Size, mode='bilinear', align_corners=True)
        
        # print(out.shape)
        
        if self.training:
            out_aux = self.aux_branch(out_aux)
            out_aux = F.interpolate(out_aux, size=input_Size, mode='bilinear', align_corners=True)
            return [out, out_aux]
        
        return [out]
        

def main():
    model = PSPNet()
    model.train()
    summary(model, (3, 1024, 512))
    
if __name__ == '__main__':
    main()



