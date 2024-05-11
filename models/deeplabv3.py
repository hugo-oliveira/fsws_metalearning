import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights

import efficientnet_pytorch

from models.crf import *
from models.attention import *

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX

class Hardsigmoid(nn.Module):
    @staticmethod
    def forward(x):
        x = (x / 6.0) + 0.5
        x[x < 0] = 0
        x[x > 1] = 1
        return x
#         return x * F.hardsigmoid(x)  # for torchscript and CoreML
#         return x * F.hardsigmoid(x)  # for torchscript and CoreML

class DeepLabV3(nn.Module):

    def __init__(self, input_channels, output_channels=2, final=True):
        
        super(DeepLabV3, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.has_final = final
        
        #########################################################################
        ### Selecting backbone layers and setting up acvtivation functions ######
        #########################################################################

        backbone = deeplabv3_mobilenet_v3_large().backbone
        backbone['0'][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        self.block1 = nn.Sequential(
            backbone['0'],
            backbone['1']
        )
        self.block2 = nn.Sequential(
            backbone['2'],
            backbone['3']
        )
        self.block3 = nn.Sequential(
            backbone['4'],
            backbone['5'],
            backbone['6']
        )
        self.block4 = nn.Sequential(
            backbone['7'],
            backbone['8'],
            backbone['9'],
            backbone['10'],
            backbone['11'],
            backbone['12'],
            backbone['13'],
            backbone['14'],
            backbone['15']
        )
        self.block1[0][2] = Hardswish()
        self.block4[0].block[0][2] = Hardswish()
        self.block4[0].block[1][2] = Hardswish()
        self.block4[1].block[0][2] = Hardswish()
        self.block4[1].block[1][2] = Hardswish()
        self.block4[2].block[0][2] = Hardswish()
        self.block4[2].block[1][2] = Hardswish()
        self.block4[3].block[0][2] = Hardswish()
        self.block4[3].block[1][2] = Hardswish()
        self.block4[4].block[0][2] = Hardswish()
        self.block4[4].block[1][2] = Hardswish()
        self.block4[5].block[0][2] = Hardswish()
        self.block4[5].block[1][2] = Hardswish()
        self.block4[6].block[0][2] = Hardswish()
        self.block4[6].block[1][2] = Hardswish()
        self.block4[7].block[0][2] = Hardswish()
        self.block4[7].block[1][2] = Hardswish()
        self.block4[8].block[0][2] = Hardswish()
        self.block4[8].block[1][2] = Hardswish()
        
        self.block3[0].block[2].scale_activation = Hardsigmoid()
        self.block3[1].block[2].scale_activation = Hardsigmoid()
        self.block3[2].block[2].scale_activation = Hardsigmoid()
        
        self.block4[4].block[2].scale_activation = Hardsigmoid()
        self.block4[5].block[2].scale_activation = Hardsigmoid()
        self.block4[6].block[2].scale_activation = Hardsigmoid()
        self.block4[7].block[2].scale_activation = Hardsigmoid()
        self.block4[8].block[2].scale_activation = Hardsigmoid()
        
        ################################
        ### Setting up last layer ######
        ################################

        self.final_input_channels = 160
        if self.has_final:
            self.final = nn.Conv2d(self.final_input_channels, output_channels, kernel_size=1)
        
        
        initialize_weights(self)
    
    def initialize(self):
        
        initialize_weights(self)
    
    def fwd_backbone(self, x):
        
        ####################################################################################################
        # Forwarding through backbone. #####################################################################
        ####################################################################################################
        
        res1 = self.block1(x)
        res2 = self.block2(res1)
        res3 = self.block3(res2)
        res4 = self.block4(res3)
            
        activ = F.interpolate(res4, x.size()[2:], mode='bilinear')

        return activ
    
    def fwd_classifier(self, x, activ, ignore_final=False):
        
        ####################################################################################################
        # Inference. #######################################################################################
        ####################################################################################################
        
        if self.has_final and not ignore_final:
            final = self.final(activ)
        else:
            final = activ

        return final
    
    def forward(self, x, ignore_final=False):
        
        activ = self.fwd_backbone(x)
        
        return self.fwd_classifier(x, activ, ignore_final)
        