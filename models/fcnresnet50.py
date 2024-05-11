import os
import torch
import torch.nn.functional as F
from torch import nn

from torchvision import models

from utils import initialize_weights

import learn2learn as l2l

class FCNResNet50(nn.Module):

    def __init__(self, input_channels, output_channels=2, final=True, pretrained=None):
        '''
        ! pretrained should be a torch loadable file with the MODEL_STATE of a pretrained model of this class
        '''
        super(FCNResNet50, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.has_final = final
        
        backbone = models.resnet50()
        del backbone.fc
        del backbone.avgpool
        self.backbone = nn.Sequential(*list(backbone.children()))
        self.backbone[0] = nn.Conv2d(input_channels, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)

        if pretrained is not None:
            assert os.path.exists(pretrained), f"The pretrained file `{pretrained}` does not exist."

            state_dict = torch.load(pretrained)['MODEL_STATE']
            # Example of pretrained value: '/scratch/oliveirahugo/lightly/src/med/ckpt/simsiam_snapshot.pth'

            # Deleting unused weights
            for key in list(state_dict.keys()):
                if 'backbone.' in key:
                    state_dict[key.replace('backbone.', '')] = state_dict.pop(key)
                else:
                    del state_dict[key]
            
            self.backbone.load_state_dict(state_dict)
        
        self.final_input_channels = 2048
        if self.has_final:
            self.final = nn.Conv2d(self.final_input_channels, output_channels, kernel_size=1)
        
    def fwd_backbone(self, x):
        
        ####################################################################################################
        # Forwarding through backbone. #####################################################################
        ####################################################################################################
        
        res0 = self.backbone[0](x)
        res0 = self.backbone[1](res0)
        res0 = self.backbone[2](res0)
        
        res1 = self.backbone[4](res0)
        res2 = self.backbone[5](res1)
        res3 = self.backbone[6](res2)
        res4 = self.backbone[7](res3)
        
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
        