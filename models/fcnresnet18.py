import os
import torch
import torch.nn.functional as F
from torch import nn

from torchvision import models

from utils import initialize_weights

import learn2learn as l2l

from models.crf import *
from models.attention import *

class FCNResNet18(nn.Module):

    def __init__(self, input_channels, output_channels=2, final=True, pretrained=None):
        '''
        pretrained should be None or a loadable torch file with the state_dict of a pretrained pytorch resnet 18 model
        '''

        super(FCNResNet18, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.has_final = final
        
        self.backbone = models.resnet18(num_classes=512, zero_init_residual=True)
        del self.backbone.fc
        self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        
        if pretrained is not None:
            assert os.path.exists(pretrained), f"The pretrained file `{pretrained}` does not exist."

            state_dict = torch.load(pretrained)['state_dict']
            # Example of value for pretrained: '/home/oliveirahugo/scratch/simsiam/ckpt/resnet18_rs/checkpoint_0040.pth.tar' 
            #                                  '/home/oliveirahugo/scratch/simsiam/ckpt/resnet18_med/checkpoint_0040.pth.tar'
        
            for key in list(state_dict.keys()):
                state_dict[key.replace('module.encoder.', '')] = state_dict.pop(key)

            # Deleting unused weights
            del state_dict['module.predictor.0.weight']
            del state_dict['module.predictor.1.weight']
            del state_dict['module.predictor.1.bias']
            del state_dict['module.predictor.1.running_mean']
            del state_dict['module.predictor.1.running_var']
            del state_dict['module.predictor.1.num_batches_tracked']
            del state_dict['module.predictor.3.weight']
            del state_dict['module.predictor.3.bias']

            del state_dict['fc.0.weight']
            del state_dict['fc.1.weight']
            del state_dict['fc.1.bias']
            del state_dict['fc.1.running_mean']
            del state_dict['fc.1.running_var']
            del state_dict['fc.1.num_batches_tracked']
            del state_dict['fc.3.weight']
            del state_dict['fc.4.weight']
            del state_dict['fc.4.bias']
            del state_dict['fc.4.running_mean']
            del state_dict['fc.4.running_var']
            del state_dict['fc.4.num_batches_tracked']
            del state_dict['fc.6.weight']
            del state_dict['fc.6.bias']
            del state_dict['fc.7.running_mean']
            del state_dict['fc.7.running_var']
            del state_dict['fc.7.num_batches_tracked']

            self.backbone.load_state_dict(state_dict)
        
        
        self.final_input_channels = 512
        if self.has_final:
            self.final = nn.Conv2d(self.final_input_channels, output_channels, kernel_size=1)
        
    def fwd_backbone(self, x):
        
        ####################################################################################################
        # Forwarding through backbone. #####################################################################
        ####################################################################################################
        
        res0 = self.backbone.conv1(x)
        res0 = self.backbone.bn1(res0)
        res0 = self.backbone.relu(res0)
        
        res1 = self.backbone.layer1(res0)
        res2 = self.backbone.layer2(res1)
        res3 = self.backbone.layer3(res2)
        res4 = self.backbone.layer4(res3)
                
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
        