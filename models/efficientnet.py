import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights

import efficientnet_pytorch

class EfficientLab(nn.Module):

    def __init__(self, input_channels, output_channels=2, final=True):
        
        super(EfficientLab, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.has_final = final
        
        block_args = [
            'r1_k3_s11_e1_i16_o20_se0.125',  # Stage 2
            'r1_k3_s22_e6_i20_o40_se0.125',  # Stage 3
            'r2_k5_s11_e6_i40_o56_se0.125',  # Stage 4
            'r2_k3_s22_e6_i56_o96_se0.125',  # Stage 5
            'r3_k5_s11_e6_i96_o160_se0.125', # Stage 6
        ]
        block_args = efficientnet_pytorch.BlockDecoder.decode(block_args)
        global_params = efficientnet_pytorch.GlobalParams(width_coefficient=1.0, depth_coefficient=1.0, image_size=128, dropout_rate=0.2, num_classes=2, batch_norm_momentum=0.99, batch_norm_epsilon=0.001, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, include_top=True)
        
        backbone = efficientnet_pytorch.EfficientNet(block_args, global_params)
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
        )
        self.stage2 = backbone._blocks[0]
        self.stage3 = nn.Sequential(*backbone._blocks[1:3])
        self.stage4 = nn.Sequential(*backbone._blocks[3:5])
        self.stage5 = nn.Sequential(*backbone._blocks[5:8])
        self.stage6 = nn.Sequential(*backbone._blocks[8:11])
        
        self.swish = backbone._swish
                    
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
        
        act1 = self.stage1(x)
        act1 = self.swish(act1)
        
        act2 = self.stage2(act1)
        act2 = self.swish(act2)
        
        act3 = self.stage3(act2)
        act3 = self.swish(act3)
        
        act4 = self.stage4(act3)
        act4 = self.swish(act4)
        
        act5 = self.stage5(act4)
        act5 = self.swish(act5)
        
        act6 = self.stage6(act5)
        act6 = self.swish(act6)
            
        activ = F.interpolate(act6, x.size()[2:], mode='bilinear')
        
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
        