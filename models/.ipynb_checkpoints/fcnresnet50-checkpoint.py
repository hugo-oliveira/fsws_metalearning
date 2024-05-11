import torch
import torch.nn.functional as F
from torch import nn

from torchvision import models

from utils import initialize_weights

import learn2learn as l2l

from models.crf import *
from models.attention import *

class FCNResNet50(nn.Module):

    def __init__(self, input_channels, output_channels=2, final=True, aggregate=None, has_att=False, has_crf=False, pretraining=None):
        
        super(FCNResNet50, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.has_final = final
        self.aggregate = aggregate
        self.has_att = has_att
        self.has_crf = has_crf
        
        backbone = models.resnet50()
        del backbone.fc
        del backbone.avgpool
        self.backbone = nn.Sequential(*list(backbone.children()))
        self.backbone[0] = nn.Conv2d(input_channels, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # self.backbone[0] = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if pretraining is not None:

            
            state_dict = torch.load('/scratch/oliveirahugo/lightly/src/med/ckpt/%s_snapshot.pth' % pretraining)['MODEL_STATE']
            for key in list(state_dict.keys()):
                if 'backbone.' in key:
                    state_dict[key.replace('backbone.', '')] = state_dict.pop(key)
                else:
                    del state_dict[key]
            
            self.backbone.load_state_dict(state_dict)
            # if pretraining == 'swav_queue':
                
            #     state_dict = torch.load('/scratch/oliveirahugo/lightly/src/med/ckpt/swav_queue_snapshot.pth')['MODEL_STATE']
            #     for key in list(state_dict.keys()):
            #         if 'backbone.' in key:
            #             state_dict[key.replace('backbone.', '')] = state_dict.pop(key)
            #         else:
            #             del state_dict[key]
                
            #     self.backbone.load_state_dict(state_dict)
            
            # elif pretraining == 'swav':
                
            #     state_dict = torch.load('/scratch/oliveirahugo/lightly/src/med/ckpt/swav_snapshot.pth')['MODEL_STATE']
            #     for key in list(state_dict.keys()):
            #         if 'backbone.' in key:
            #             state_dict[key.replace('backbone.', '')] = state_dict.pop(key)
            #         else:
            #             del state_dict[key]
                
            #     self.backbone.load_state_dict(state_dict)
                
            # elif pretraining == 'simsiam':
                
            #     state_dict = torch.load('/scratch/oliveirahugo/lightly/src/med/ckpt/simsiam_snapshot.pth')['MODEL_STATE']
            #     for key in list(state_dict.keys()):
            #         if 'backbone.' in key:
            #             state_dict[key.replace('backbone.', '')] = state_dict.pop(key)
            #         else:
            #             del state_dict[key]
                
            #     self.backbone.load_state_dict(state_dict)
                
            # elif pretraining == 'fastsiam':
                
            #     state_dict = torch.load('/scratch/oliveirahugo/lightly/src/med/ckpt/fastsiam_snapshot.pth')['MODEL_STATE']
            #     for key in list(state_dict.keys()):
            #         if 'backbone.' in key:
            #             state_dict[key.replace('backbone.', '')] = state_dict.pop(key)
            #         else:
            #             del state_dict[key]
                
            #     self.backbone.load_state_dict(state_dict)
                
            # elif pretraining == 'barlow_twins':
                
            #     state_dict = torch.load('/scratch/oliveirahugo/lightly/src/med/ckpt/barlow_twins_snapshot.pth')['MODEL_STATE']
            #     for key in list(state_dict.keys()):
            #         if 'backbone.' in key:
            #             state_dict[key.replace('backbone.', '')] = state_dict.pop(key)
            #         else:
            #             del state_dict[key]
                
            #     self.backbone.load_state_dict(state_dict)
        
        if self.has_att:
            self.attn1 = GridAttentionBlock2D_TORR(in_channels=256,
                                                   gating_channels=256,
                                                   inter_channels=256)
            self.attn2 = GridAttentionBlock2D_TORR(in_channels=512,
                                                   gating_channels=512,
                                                   inter_channels=512)
            self.attn3 = GridAttentionBlock2D_TORR(in_channels=1024,
                                                   gating_channels=1024,
                                                   inter_channels=1024)
            self.attn4 = GridAttentionBlock2D_TORR(in_channels=2048,
                                                   gating_channels=2048,
                                                   inter_channels=2048)
        
        if self.aggregate is None:
            self.final_input_channels = 2048
            if self.has_final:
                self.final = nn.Conv2d(self.final_input_channels, output_channels, kernel_size=1)
        elif self.aggregate == 'concat':
            self.final_input_channels = 256 + 512 + 1024 + 2048
            if self.has_final:
                self.final = nn.Conv2d(self.final_input_channels, output_channels, kernel_size=1)
        elif self.aggregate == 'deepsup':
            self.final_input_channels = [256,
                                         256 + 512,
                                         256 + 1024,
                                         256 + 2048]
            if self.has_final:
                self.final1 = nn.Conv2d(self.final_input_channels[0], output_channels, kernel_size=1)
                self.final2 = nn.Conv2d(self.final_input_channels[1], output_channels, kernel_size=1)
                self.final3 = nn.Conv2d(self.final_input_channels[2], output_channels, kernel_size=1)
                self.final4 = nn.Conv2d(self.final_input_channels[3], output_channels, kernel_size=1)
        
        if self.has_crf and self.has_final:
            self.crf = CRF(output_channels, filter_size=5, n_iter=5)
        
    def fwd_backbone(self, x):
        
        ####################################################################################################
        # Forwarding through backbone. #####################################################################
        ####################################################################################################
        
        # res0 = self.backbone.conv1(x)
        # res0 = self.backbone.bn1(res0)
        # res0 = self.backbone.relu(res0)
        
        # res1 = self.backbone.layer1(res0)
        # res2 = self.backbone.layer2(res1)
        # res3 = self.backbone.layer3(res2)
        # res4 = self.backbone.layer4(res3)
        res0 = self.backbone[0](x)
        res0 = self.backbone[1](res0)
        res0 = self.backbone[2](res0)
        
        res1 = self.backbone[4](res0)
        res2 = self.backbone[5](res1)
        res3 = self.backbone[6](res2)
        res4 = self.backbone[7](res3)
        
        if self.has_att:
            act1, out1 = self.attn1(res1, res4)
            act2, out2 = self.attn2(res2, res4)
            act3, out3 = self.attn3(res3, res4)
            act4, out4 = self.attn4(res4, res4)
        
        # Aggregating activations.
        if self.aggregate is None:
            activ = F.interpolate(res4, x.size()[2:], mode='bilinear')
        elif self.aggregate == 'concat':
            activ = torch.cat([F.interpolate(res1, x.size()[2:], mode='bilinear'),
                               F.interpolate(res2, x.size()[2:], mode='bilinear'),
                               F.interpolate(res3, x.size()[2:], mode='bilinear'),
                               F.interpolate(res4, x.size()[2:], mode='bilinear')], 1)
        elif self.aggregate == 'deepsup':
            res1_interp = F.interpolate(res1, x.size()[2:], mode='bilinear')
            activ1 = res1_interp
            activ2 = torch.cat([F.interpolate(res2, x.size()[2:], mode='bilinear'),
                                res1_interp], 1)
            activ3 = torch.cat([F.interpolate(res3, x.size()[2:], mode='bilinear'),
                                res1_interp], 1)
            activ4 = torch.cat([F.interpolate(res4, x.size()[2:], mode='bilinear'),
                                res1_interp], 1)
            
            activ = (activ1, activ2, activ3, activ4)
        
        return activ
    
    def fwd_classifier(self, x, activ, ignore_final=False):
        
        ####################################################################################################
        # Inference. #######################################################################################
        ####################################################################################################
        
        if self.aggregate is None:
#             activ = F.interpolate(res4, x.size()[2:], mode='bilinear')
            if self.has_final and not ignore_final:
                final = self.final(activ)
            else:
                final = activ
        elif self.aggregate == 'concat':
#             activ = torch.cat([F.interpolate(res1, x.size()[2:], mode='bilinear'),
#                                F.interpolate(res2, x.size()[2:], mode='bilinear'),
#                                F.interpolate(res3, x.size()[2:], mode='bilinear'),
#                                F.interpolate(res4, x.size()[2:], mode='bilinear')], 1)
            if self.has_final and not ignore_final:
                final = self.final(activ)
            else:
                final = activ
        elif self.aggregate == 'deepsup':
            activ1, activ2, activ3, activ4 = activ
#             res1_interp = F.interpolate(res1, x.size()[2:], mode='bilinear')
#             activ1 = res1_interp
#             activ2 = torch.cat([F.interpolate(res2, x.size()[2:], mode='bilinear'),
#                                 res1_interp], 1)
#             activ3 = torch.cat([F.interpolate(res3, x.size()[2:], mode='bilinear'),
#                                 res1_interp], 1)
#             activ4 = torch.cat([F.interpolate(res4, x.size()[2:], mode='bilinear'),
#                                 res1_interp], 1)
            if self.has_final and not ignore_final:
                final1 = self.final1(activ1)
                final2 = self.final2(activ2)
                final3 = self.final3(activ3)
                final4 = self.final4(activ4)
                if self.has_crf:
                    final1 = self.crf(final1)
                    final2 = self.crf(final2)
                    final3 = self.crf(final3)
                    final4 = self.crf(final4)
            else:
                final1 = activ1
                final2 = activ2
                final3 = activ3
                final4 = activ4
            final = [final1, final2, final3, final4]
            
        return final
    
    def forward(self, x, ignore_final=False):
        
        activ = self.fwd_backbone(x)
        
        return self.fwd_classifier(x, activ, ignore_final)
        