import torch
import torch.nn.functional as F
from torch import nn

from utils import initialize_weights

class _EncoderBlock(nn.Module):
    '''
    A module (block) with the following structure:
    Conv2d (in,  out) + Batch Norm + ReLU,
    Conv2d (out, out) + Batch Norm + ReLU,
    Dropout (optional),
    Max Pooling (2x2) [halves the input spatial dimesions]
    '''

    def __init__(self, in_channels, out_channels, dropout=False):
        
        super(_EncoderBlock, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if dropout:
            
            layers.append(nn.Dropout())
            
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.encode = nn.Sequential(*layers)
    
    def forward(self, x):
        
        return self.encode(x)

class _DecoderBlock(nn.Module):
    '''
    A module (block) with the following structure:
    Dropout
    Conv2d (in,  mid) + Batch Norm + ReLU,
    Conv2d (mid, out) + Batch Norm + ReLU,
    Conv2d Transposed (2x2) [doubles the input spatial dimesions]
    '''

    def __init__(self, in_channels, middle_channels, out_channels):
        
        super(_DecoderBlock, self).__init__()
        
        self.decode = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)
        )
    
    def forward(self, x):
        
        return self.decode(x)


class UNet(nn.Module):

    def __init__(self, input_channels, mid_channels=16, output_channels=2, final=True):
        
        super(UNet, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.has_final = final
        
        # Encoder Part
        self.enc1 = _EncoderBlock(input_channels, mid_channels * 1)
        self.enc2 = _EncoderBlock(mid_channels * 1, mid_channels * 2)
        self.enc3 = _EncoderBlock(mid_channels * 2, mid_channels * 4, dropout=True)
        self.enc4 = _EncoderBlock(mid_channels * 4, mid_channels * 8, dropout=True)
        self.enc5 = _EncoderBlock(mid_channels * 8, mid_channels * 16, dropout=True)
        
        self.center = _DecoderBlock(mid_channels * 16, mid_channels * 32, mid_channels * 16)
        
        # Decoder Part
        self.dec5 = _DecoderBlock(mid_channels * 32, mid_channels * 16, mid_channels * 8)
        self.dec4 = _DecoderBlock(mid_channels * 16, mid_channels * 8, mid_channels * 4)
        self.dec3 = _DecoderBlock(mid_channels * 8, mid_channels * 4, mid_channels * 2)
        self.dec2 = _DecoderBlock(mid_channels * 4, mid_channels * 2, mid_channels * 1)

        self.dec1 = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(mid_channels * 2, mid_channels * 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels * 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 1, mid_channels * 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels * 1),
            nn.ReLU(inplace=True),
        )
                    
        
        self.final_input_channels = mid_channels * 1
        if self.has_final:
            self.final = nn.Conv2d(self.final_input_channels, output_channels, kernel_size=1)
        
        initialize_weights(self)
    
    def initialize(self):
        
        initialize_weights(self)
    
    def fwd_backbone(self, x):
        
        ####################################################################################################
        # Encoders. ########################################################################################
        ####################################################################################################
        
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        ####################################################################################################
        # Decoders. ########################################################################################
        ####################################################################################################
        
        cntr = self.center(enc5)
        
        dec5 = self.dec5(torch.cat([cntr, F.interpolate(enc5, cntr.size()[2:], mode='bilinear')], 1))
        dec4 = self.dec4(torch.cat([dec5, F.interpolate(enc4, dec5.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
    
        activ = dec1
                    
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
        