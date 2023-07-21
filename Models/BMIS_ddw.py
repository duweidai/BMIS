import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
from torch.nn.parameter import Parameter

from math import floor
from functools import partial



class PCDM(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1) 
        output = self.bn(output)
        return F.relu(output)


# ############################################## EACB and DACB ############################################################


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
        source: https://github.com/BangguWu/ECANet
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

      

class EACB(nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        
        self.eca = eca_layer(chann, k_size=5)
        

    def forward(self, input):
        
        
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        output = self.eca(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)

# ############################################## CSAM ############################################################

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, ratio):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_channels//ratio),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_channels//ratio),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out



class CSAM(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels, ratio=2):
        super(CSAM, self).__init__()
        print("ratio is: ", ratio)
        self.sab = SpatialAttentionBlock(in_channels, ratio)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        
        return out


# ############################################## UpsamplerBlock ############################################################

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


# ############################################## the proposed BMIS ############################################################


class BMIS(nn.Module):

    def __init__(self, image_size=(224, 224), classes=2, channels=3, encoder=None):  #use encoder to pass pretrained encoder
        super(BMIS, self).__init__()
        
        drop_1 = 0.03           # 0.03
        drop_2 = 0.3            # 0.3
        
        dim_1 = 64              # default 64
        dim_2 = 96              # 128 
        
        self.initial_block = PCDM(3,16)
        
        self.down_1 = PCDM(16,dim_1)
        
        self.encoder_1_1 = EACB(dim_1, drop_1, 1)
        self.encoder_1_2 = EACB(dim_1, drop_1, 1)
        self.encoder_1_3 = EACB(dim_1, drop_1, 1)
        self.encoder_1_4 = EACB(dim_1, drop_1, 1)
        self.encoder_1_5 = EACB(dim_1, drop_1, 1)

        self.down_2 = PCDM(dim_1,dim_2)
        
        self.encoder_2_1 = EACB(dim_2, drop_2, 2)
        self.encoder_2_2 = EACB(dim_2, drop_2, 3)
        self.encoder_2_3 = EACB(dim_2, drop_2, 4)
        self.encoder_2_4 = EACB(dim_2, drop_2, 5)
        
        self.affinity_attention = CSAM(dim_2, ratio=2)
        
        # start decoder #########################################
        
        self.up_1 = UpsamplerBlock(dim_2, dim_1)
        self.decoder_1_1 = EACB(dim_1, 0, 1)
        
        self.up_2 = UpsamplerBlock(dim_1,16)
        self.decoder_2_1 = EACB(16, 0, 1)
       
        self.output_conv = nn.ConvTranspose2d( 16, classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        
    def forward(self, input):
    
        e_0 = self.initial_block(input)           
        e_1 = self.down_1(e_0)                    
        
        e_2 = self.encoder_1_1(e_1)                 
        e_3 = self.encoder_1_2(e_2)                 
        e_4 = self.encoder_1_3(e_3)                 
        e_5 = self.encoder_1_4(e_4)                 
        e_6 = self.encoder_1_5(e_5)                 
       
        e_7 = self.down_2(e_6)                    
        
        e_8 = self.encoder_2_1(e_7)                 
        e_9 = self.encoder_2_2(e_8)
        e_10 = self.encoder_2_3(e_9)
        e_11 = self.encoder_2_4(e_10)
        
        attention = self.affinity_attention(e_11)
        attention_fuse = e_11 + attention     
        
        # start decoder #####################################################################
        
        d_1 = self.up_1(attention_fuse)            
        d_2 = self.decoder_1_1(d_1)                   
        d_3 = self.up_2(d_2)              
        d_4 = self.decoder_2_1(d_3)         
     
        logit = self.output_conv(d_4)     
        out = torch.sigmoid(logit)

        return out
        
        
if __name__ == '__main__':
    input = torch.rand(8, 3, 224, 224)
    model = BMIS()
    out = model(input)
    print(out.shape)
    print(torch.max(out))
    print(torch.min(out))