import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

def to_gpu(x):
    return x.cuda()

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    
    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        mask_var = self.get_mask()
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True
    
    def get_mask(self):
        return to_gpu(self.mask)
    
    def forward(self, x):
        if self.mask_flag == True:
            mask_var = self.get_mask()
            weight = self.weight * mask_var
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
        
        
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
    
    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        mask_var = self.get_mask()
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True
    
    def get_mask(self):
        return to_gpu(self.mask)
    
    def forward(self, x):
        if self.mask_flag == True:
            mask_var = self.get_mask()
            weight = self.weight * mask_var
            return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class GatedConv2d(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size, stride=1, padding=0, unprunedRatio=1.0, bias=True):
        super().__init__()

        # as we are performing bn immediately after, which removes the channel mean, the bias will never change 
        # so we don't need bias here as after each channel bn is performed
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size, stride=stride, padding=padding, bias=False) 
        
        # affine=False just performs normalisation as matrix - mean / sqrt(var + eps) and does not have 
        # BN trainable parameters gamma and beta
        self.bn = nn.BatchNorm2d(outChannels, affine=False)

        self.gate = nn.Linear(inChannels, outChannels)
        self.gate.reset_parameters()
        self.gate.bias.data.fill_(1.0)
        
        self.beta = torch.nn.Parameter(torch.ones(outChannels))
        self.unprunedRatio = unprunedRatio
        self.prunedChannelIdx = list(range(outChannels))

        # E_g_x is the expectation of the l1_norm of the gate activations over
        # the input channels
        self.register_buffer('E_g_x', torch.tensor(0.0))

    def forward(self, x): 
        ss = F.avg_pool2d(x, x.shape[2])
        # for each image in batch, the avg pool of each feature map
        tmp = self.gate(ss.view(x.shape[0], x.shape[1]))
        gates = F.relu(tmp)
        activationSum = torch.sum(gates, 1)
        self.E_g_x = torch.mean(activationSum) 
        
        beta = self.beta.repeat(x.shape[0], 1)
        if self.unprunedRatio < 1.0:
            numPrunedChannels = self.conv.out_channels - round(self.conv.out_channels * self.unprunedRatio)
            self.prunedChannelIdx = torch.topk(gates, numPrunedChannels, dim=1, largest=False)[1]
            gates.scatter_(1, self.prunedChannelIdx, 0)

        x = self.conv(x)
        x = self.bn(x) 
        x += beta.unsqueeze(2).unsqueeze(3)
        x = x.mul(gates.unsqueeze(2).unsqueeze(3))
        x = F.relu(x)

        return x
            







        
