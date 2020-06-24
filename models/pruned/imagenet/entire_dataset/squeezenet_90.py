import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 2, kernel_size=(7, 7), stride=(2, 2))
        self.features_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.features_3_squeeze = nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1))
        self.features_3_expand1x1 = nn.Conv2d(2, 64, kernel_size=(1, 1), stride=(1, 1))
        self.features_3_expand3x3 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_4_squeeze = nn.Conv2d(66, 2, kernel_size=(1, 1), stride=(1, 1))
        self.features_4_expand1x1 = nn.Conv2d(2, 64, kernel_size=(1, 1), stride=(1, 1))
        self.features_4_expand3x3 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_5_squeeze = nn.Conv2d(66, 2, kernel_size=(1, 1), stride=(1, 1))
        self.features_5_expand1x1 = nn.Conv2d(2, 127, kernel_size=(1, 1), stride=(1, 1))
        self.features_5_expand3x3 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_6 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.features_7_squeeze = nn.Conv2d(129, 2, kernel_size=(1, 1), stride=(1, 1))
        self.features_7_expand1x1 = nn.Conv2d(2, 128, kernel_size=(1, 1), stride=(1, 1))
        self.features_7_expand3x3 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_8_squeeze = nn.Conv2d(130, 6, kernel_size=(1, 1), stride=(1, 1))
        self.features_8_expand1x1 = nn.Conv2d(6, 186, kernel_size=(1, 1), stride=(1, 1))
        self.features_8_expand3x3 = nn.Conv2d(6, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_9_squeeze = nn.Conv2d(189, 2, kernel_size=(1, 1), stride=(1, 1))
        self.features_9_expand1x1 = nn.Conv2d(2, 184, kernel_size=(1, 1), stride=(1, 1))
        self.features_9_expand3x3 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_10_squeeze = nn.Conv2d(186, 3, kernel_size=(1, 1), stride=(1, 1))
        self.features_10_expand1x1 = nn.Conv2d(3, 157, kernel_size=(1, 1), stride=(1, 1))
        self.features_10_expand3x3 = nn.Conv2d(3, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features_11 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.features_12_squeeze = nn.Conv2d(159, 2, kernel_size=(1, 1), stride=(1, 1))
        self.features_12_expand1x1 = nn.Conv2d(2, 164, kernel_size=(1, 1), stride=(1, 1))
        self.features_12_expand3x3 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.final_conv = nn.Conv2d(166, 1000, kernel_size=(1, 1), stride=(1, 1))
        self.classifier_3 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.features_2(x)
        x = self.features_3_squeeze(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.features_3_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_3_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.features_4_squeeze(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.features_4_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_4_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.features_5_squeeze(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.features_5_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_5_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.features_6(x)
        x = self.features_7_squeeze(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.features_7_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_7_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.features_8_squeeze(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.features_8_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_8_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.features_9_squeeze(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.features_9_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_9_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.features_10_squeeze(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.features_10_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_10_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.features_11(x)
        x = self.features_12_squeeze(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.features_12_expand1x1(x_0)
        x_1 = x
        x_1 = self.features_12_expand3x3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.final_conv(x)
        x = F.relu(x)
        x = self.classifier_3(x)
        return x.squeeze(2).squeeze(2)

def squeezenet(**kwargs):
    return SqueezeNet(**kwargs)
