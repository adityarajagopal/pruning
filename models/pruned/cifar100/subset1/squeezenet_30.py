import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 81, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(81, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fire2_conv1 = nn.Conv2d(81, 15, kernel_size=(1, 1), stride=(1, 1))
        self.fire2_bn1 = nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire2_conv2 = nn.Conv2d(15, 64, kernel_size=(1, 1), stride=(1, 1))
        self.fire2_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire2_conv3 = nn.Conv2d(15, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fire2_bn3 = nn.BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire3_conv1 = nn.Conv2d(122, 16, kernel_size=(1, 1), stride=(1, 1))
        self.fire3_bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire3_conv2 = nn.Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
        self.fire3_bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire3_conv3 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fire3_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire4_conv1 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
        self.fire4_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire4_conv2 = nn.Conv2d(32, 111, kernel_size=(1, 1), stride=(1, 1))
        self.fire4_bn2 = nn.BatchNorm2d(111, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire4_conv3 = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fire4_bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fire5_conv1 = nn.Conv2d(239, 32, kernel_size=(1, 1), stride=(1, 1))
        self.fire5_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire5_conv2 = nn.Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        self.fire5_bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire5_conv3 = nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fire5_bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire6_conv1 = nn.Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
        self.fire6_bn1 = nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire6_conv2 = nn.Conv2d(48, 170, kernel_size=(1, 1), stride=(1, 1))
        self.fire6_bn2 = nn.BatchNorm2d(170, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire6_conv3 = nn.Conv2d(48, 164, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fire6_bn3 = nn.BatchNorm2d(164, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire7_conv1 = nn.Conv2d(334, 48, kernel_size=(1, 1), stride=(1, 1))
        self.fire7_bn1 = nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire7_conv2 = nn.Conv2d(48, 190, kernel_size=(1, 1), stride=(1, 1))
        self.fire7_bn2 = nn.BatchNorm2d(190, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire7_conv3 = nn.Conv2d(48, 190, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fire7_bn3 = nn.BatchNorm2d(190, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire8_conv1 = nn.Conv2d(380, 64, kernel_size=(1, 1), stride=(1, 1))
        self.fire8_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire8_conv2 = nn.Conv2d(64, 247, kernel_size=(1, 1), stride=(1, 1))
        self.fire8_bn2 = nn.BatchNorm2d(247, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire8_conv3 = nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fire8_bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.fire9_conv1 = nn.Conv2d(503, 64, kernel_size=(1, 1), stride=(1, 1))
        self.fire9_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire9_conv2 = nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
        self.fire9_bn2 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fire9_conv3 = nn.Conv2d(64, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fire9_bn3 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(4, 100, kernel_size=(1, 1), stride=(1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.fire2_conv1(x)
        x = self.fire2_bn1(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.fire2_conv2(x_0)
        x_0 = self.fire2_bn2(x_0)
        x_1 = x
        x_1 = self.fire2_conv3(x_1)
        x_1 = self.fire2_bn3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.fire3_conv1(x)
        x = self.fire3_bn1(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.fire3_conv2(x_0)
        x_0 = self.fire3_bn2(x_0)
        x_1 = x
        x_1 = self.fire3_conv3(x_1)
        x_1 = self.fire3_bn3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.fire4_conv1(x)
        x = self.fire4_bn1(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.fire4_conv2(x_0)
        x_0 = self.fire4_bn2(x_0)
        x_1 = x
        x_1 = self.fire4_conv3(x_1)
        x_1 = self.fire4_bn3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.fire5_conv1(x)
        x = self.fire5_bn1(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.fire5_conv2(x_0)
        x_0 = self.fire5_bn2(x_0)
        x_1 = x
        x_1 = self.fire5_conv3(x_1)
        x_1 = self.fire5_bn3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.fire6_conv1(x)
        x = self.fire6_bn1(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.fire6_conv2(x_0)
        x_0 = self.fire6_bn2(x_0)
        x_1 = x
        x_1 = self.fire6_conv3(x_1)
        x_1 = self.fire6_bn3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.fire7_conv1(x)
        x = self.fire7_bn1(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.fire7_conv2(x_0)
        x_0 = self.fire7_bn2(x_0)
        x_1 = x
        x_1 = self.fire7_conv3(x_1)
        x_1 = self.fire7_bn3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.fire8_conv1(x)
        x = self.fire8_bn1(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.fire8_conv2(x_0)
        x_0 = self.fire8_bn2(x_0)
        x_1 = x
        x_1 = self.fire8_conv3(x_1)
        x_1 = self.fire8_bn3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = self.fire9_conv1(x)
        x = self.fire9_bn1(x)
        x = F.relu(x)
        x_0 = x
        x_0 = self.fire9_conv2(x_0)
        x_0 = self.fire9_bn2(x_0)
        x_1 = x
        x_1 = self.fire9_conv3(x_1)
        x_1 = self.fire9_bn3(x_1)
        x = torch.cat([x_0,x_1], 1)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.softmax(x)
        return x.squeeze()

def squeezenet(**kwargs):
    return SqueezeNet(**kwargs)
