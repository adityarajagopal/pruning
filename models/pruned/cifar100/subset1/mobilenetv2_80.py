import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 25, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_0_conv1 = nn.Conv2d(25, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_0_bn1 = nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_0_conv2 = nn.Conv2d(18, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=18, bias=False)
        self.layers_0_bn2 = nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_0_conv3 = nn.Conv2d(18, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_0_bn3 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_0_shortcut_0 = nn.Conv2d(25, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_0_shortcut_1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_1_conv1 = nn.Conv2d(16, 50, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_1_bn1 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_1_conv2 = nn.Conv2d(50, 50, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=50, bias=False)
        self.layers_1_bn2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_1_conv3 = nn.Conv2d(50, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_1_bn3 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_1_shortcut_0 = nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_1_shortcut_1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_2_conv1 = nn.Conv2d(16, 70, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_2_bn1 = nn.BatchNorm2d(70, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_2_conv2 = nn.Conv2d(70, 70, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=70, bias=False)
        self.layers_2_bn2 = nn.BatchNorm2d(70, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_2_conv3 = nn.Conv2d(70, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_2_bn3 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_3_conv1 = nn.Conv2d(16, 117, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_3_bn1 = nn.BatchNorm2d(117, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_3_conv2 = nn.Conv2d(117, 117, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=117, bias=False)
        self.layers_3_bn2 = nn.BatchNorm2d(117, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_3_conv3 = nn.Conv2d(117, 31, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_3_bn3 = nn.BatchNorm2d(31, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_4_conv1 = nn.Conv2d(31, 106, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_4_bn1 = nn.BatchNorm2d(106, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_4_conv2 = nn.Conv2d(106, 106, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=106, bias=False)
        self.layers_4_bn2 = nn.BatchNorm2d(106, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_4_conv3 = nn.Conv2d(106, 31, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_4_bn3 = nn.BatchNorm2d(31, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_5_conv1 = nn.Conv2d(31, 115, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_5_bn1 = nn.BatchNorm2d(115, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_5_conv2 = nn.Conv2d(115, 115, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=115, bias=False)
        self.layers_5_bn2 = nn.BatchNorm2d(115, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_5_conv3 = nn.Conv2d(115, 31, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_5_bn3 = nn.BatchNorm2d(31, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_6_conv1 = nn.Conv2d(31, 132, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_6_bn1 = nn.BatchNorm2d(132, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_6_conv2 = nn.Conv2d(132, 132, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=132, bias=False)
        self.layers_6_bn2 = nn.BatchNorm2d(132, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_6_conv3 = nn.Conv2d(132, 23, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_6_bn3 = nn.BatchNorm2d(23, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_7_conv1 = nn.Conv2d(23, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_7_bn1 = nn.BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_7_conv2 = nn.Conv2d(208, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=208, bias=False)
        self.layers_7_bn2 = nn.BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_7_conv3 = nn.Conv2d(208, 23, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_7_bn3 = nn.BatchNorm2d(23, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_8_conv1 = nn.Conv2d(23, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_bn1 = nn.BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_8_conv2 = nn.Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)
        self.layers_8_bn2 = nn.BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_8_conv3 = nn.Conv2d(184, 23, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_8_bn3 = nn.BatchNorm2d(23, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_9_conv1 = nn.Conv2d(23, 140, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_bn1 = nn.BatchNorm2d(140, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_9_conv2 = nn.Conv2d(140, 140, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=140, bias=False)
        self.layers_9_bn2 = nn.BatchNorm2d(140, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_9_conv3 = nn.Conv2d(140, 23, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_9_bn3 = nn.BatchNorm2d(23, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_10_conv1 = nn.Conv2d(23, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_bn1 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_10_conv2 = nn.Conv2d(200, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=200, bias=False)
        self.layers_10_bn2 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_10_conv3 = nn.Conv2d(200, 45, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_bn3 = nn.BatchNorm2d(45, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_10_shortcut_0 = nn.Conv2d(23, 45, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_10_shortcut_1 = nn.BatchNorm2d(45, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_11_conv1 = nn.Conv2d(45, 223, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_bn1 = nn.BatchNorm2d(223, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_11_conv2 = nn.Conv2d(223, 223, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=223, bias=False)
        self.layers_11_bn2 = nn.BatchNorm2d(223, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_11_conv3 = nn.Conv2d(223, 45, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_11_bn3 = nn.BatchNorm2d(45, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_12_conv1 = nn.Conv2d(45, 223, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_bn1 = nn.BatchNorm2d(223, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_12_conv2 = nn.Conv2d(223, 223, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=223, bias=False)
        self.layers_12_bn2 = nn.BatchNorm2d(223, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_12_conv3 = nn.Conv2d(223, 45, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_12_bn3 = nn.BatchNorm2d(45, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_13_conv1 = nn.Conv2d(45, 242, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_bn1 = nn.BatchNorm2d(242, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_13_conv2 = nn.Conv2d(242, 242, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=242, bias=False)
        self.layers_13_bn2 = nn.BatchNorm2d(242, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_13_conv3 = nn.Conv2d(242, 54, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_13_bn3 = nn.BatchNorm2d(54, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_14_conv1 = nn.Conv2d(54, 455, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_14_bn1 = nn.BatchNorm2d(455, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_14_conv2 = nn.Conv2d(455, 455, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=455, bias=False)
        self.layers_14_bn2 = nn.BatchNorm2d(455, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_14_conv3 = nn.Conv2d(455, 54, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_14_bn3 = nn.BatchNorm2d(54, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_15_conv1 = nn.Conv2d(54, 338, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_15_bn1 = nn.BatchNorm2d(338, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_15_conv2 = nn.Conv2d(338, 338, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=338, bias=False)
        self.layers_15_bn2 = nn.BatchNorm2d(338, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_15_conv3 = nn.Conv2d(338, 54, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_15_bn3 = nn.BatchNorm2d(54, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_16_conv1 = nn.Conv2d(54, 286, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_16_bn1 = nn.BatchNorm2d(286, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_16_conv2 = nn.Conv2d(286, 286, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=286, bias=False)
        self.layers_16_bn2 = nn.BatchNorm2d(286, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_16_conv3 = nn.Conv2d(286, 119, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_16_bn3 = nn.BatchNorm2d(119, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layers_16_shortcut_0 = nn.Conv2d(54, 119, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layers_16_shortcut_1 = nn.BatchNorm2d(119, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(119, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.linear = nn.Linear(in_features=960, out_features=100, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x_main = x
        x_main = self.layers_0_conv1(x_main)
        x_main = self.layers_0_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_0_conv2(x_main)
        x_main = self.layers_0_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_0_conv3(x_main)
        x_main = self.layers_0_bn3(x_main)
        x_residual = x
        x_residual = self.layers_0_shortcut_0(x_residual)
        x_residual = self.layers_0_shortcut_1(x_residual)
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_1_conv1(x_main)
        x_main = self.layers_1_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_1_conv2(x_main)
        x_main = self.layers_1_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_1_conv3(x_main)
        x_main = self.layers_1_bn3(x_main)
        x_residual = x
        x_residual = self.layers_1_shortcut_0(x_residual)
        x_residual = self.layers_1_shortcut_1(x_residual)
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_2_conv1(x_main)
        x_main = self.layers_2_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_2_conv2(x_main)
        x_main = self.layers_2_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_2_conv3(x_main)
        x_main = self.layers_2_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.layers_3_conv1(x)
        x = self.layers_3_bn1(x)
        x = F.relu(x)
        x = self.layers_3_conv2(x)
        x = self.layers_3_bn2(x)
        x = F.relu(x)
        x = self.layers_3_conv3(x)
        x = self.layers_3_bn3(x)
        x_main = x
        x_main = self.layers_4_conv1(x_main)
        x_main = self.layers_4_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_4_conv2(x_main)
        x_main = self.layers_4_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_4_conv3(x_main)
        x_main = self.layers_4_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_5_conv1(x_main)
        x_main = self.layers_5_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_5_conv2(x_main)
        x_main = self.layers_5_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_5_conv3(x_main)
        x_main = self.layers_5_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.layers_6_conv1(x)
        x = self.layers_6_bn1(x)
        x = F.relu(x)
        x = self.layers_6_conv2(x)
        x = self.layers_6_bn2(x)
        x = F.relu(x)
        x = self.layers_6_conv3(x)
        x = self.layers_6_bn3(x)
        x_main = x
        x_main = self.layers_7_conv1(x_main)
        x_main = self.layers_7_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_7_conv2(x_main)
        x_main = self.layers_7_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_7_conv3(x_main)
        x_main = self.layers_7_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_8_conv1(x_main)
        x_main = self.layers_8_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_8_conv2(x_main)
        x_main = self.layers_8_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_8_conv3(x_main)
        x_main = self.layers_8_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_9_conv1(x_main)
        x_main = self.layers_9_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_9_conv2(x_main)
        x_main = self.layers_9_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_9_conv3(x_main)
        x_main = self.layers_9_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_10_conv1(x_main)
        x_main = self.layers_10_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_10_conv2(x_main)
        x_main = self.layers_10_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_10_conv3(x_main)
        x_main = self.layers_10_bn3(x_main)
        x_residual = x
        x_residual = self.layers_10_shortcut_0(x_residual)
        x_residual = self.layers_10_shortcut_1(x_residual)
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_11_conv1(x_main)
        x_main = self.layers_11_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_11_conv2(x_main)
        x_main = self.layers_11_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_11_conv3(x_main)
        x_main = self.layers_11_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_12_conv1(x_main)
        x_main = self.layers_12_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_12_conv2(x_main)
        x_main = self.layers_12_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_12_conv3(x_main)
        x_main = self.layers_12_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x = self.layers_13_conv1(x)
        x = self.layers_13_bn1(x)
        x = F.relu(x)
        x = self.layers_13_conv2(x)
        x = self.layers_13_bn2(x)
        x = F.relu(x)
        x = self.layers_13_conv3(x)
        x = self.layers_13_bn3(x)
        x_main = x
        x_main = self.layers_14_conv1(x_main)
        x_main = self.layers_14_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_14_conv2(x_main)
        x_main = self.layers_14_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_14_conv3(x_main)
        x_main = self.layers_14_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_15_conv1(x_main)
        x_main = self.layers_15_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_15_conv2(x_main)
        x_main = self.layers_15_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_15_conv3(x_main)
        x_main = self.layers_15_bn3(x_main)
        x_residual = x
        x = x_main + x_residual
        x_main = x
        x_main = self.layers_16_conv1(x_main)
        x_main = self.layers_16_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_16_conv2(x_main)
        x_main = self.layers_16_bn2(x_main)
        x_main = F.relu(x_main)
        x_main = self.layers_16_conv3(x_main)
        x_main = self.layers_16_bn3(x_main)
        x_residual = x
        x_residual = self.layers_16_shortcut_0(x_residual)
        x_residual = self.layers_16_shortcut_1(x_residual)
        x = x_main + x_residual
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def mobilenetv2(**kwargs):
    return MobileNetV2(**kwargs)