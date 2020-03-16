import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv1 = nn.Conv2d(7, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_0_bn1 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv2 = nn.Conv2d(10, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_0_bn2 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv1 = nn.Conv2d(7, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_1_bn1 = nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv2 = nn.Conv2d(12, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_1_bn2 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv1 = nn.Conv2d(7, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_2_bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv2 = nn.Conv2d(16, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_2_bn2 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv1 = nn.Conv2d(7, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer2_0_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv2 = nn.Conv2d(32, 19, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_0_bn2 = nn.BatchNorm2d(19, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_downsample_0 = nn.Conv2d(7, 19, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer2_0_downsample_1 = nn.BatchNorm2d(19, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv1 = nn.Conv2d(19, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_1_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv2 = nn.Conv2d(32, 19, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_1_bn2 = nn.BatchNorm2d(19, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv1 = nn.Conv2d(19, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_2_bn1 = nn.BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv2 = nn.Conv2d(30, 19, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_2_bn2 = nn.BatchNorm2d(19, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv1 = nn.Conv2d(19, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer3_0_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv2 = nn.Conv2d(64, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_0_bn2 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_downsample_0 = nn.Conv2d(19, 26, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer3_0_downsample_1 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv1 = nn.Conv2d(26, 61, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_1_bn1 = nn.BatchNorm2d(61, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv2 = nn.Conv2d(61, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_1_bn2 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv1 = nn.Conv2d(26, 47, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_2_bn1 = nn.BatchNorm2d(47, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv2 = nn.Conv2d(47, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_2_bn2 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        self.fc = nn.Linear(in_features=26, out_features=100, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x_main = x
        x_main = self.layer1_0_conv1(x_main)
        x_main = self.layer1_0_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layer1_0_conv2(x_main)
        x_main = self.layer1_0_bn2(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual)
        x_main = x
        x_main = self.layer1_1_conv1(x_main)
        x_main = self.layer1_1_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layer1_1_conv2(x_main)
        x_main = self.layer1_1_bn2(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual)
        x_main = x
        x_main = self.layer1_2_conv1(x_main)
        x_main = self.layer1_2_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layer1_2_conv2(x_main)
        x_main = self.layer1_2_bn2(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual)
        x_main = x
        x_main = self.layer2_0_conv1(x_main)
        x_main = self.layer2_0_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layer2_0_conv2(x_main)
        x_main = self.layer2_0_bn2(x_main)
        x_residual = x
        x_residual = self.layer2_0_downsample_0(x_residual)
        x_residual = self.layer2_0_downsample_1(x_residual)
        x = F.relu(x_main + x_residual)
        x_main = x
        x_main = self.layer2_1_conv1(x_main)
        x_main = self.layer2_1_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layer2_1_conv2(x_main)
        x_main = self.layer2_1_bn2(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual)
        x_main = x
        x_main = self.layer2_2_conv1(x_main)
        x_main = self.layer2_2_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layer2_2_conv2(x_main)
        x_main = self.layer2_2_bn2(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual)
        x_main = x
        x_main = self.layer3_0_conv1(x_main)
        x_main = self.layer3_0_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layer3_0_conv2(x_main)
        x_main = self.layer3_0_bn2(x_main)
        x_residual = x
        x_residual = self.layer3_0_downsample_0(x_residual)
        x_residual = self.layer3_0_downsample_1(x_residual)
        x = F.relu(x_main + x_residual)
        x_main = x
        x_main = self.layer3_1_conv1(x_main)
        x_main = self.layer3_1_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layer3_1_conv2(x_main)
        x_main = self.layer3_1_bn2(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual)
        x_main = x
        x_main = self.layer3_2_conv1(x_main)
        x_main = self.layer3_2_bn1(x_main)
        x_main = F.relu(x_main)
        x_main = self.layer3_2_conv2(x_main)
        x_main = self.layer3_2_bn2(x_main)
        x_residual = x
        x = F.relu(x_main + x_residual)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet20(**kwargs):
    return ResNet20(**kwargs)
