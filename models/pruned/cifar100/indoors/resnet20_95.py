import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv1 = nn.Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_0_bn1 = nn.BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv2 = nn.Conv2d(6, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_0_bn2 = nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv1 = nn.Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_1_bn1 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv2 = nn.Conv2d(10, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_1_bn2 = nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv1 = nn.Conv2d(3, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_2_bn1 = nn.BatchNorm2d(14, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv2 = nn.Conv2d(14, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_2_bn2 = nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv1 = nn.Conv2d(3, 31, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer2_0_bn1 = nn.BatchNorm2d(31, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv2 = nn.Conv2d(31, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_0_bn2 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_downsample_0 = nn.Conv2d(3, 7, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer2_0_downsample_1 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv1 = nn.Conv2d(7, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_1_bn1 = nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv2 = nn.Conv2d(18, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_1_bn2 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv1 = nn.Conv2d(7, 23, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_2_bn1 = nn.BatchNorm2d(23, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv2 = nn.Conv2d(23, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_2_bn2 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv1 = nn.Conv2d(7, 45, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer3_0_bn1 = nn.BatchNorm2d(45, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv2 = nn.Conv2d(45, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_0_bn2 = nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_downsample_0 = nn.Conv2d(7, 4, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer3_0_downsample_1 = nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv1 = nn.Conv2d(4, 46, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_1_bn1 = nn.BatchNorm2d(46, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv2 = nn.Conv2d(46, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_1_bn2 = nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv1 = nn.Conv2d(4, 34, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_2_bn1 = nn.BatchNorm2d(34, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv2 = nn.Conv2d(34, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_2_bn2 = nn.BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        self.fc = nn.Linear(in_features=4, out_features=100, bias=True)

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
