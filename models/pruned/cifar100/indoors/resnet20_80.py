import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv1 = nn.Conv2d(8, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_0_bn1 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_0_conv2 = nn.Conv2d(10, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_0_bn2 = nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv1 = nn.Conv2d(8, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_1_bn1 = nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_1_conv2 = nn.Conv2d(12, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_1_bn2 = nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv1 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_2_bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_2_conv2 = nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_2_bn2 = nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv1 = nn.Conv2d(8, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer2_0_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_conv2 = nn.Conv2d(32, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_0_bn2 = nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_0_downsample_0 = nn.Conv2d(8, 12, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer2_0_downsample_1 = nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv1 = nn.Conv2d(12, 25, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_1_bn1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_1_conv2 = nn.Conv2d(25, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_1_bn2 = nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv1 = nn.Conv2d(12, 31, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_2_bn1 = nn.BatchNorm2d(31, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_2_conv2 = nn.Conv2d(31, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_2_bn2 = nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv1 = nn.Conv2d(12, 57, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer3_0_bn1 = nn.BatchNorm2d(57, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_conv2 = nn.Conv2d(57, 13, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_0_bn2 = nn.BatchNorm2d(13, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_0_downsample_0 = nn.Conv2d(12, 13, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer3_0_downsample_1 = nn.BatchNorm2d(13, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv1 = nn.Conv2d(13, 52, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_1_bn1 = nn.BatchNorm2d(52, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_1_conv2 = nn.Conv2d(52, 13, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_1_bn2 = nn.BatchNorm2d(13, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv1 = nn.Conv2d(13, 42, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_2_bn1 = nn.BatchNorm2d(42, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_2_conv2 = nn.Conv2d(42, 13, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_2_bn2 = nn.BatchNorm2d(13, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        self.fc = nn.Linear(in_features=13, out_features=100, bias=True)

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
