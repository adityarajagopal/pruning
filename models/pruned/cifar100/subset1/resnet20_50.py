import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet20(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.bn1 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_0_conv1 = nn.Conv2d(10, 15, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_0_bn1 = nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_0_conv2 = nn.Conv2d(15, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_0_bn2 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_1_conv1 = nn.Conv2d(10, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_1_bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_1_conv2 = nn.Conv2d(16, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_1_bn2 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_2_conv1 = nn.Conv2d(10, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_2_bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_2_conv2 = nn.Conv2d(16, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_2_bn2 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_0_conv1 = nn.Conv2d(10, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
		self.layer2_0_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_0_conv2 = nn.Conv2d(32, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer2_0_bn2 = nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_0_downsample_0 = nn.Conv2d(10, 18, kernel_size=(1, 1), stride=(2, 2), bias=False)
		self.layer2_0_downsample_1 = nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_1_conv1 = nn.Conv2d(18, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer2_1_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_1_conv2 = nn.Conv2d(32, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer2_1_bn2 = nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_2_conv1 = nn.Conv2d(18, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer2_2_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_2_conv2 = nn.Conv2d(32, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer2_2_bn2 = nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_0_conv1 = nn.Conv2d(18, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
		self.layer3_0_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_0_conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer3_0_bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_0_downsample_0 = nn.Conv2d(18, 32, kernel_size=(1, 1), stride=(2, 2), bias=False)
		self.layer3_0_downsample_1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_1_conv1 = nn.Conv2d(32, 59, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer3_1_bn1 = nn.BatchNorm2d(59, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_1_conv2 = nn.Conv2d(59, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer3_1_bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_2_conv1 = nn.Conv2d(32, 53, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer3_2_bn1 = nn.BatchNorm2d(53, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_2_conv2 = nn.Conv2d(53, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer3_2_bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
		self.fc = nn.Linear(in_features=32, out_features=100, bias=True)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.layer1_0_bn1(self.layer1_0_conv1(x)))
		out = self.layer1_0_bn2(self.layer1_0_conv2(out))
		x = F.relu(out + x)
		out = F.relu(self.layer1_1_bn1(self.layer1_1_conv1(x)))
		out = self.layer1_1_bn2(self.layer1_1_conv2(out))
		x = F.relu(out + x)
		out = F.relu(self.layer1_2_bn1(self.layer1_2_conv1(x)))
		out = self.layer1_2_bn2(self.layer1_2_conv2(out))
		x = F.relu(out + x)
		out = F.relu(self.layer2_0_bn1(self.layer2_0_conv1(x)))
		out = self.layer2_0_bn2(self.layer2_0_conv2(out))
		x = F.relu(out + self.layer2_0_downsample_1(self.layer2_0_downsample_0(x)))
		x = F.relu(out + x)
		out = F.relu(self.layer2_1_bn1(self.layer2_1_conv1(x)))
		out = self.layer2_1_bn2(self.layer2_1_conv2(out))
		x = F.relu(out + x)
		out = F.relu(self.layer2_2_bn1(self.layer2_2_conv1(x)))
		out = self.layer2_2_bn2(self.layer2_2_conv2(out))
		x = F.relu(out + x)
		out = F.relu(self.layer3_0_bn1(self.layer3_0_conv1(x)))
		out = self.layer3_0_bn2(self.layer3_0_conv2(out))
		x = F.relu(out + self.layer3_0_downsample_1(self.layer3_0_downsample_0(x)))
		x = F.relu(out + x)
		out = F.relu(self.layer3_1_bn1(self.layer3_1_conv1(x)))
		out = self.layer3_1_bn2(self.layer3_1_conv2(out))
		x = F.relu(out + x)
		out = F.relu(self.layer3_2_bn1(self.layer3_2_conv1(x)))
		out = self.layer3_2_bn2(self.layer3_2_conv2(out))
		x = F.relu(out + x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

def resnet(**kwargs):
	return ResNet(**kwargs)
