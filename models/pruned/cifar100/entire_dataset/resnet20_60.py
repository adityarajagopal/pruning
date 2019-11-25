import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet20(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.bn1 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_0_conv1 = nn.Conv2d(7, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_0_bn1 = nn.BatchNorm2d(14, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_0_conv2 = nn.Conv2d(14, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_0_bn2 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_1_conv1 = nn.Conv2d(7, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_1_bn1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_1_conv2 = nn.Conv2d(16, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_1_bn2 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_2_conv1 = nn.Conv2d(7, 15, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_2_bn1 = nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_2_conv2 = nn.Conv2d(15, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer1_2_bn2 = nn.BatchNorm2d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_0_conv1 = nn.Conv2d(7, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
		self.layer2_0_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_0_conv2 = nn.Conv2d(32, 15, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer2_0_bn2 = nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_0_downsample_0 = nn.Conv2d(7, 15, kernel_size=(1, 1), stride=(2, 2), bias=False)
		self.layer2_0_downsample_1 = nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_1_conv1 = nn.Conv2d(15, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer2_1_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_1_conv2 = nn.Conv2d(32, 15, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer2_1_bn2 = nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_2_conv1 = nn.Conv2d(15, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer2_2_bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_2_conv2 = nn.Conv2d(32, 15, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer2_2_bn2 = nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_0_conv1 = nn.Conv2d(15, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
		self.layer3_0_bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_0_conv2 = nn.Conv2d(64, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer3_0_bn2 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_0_downsample_0 = nn.Conv2d(15, 26, kernel_size=(1, 1), stride=(2, 2), bias=False)
		self.layer3_0_downsample_1 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_1_conv1 = nn.Conv2d(26, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer3_1_bn1 = nn.BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_1_conv2 = nn.Conv2d(58, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer3_1_bn2 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_2_conv1 = nn.Conv2d(26, 49, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer3_2_bn1 = nn.BatchNorm2d(49, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_2_conv2 = nn.Conv2d(49, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.layer3_2_bn2 = nn.BatchNorm2d(26, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
		self.fc = nn.Linear(in_features=26, out_features=100, bias=True)

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
