import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet20(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 11, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.bn1 = nn.BatchNorm2d(11, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_0_conv1 = nn.Conv2d(11, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_0_conv2 = nn.Conv2d(12, 11, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(11, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_1_conv1 = nn.Conv2d(11, 15, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_1_conv2 = nn.Conv2d(15, 11, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(11, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_2_conv1 = nn.Conv2d(11, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer1_2_conv2 = nn.Conv2d(16, 11, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(11, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_0_conv1 = nn.Conv2d(11, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_0_conv2 = nn.Conv2d(32, 28, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_0_downsample_0 = nn.Conv2d(11, 28, kernel_size=(1, 1), stride=(2, 2), bias=False)
		self.layer2_0_downsample_1 = nn.BatchNorm2d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_1_conv1 = nn.Conv2d(28, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_1_conv2 = nn.Conv2d(32, 28, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_2_conv1 = nn.Conv2d(28, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer2_2_conv2 = nn.Conv2d(32, 28, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_0_conv1 = nn.Conv2d(28, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_0_conv2 = nn.Conv2d(64, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_0_downsample_0 = nn.Conv2d(28, 40, kernel_size=(1, 1), stride=(2, 2), bias=False)
		self.layer3_0_downsample_1 = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_1_conv1 = nn.Conv2d(40, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_1_conv2 = nn.Conv2d(64, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_2_conv1 = nn.Conv2d(40, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layer3_2_conv2 = nn.Conv2d(64, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self. = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
		self.fc = nn.Linear(in_features=40, out_features=100, bias=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		out = x
		out = self.layer1_0_conv1(out)
		out = self.layer1_0_bn1(out)
		out = F.relu(out)
		out = self.layer1_0_conv2(out)
		out = self.layer1_0_bn2(out)
		out_res = x
		x = F.relu(out + out_res)
		out = x
		out = self.layer1_1_conv1(out)
		out = self.layer1_1_bn1(out)
		out = F.relu(out)
		out = self.layer1_1_conv2(out)
		out = self.layer1_1_bn2(out)
		out_res = x
		x = F.relu(out + out_res)
		out = x
		out = self.layer1_2_conv1(out)
		out = self.layer1_2_bn1(out)
		out = F.relu(out)
		out = self.layer1_2_conv2(out)
		out = self.layer1_2_bn2(out)
		out_res = x
		x = F.relu(out + out_res)
		out = x
		out = self.layer2_0_conv1(out)
		out = self.layer2_0_bn1(out)
		out = F.relu(out)
		out = self.layer2_0_conv2(out)
		out = self.layer2_0_bn2(out)
		out_res = x
		out_res = self.layer2_0_downsample_0(out_res)
		out_res = self.layer2_0_downsample_1(out_res)
		x = F.relu(out + out_res)
		out = x
		out = self.layer2_1_conv1(out)
		out = self.layer2_1_bn1(out)
		out = F.relu(out)
		out = self.layer2_1_conv2(out)
		out = self.layer2_1_bn2(out)
		out_res = x
		x = F.relu(out + out_res)
		out = x
		out = self.layer2_2_conv1(out)
		out = self.layer2_2_bn1(out)
		out = F.relu(out)
		out = self.layer2_2_conv2(out)
		out = self.layer2_2_bn2(out)
		out_res = x
		x = F.relu(out + out_res)
		out = x
		out = self.layer3_0_conv1(out)
		out = self.layer3_0_bn1(out)
		out = F.relu(out)
		out = self.layer3_0_conv2(out)
		out = self.layer3_0_bn2(out)
		out_res = x
		out_res = self.layer3_0_downsample_0(out_res)
		out_res = self.layer3_0_downsample_1(out_res)
		x = F.relu(out + out_res)
		out = x
		out = self.layer3_1_conv1(out)
		out = self.layer3_1_bn1(out)
		out = F.relu(out)
		out = self.layer3_1_conv2(out)
		out = self.layer3_1_bn2(out)
		out_res = x
		x = F.relu(out + out_res)
		out = x
		out = self.layer3_2_conv1(out)
		out = self.layer3_2_bn1(out)
		out = F.relu(out)
		out = self.layer3_2_conv2(out)
		out = self.layer3_2_bn2(out)
		out_res = x
		x = F.relu(out + out_res)
		x = self.avgpool(x)
		x = self.fc(x)
