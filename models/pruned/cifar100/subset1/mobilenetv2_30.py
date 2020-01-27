import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV2(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 27, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
		self.bn1 = nn.BatchNorm2d(27, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_0_conv1 = nn.Conv2d(27, 18, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_0_bn1 = nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_0_conv2 = nn.Conv2d(18, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=18, bias=False)
		self.layers_0_bn2 = nn.BatchNorm2d(18, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_0_conv3 = nn.Conv2d(18, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_0_bn3 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_0_shortcut_0 = nn.Conv2d(27, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_0_shortcut_1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_1_conv1 = nn.Conv2d(16, 63, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_1_bn1 = nn.BatchNorm2d(63, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_1_conv2 = nn.Conv2d(63, 63, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=63, bias=False)
		self.layers_1_bn2 = nn.BatchNorm2d(63, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_1_conv3 = nn.Conv2d(63, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_1_bn3 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_1_shortcut_0 = nn.Conv2d(16, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_1_shortcut_1 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_2_conv1 = nn.Conv2d(24, 92, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_2_bn1 = nn.BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_2_conv2 = nn.Conv2d(92, 92, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=92, bias=False)
		self.layers_2_bn2 = nn.BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_2_conv3 = nn.Conv2d(92, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_2_bn3 = nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_2_shortcut = nn.Sequential()
		self.layers_3_conv1 = nn.Conv2d(24, 122, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_3_bn1 = nn.BatchNorm2d(122, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_3_conv2 = nn.Conv2d(122, 122, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=122, bias=False)
		self.layers_3_bn2 = nn.BatchNorm2d(122, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_3_conv3 = nn.Conv2d(122, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_3_bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_4_conv1 = nn.Conv2d(32, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_4_bn1 = nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_4_conv2 = nn.Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
		self.layers_4_bn2 = nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_4_conv3 = nn.Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_4_bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_4_shortcut = nn.Sequential()
		self.layers_5_conv1 = nn.Conv2d(32, 141, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_5_bn1 = nn.BatchNorm2d(141, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_5_conv2 = nn.Conv2d(141, 141, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=141, bias=False)
		self.layers_5_bn2 = nn.BatchNorm2d(141, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_5_conv3 = nn.Conv2d(141, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_5_bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_5_shortcut = nn.Sequential()
		self.layers_6_conv1 = nn.Conv2d(32, 154, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_6_bn1 = nn.BatchNorm2d(154, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_6_conv2 = nn.Conv2d(154, 154, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=154, bias=False)
		self.layers_6_bn2 = nn.BatchNorm2d(154, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_6_conv3 = nn.Conv2d(154, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_6_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_7_conv1 = nn.Conv2d(64, 287, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_7_bn1 = nn.BatchNorm2d(287, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_7_conv2 = nn.Conv2d(287, 287, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=287, bias=False)
		self.layers_7_bn2 = nn.BatchNorm2d(287, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_7_conv3 = nn.Conv2d(287, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_7_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_7_shortcut = nn.Sequential()
		self.layers_8_conv1 = nn.Conv2d(64, 264, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_8_bn1 = nn.BatchNorm2d(264, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_8_conv2 = nn.Conv2d(264, 264, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=264, bias=False)
		self.layers_8_bn2 = nn.BatchNorm2d(264, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_8_conv3 = nn.Conv2d(264, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_8_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_8_shortcut = nn.Sequential()
		self.layers_9_conv1 = nn.Conv2d(64, 238, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_9_bn1 = nn.BatchNorm2d(238, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_9_conv2 = nn.Conv2d(238, 238, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=238, bias=False)
		self.layers_9_bn2 = nn.BatchNorm2d(238, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_9_conv3 = nn.Conv2d(238, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_9_bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_9_shortcut = nn.Sequential()
		self.layers_10_conv1 = nn.Conv2d(64, 276, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_10_bn1 = nn.BatchNorm2d(276, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_10_conv2 = nn.Conv2d(276, 276, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=276, bias=False)
		self.layers_10_bn2 = nn.BatchNorm2d(276, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_10_conv3 = nn.Conv2d(276, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_10_bn3 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_10_shortcut_0 = nn.Conv2d(64, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_10_shortcut_1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_11_conv1 = nn.Conv2d(96, 357, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_11_bn1 = nn.BatchNorm2d(357, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_11_conv2 = nn.Conv2d(357, 357, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=357, bias=False)
		self.layers_11_bn2 = nn.BatchNorm2d(357, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_11_conv3 = nn.Conv2d(357, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_11_bn3 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_11_shortcut = nn.Sequential()
		self.layers_12_conv1 = nn.Conv2d(96, 377, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_12_bn1 = nn.BatchNorm2d(377, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_12_conv2 = nn.Conv2d(377, 377, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=377, bias=False)
		self.layers_12_bn2 = nn.BatchNorm2d(377, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_12_conv3 = nn.Conv2d(377, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_12_bn3 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_12_shortcut = nn.Sequential()
		self.layers_13_conv1 = nn.Conv2d(96, 366, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_13_bn1 = nn.BatchNorm2d(366, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_13_conv2 = nn.Conv2d(366, 366, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=366, bias=False)
		self.layers_13_bn2 = nn.BatchNorm2d(366, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_13_conv3 = nn.Conv2d(366, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_13_bn3 = nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_14_conv1 = nn.Conv2d(160, 648, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_14_bn1 = nn.BatchNorm2d(648, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_14_conv2 = nn.Conv2d(648, 648, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=648, bias=False)
		self.layers_14_bn2 = nn.BatchNorm2d(648, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_14_conv3 = nn.Conv2d(648, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_14_bn3 = nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_14_shortcut = nn.Sequential()
		self.layers_15_conv1 = nn.Conv2d(160, 578, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_15_bn1 = nn.BatchNorm2d(578, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_15_conv2 = nn.Conv2d(578, 578, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=578, bias=False)
		self.layers_15_bn2 = nn.BatchNorm2d(578, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_15_conv3 = nn.Conv2d(578, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_15_bn3 = nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_15_shortcut = nn.Sequential()
		self.layers_16_conv1 = nn.Conv2d(160, 472, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_16_bn1 = nn.BatchNorm2d(472, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_16_conv2 = nn.Conv2d(472, 472, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=472, bias=False)
		self.layers_16_bn2 = nn.BatchNorm2d(472, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_16_conv3 = nn.Conv2d(472, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_16_bn3 = nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.layers_16_shortcut_0 = nn.Conv2d(160, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.layers_16_shortcut_1 = nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.conv2 = nn.Conv2d(320, 1261, kernel_size=(1, 1), stride=(1, 1), bias=False)
		self.bn2 = nn.BatchNorm2d(1261, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.linear = nn.Linear(in_features=1261, out_features=100, bias=True)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.layers_0_bn1(self.layers_0_conv1(x)))
		out = F.relu(self.layers_0_bn2(self.layers_0_conv2(out)))
		out = self.layers_0_bn3(self.layers_0_conv3(out))
		x = out + self.layers_0_shortcut_1(self.layers_0_shortcut_0(x))
		out = F.relu(self.layers_1_bn1(self.layers_1_conv1(x)))
		out = F.relu(self.layers_1_bn2(self.layers_1_conv2(out)))
		out = self.layers_1_bn3(self.layers_1_conv3(out))
		x = out + self.layers_1_shortcut_1(self.layers_1_shortcut_0(x))
		out = F.relu(self.layers_2_bn1(self.layers_2_conv1(x)))
		out = F.relu(self.layers_2_bn2(self.layers_2_conv2(out)))
		out = self.layers_2_bn3(self.layers_2_conv3(out))
		x = out + self.layers_2_shortcut(x)
		out = F.relu(self.layers_3_bn1(self.layers_3_conv1(x)))
		out = F.relu(self.layers_3_bn2(self.layers_3_conv2(out)))
		out = self.layers_3_bn3(self.layers_3_conv3(out))
		x = out
		out = F.relu(self.layers_4_bn1(self.layers_4_conv1(x)))
		out = F.relu(self.layers_4_bn2(self.layers_4_conv2(out)))
		out = self.layers_4_bn3(self.layers_4_conv3(out))
		x = out + self.layers_4_shortcut(x)
		out = F.relu(self.layers_5_bn1(self.layers_5_conv1(x)))
		out = F.relu(self.layers_5_bn2(self.layers_5_conv2(out)))
		out = self.layers_5_bn3(self.layers_5_conv3(out))
		x = out + self.layers_5_shortcut(x)
		out = F.relu(self.layers_6_bn1(self.layers_6_conv1(x)))
		out = F.relu(self.layers_6_bn2(self.layers_6_conv2(out)))
		out = self.layers_6_bn3(self.layers_6_conv3(out))
		x = out
		out = F.relu(self.layers_7_bn1(self.layers_7_conv1(x)))
		out = F.relu(self.layers_7_bn2(self.layers_7_conv2(out)))
		out = self.layers_7_bn3(self.layers_7_conv3(out))
		x = out + self.layers_7_shortcut(x)
		out = F.relu(self.layers_8_bn1(self.layers_8_conv1(x)))
		out = F.relu(self.layers_8_bn2(self.layers_8_conv2(out)))
		out = self.layers_8_bn3(self.layers_8_conv3(out))
		x = out + self.layers_8_shortcut(x)
		out = F.relu(self.layers_9_bn1(self.layers_9_conv1(x)))
		out = F.relu(self.layers_9_bn2(self.layers_9_conv2(out)))
		out = self.layers_9_bn3(self.layers_9_conv3(out))
		x = out + self.layers_9_shortcut(x)
		out = F.relu(self.layers_10_bn1(self.layers_10_conv1(x)))
		out = F.relu(self.layers_10_bn2(self.layers_10_conv2(out)))
		out = self.layers_10_bn3(self.layers_10_conv3(out))
		x = out + self.layers_10_shortcut_1(self.layers_10_shortcut_0(x))
		out = F.relu(self.layers_11_bn1(self.layers_11_conv1(x)))
		out = F.relu(self.layers_11_bn2(self.layers_11_conv2(out)))
		out = self.layers_11_bn3(self.layers_11_conv3(out))
		x = out + self.layers_11_shortcut(x)
		out = F.relu(self.layers_12_bn1(self.layers_12_conv1(x)))
		out = F.relu(self.layers_12_bn2(self.layers_12_conv2(out)))
		out = self.layers_12_bn3(self.layers_12_conv3(out))
		x = out + self.layers_12_shortcut(x)
		out = F.relu(self.layers_13_bn1(self.layers_13_conv1(x)))
		out = F.relu(self.layers_13_bn2(self.layers_13_conv2(out)))
		out = self.layers_13_bn3(self.layers_13_conv3(out))
		x = out
		out = F.relu(self.layers_14_bn1(self.layers_14_conv1(x)))
		out = F.relu(self.layers_14_bn2(self.layers_14_conv2(out)))
		out = self.layers_14_bn3(self.layers_14_conv3(out))
		x = out + self.layers_14_shortcut(x)
		out = F.relu(self.layers_15_bn1(self.layers_15_conv1(x)))
		out = F.relu(self.layers_15_bn2(self.layers_15_conv2(out)))
		out = self.layers_15_bn3(self.layers_15_conv3(out))
		x = out + self.layers_15_shortcut(x)
		out = F.relu(self.layers_16_bn1(self.layers_16_conv1(x)))
		out = F.relu(self.layers_16_bn2(self.layers_16_conv2(out)))
		out = self.layers_16_bn3(self.layers_16_conv3(out))
		x = out + self.layers_16_shortcut_1(self.layers_16_shortcut_0(x))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.avg_pool2d(x,4)
		x = x.view(x.size(0), -1)
		x = self.linear(x)
		return x

def mobilenetv2(**kwargs):
	return MobileNetV2(**kwargs)