import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(5, 5))
		self.conv2 = nn.Conv2d(64, 183, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
		self.conv3 = nn.Conv2d(183, 179, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.conv4 = nn.Conv2d(179, 87, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.conv5 = nn.Conv2d(87, 244, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		self.classifier = nn.Linear(in_features=244, out_features=100, bias=True)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.maxpool2d(x)
		x = F.relu(self.conv2(x))
		x = self.maxpool2d(x)
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = self.maxpool2d(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

def alexnet(**kwargs):
	return AlexNet(**kwargs)
