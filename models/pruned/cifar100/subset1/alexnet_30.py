import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(5, 5))
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(64, 188, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv3 = nn.Conv2d(188, 289, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(289, 186, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(186, 252, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2d_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.classifier = nn.Linear(in_features=252, out_features=100, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool2d_1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2d_2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxpool2d_3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def alexnet(**kwargs):
    return AlexNet(**kwargs)
