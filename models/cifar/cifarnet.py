import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['cifarnet']

class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.gconv0 = nn.Conv2d(3, 64, padding=0, kernel_size=3)
        self.gconv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.gconv2 = nn.Conv2d(64, 128, stride=2, kernel_size=3, padding=1)
        self.gconv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.drop3 = nn.Dropout2d()
        self.gconv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.gconv5 = nn.Conv2d(128, 192, stride=2, kernel_size=3, padding=1)
        self.gconv6 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.drop6 = nn.Dropout2d()
        self.gconv7 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.gconv0(x)
        x = self.gconv1(x)
        x = self.gconv2(x)
        x = self.gconv3(x)
        # x = self.drop3(x)
        x = self.gconv4(x)
        x = self.gconv5(x)
        x = self.gconv6(x)
        # x = self.drop6(x)
        x = self.gconv7(x)
        x = self.pool(x)
        x = x.view(-1, 192)
        x = self.fc(x)

        return  x

    def gated_forward(self, x):
        x = self.gconv0(x)
        x = self.gconv0(x)
        x = self.gconv1(x)
        x = self.gconv2(x)
        x = self.gconv3(x)
        x = self.drop3(x)
        x = self.gconv4(x)
        x = self.gconv5(x)
        x = self.gconv6(x)
        x = self.drop6(x)
        x = self.gconv7(x)
        x = self.pool(x)
        x = x.view(-1, 192)
        x = self.fc(x)

        return  x

def cifarnet(**kwargs):
    return CifarNet(**kwargs)
