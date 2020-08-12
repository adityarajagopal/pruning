import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['alexnet']

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes) 
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool2d(x) 
        
        x = self.conv2(x) 
        x = self.relu(x)
        x = self.maxpool2d(x)
        
        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x) 
        x = self.maxpool2d(x)

        x = x.view(x.size(0), 256*6*6)

        x = self.dropout(x) 
        x = self.fc1(x) 
        x = self.relu(x)

        x = self.dropout(x)
        x = self.fc2(x) 
        x = self.relu(x)

        x = self.fc3(x)

        return x 


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

