import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()

