#!/usr/bin/env python
# coding: utf-8

import os
import time
import glob
import random
import numpy as np
from PIL import Image
import shutil

from typing import List, Tuple, Dict, AnyStr, KeysView, Any

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision.models import resnet

import config as CFG 
from utils import *
from ImageByFolderDataset import *

import matplotlib.pyplot as plt
from tqdm import tqdm


class ImageByFolderResNet(nn.Module):
	def __init__(self,resnet_type="resnet50"):
		super(ImageByFolderResNet, self).__init__()
		self.resnet_n = nn.Sequential(resnet.resnet50(pretrained=True),)
		if resnet_type.lower() == "resnet101":	
			self.resnet_n = nn.Sequential(resnet.resnet101(pretrained=True),)
		if resnet_type.lower() == "resnet152":	
			self.resnet_n = nn.Sequential(resnet.resnet152(pretrained=True),)

		self.flatten = nn.Flatten()
		self.linear0 = nn.Linear(1000,CFG.num_classes)
		self.relu = nn.ReLU()

		_layers = []
		_layers.append(self.resnet_n)
		_layers.append(self.flatten)
		_layers.append(self.linear0)
		_layers.append(self.relu)

		self.layers = _layers

	def forward(self,x):
		output = self.resnet_n(x)
		output = self.linear0(output)
		output = self.relu(output)
		return output

	def __len__(self):
		return len(self.layers)

	def __getitem__(self, item):
		return self.layers[item]	

	def __name__(self):
		return "ImageByFolderResNet"







# class ImageByFolderResNet(nn.Module):
# 	def __init__(self):
# 		super(ImageByFolderResNet, self).__init__()
# 		self.resnet50 = resnet.resnet50(pretrained=True)
# 		num_features = self.resnet50.fc.in_features
# 		for param in self.resnet50.parameters():
# 			param.requires_grad = False

# 		self.resnet50.fc = nn.Sequential(
# 			nn.Linear(num_features,CFG.num_classes),
# 			nn.LogSoftmax(dim=1)
# 			)

# 	def forward(self,x):
# 		return x



# 	def __len__(self):
# 		return len(self.layers)

# 	def __getitem__(self, item):
# 		return self.layers[item]	

# 	def __name__(self):
# 		return "ImageByFolderResNet"


