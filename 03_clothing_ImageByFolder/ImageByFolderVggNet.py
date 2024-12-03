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
from torchvision.models import vgg

import config as CFG 
from utils import *
from ImageByFolderDataset import *

import matplotlib.pyplot as plt
from tqdm import tqdm


class ImageByFolderVggNet(nn.Module):
	def __init__(self,vgg_type="vgg16"):
		super(ImageByFolderVggNet, self).__init__()
		self.vgg_n = nn.Sequential(vgg.vgg16(pretrained=True),)
		if vgg_type.lower() == "vgg11":	
			self.vgg_n = nn.Sequential(vgg.vgg11(pretrained=True),)
		if vgg_type.lower() == "vgg13":	
			self.vgg_n = nn.Sequential(vgg.vgg13(pretrained=True),)
		if vgg_type.lower() == "vgg19":	
			self.vgg_n = nn.Sequential(vgg.vgg19(pretrained=True),)

		self.flatten = nn.Flatten()
		self.linear0 = nn.Linear(1000,CFG.num_classes)
		self.relu = nn.ReLU()

		_layers = []
		_layers.append(self.vgg_n)
		_layers.append(self.flatten)
		_layers.append(self.linear0)
		_layers.append(self.relu)

		self.layers = _layers

	def forward(self,x):
		output = self.vgg_n(x)
		output = self.linear0(output)
		output = self.relu(output)
		return output

	def __len__(self):
		return len(self.layers)

	def __getitem__(self, item):
		return self.layers[item]	

	def __name__(self):
		return "ImageByFolderVggNet"






