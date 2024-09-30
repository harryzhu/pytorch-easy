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

import config as CFG 
from utils import *
from ImageByFolderDataset import *

import matplotlib.pyplot as plt
from tqdm import tqdm

class ImageByFolderNet(nn.Module):
	def __init__(self):
		super(ImageByFolderNet, self).__init__()
		self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3,3))
		self.pool0 = nn.AvgPool2d(kernel_size=(3,3))
		self.conv1 = nn.Conv2d(8,24,kernel_size=(3,3))		
		self.pool1 = nn.AvgPool2d(kernel_size=(3,3))
		self.flatten = nn.Flatten()
		self.linear0 = nn.Linear(17496, 5200)
		self.linear1 = nn.Linear(5200,4000)
		self.linear2 = nn.Linear(4000,CFG.num_classes)
		self.relu = nn.ReLU()
		
		_layers = []
		_layers.append(self.conv0)
		_layers.append(self.pool0)
		_layers.append(self.conv1)
		_layers.append(self.pool1)
		_layers.append(self.flatten)
		_layers.append(self.linear0)
		_layers.append(self.relu)
		_layers.append(self.linear1)
		_layers.append(self.relu)
		_layers.append(self.linear2)
		_layers.append(self.relu)

		self.layers = _layers

	def forward(self,x):
		#return x
		#print("=== x ===",x.shape)
		output = self.conv0(x)
		output = self.pool0(output)
		#print("conv0:",output.shape)
		output = self.conv1(output)	
		output = self.pool1(output)
		#print("conv1:",output.shape)
		output = self.flatten(output)
		#print("flatten:",output.shape)
		# #print(output)
		output = self.linear0(output)
		#print("linear0:",output.shape)
		output = self.relu(output)
		output = self.linear1(output)
		#print("linear1:",output.shape)
		output = self.relu(output)
		output = self.linear2(output)
		#print("linear2:",output.shape)
		output = self.relu(output)
		#print("output:",output.shape)
		#print(output)
		return output


	def __len__(self):
		return len(self.layers)

	def __getitem__(self, item):
		return self.layers[item]	

	def __name__(self):
		return "ImageByFolderNet"


