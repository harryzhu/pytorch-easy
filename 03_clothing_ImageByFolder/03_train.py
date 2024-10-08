#!/usr/bin/env python
# coding: utf-8

import os
import time
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

import config as CFG
from utils import *
from ImageByFolderDataset import *
from ImageByFolderNet import *
from ImageByFolderResNet import *
from ImageByFolderGoogLeNet import *
from ImageByFolderAlexNet import *
from ImageByFolderEfficientNet import *
from ImageByFolderMobileNet import *
from ImageByFolderVggNet import *

#model = ImageByFolderNet()

#model = ImageByFolderResNet()
#model = ImageByFolderResNet("resnet50")
#model = ImageByFolderResNet("resnet101")
#model = ImageByFolderResNet("resnet152")

#model = ImageByFolderGoogLeNet()
#model = ImageByFolderAlexNet()
#model = ImageByFolderEfficientNet()
model = ImageByFolderMobileNet()
#model = ImageByFolderVggNet()

model_name = get_model_name(model)
model_output_dir = f'output/{model_name}'
print(model_output_dir)
if not os.path.isdir(model_output_dir):
	os.makedirs(model_output_dir)

if model_name == "ImageByFolderNet":
	num_epoch = 300
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

if model_name == "ImageByFolderResNet":
	num_epoch = 100
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

if model_name == "ImageByFolderGoogLeNet":
	num_epoch = 300
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if model_name == "ImageByFolderAlexNet":
	num_epoch = 300
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

if model_name == "ImageByFolderEfficientNet":
	num_epoch = 300
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if model_name == "ImageByFolderMobileNet":
	num_epoch = 100
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if model_name == "ImageByFolderVggNet":
	num_epoch = 300
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


accumulator = train(model, loss_fn, dataloader_train, optimizer, num_epoch, model_output_dir)
print("------------------")
#print(accumulator)

